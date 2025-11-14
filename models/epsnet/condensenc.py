import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.naive_bayes import logsumexp

import itertools

from utils.chem import BOND_TYPES
from utils import activation_loader
from utils.align_mol import kabsch_algorithm, weighted_kabsch_algorithm
from utils.atom_matching import sinkhorn_algorithm
from utils.gumbel_softmax_t import TemperatureScheduler
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_ts_graph_order_radius
from ..encoder import get_edge_encoder, load_encoder
from ..geometry import get_distance, eq_transform


def gumbel_softmax_hard(soft_matrix, temperature=1.0):
    """
    Convert soft matrix to hard matching using Gumbel-Softmax with improved numerical stability
    
    Args:
        soft_matrix (torch.Tensor): Soft assignment matrix
        temperature (float): Temperature for Gumbel-Softmax (lower = harder)
    
    Returns:
        torch.Tensor: Hard binary matching matrix
    """
    eps = 1e-10  # Small epsilon for numerical stability

    # Clamp input values to prevent log(0)
    soft_matrix = torch.clamp(soft_matrix, min=eps, max=1.0)

    # Add Gumbel noise with numerical stability
    U = torch.rand_like(soft_matrix)
    U = torch.clamp(U, min=eps, max=1.0)
    gumbel_noise = -torch.log(-torch.log(U))

    # Apply Gumbel-Softmax with stable log computation
    logits = torch.log(soft_matrix)
    noisy_logits = (logits + gumbel_noise) / temperature

    # Subtract max for numerical stability in softmax
    noisy_logits_max = torch.max(noisy_logits, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(noisy_logits - noisy_logits_max)
    row_softmax = exp_logits / (torch.sum(exp_logits, dim=1, keepdim=True) + eps)

    # # In forward pass, convert to hard assignment
    # indices = torch.argmax(row_softmax, dim=1)
    # hard_matrix = torch.zeros_like(soft_matrix)
    # hard_matrix.scatter_(1, indices.unsqueeze(1), 1)

    # # Straight-through estimator: use hard values in forward pass
    # # but gradients flow through soft values
    # hard_matrix = (hard_matrix - row_softmax).detach() + row_softmax

    return row_softmax

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas




class CondenseEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder = get_edge_encoder(config)
        assert config.hidden_dim % 2 == 0
        self.atom_embedding = nn.Embedding(100, config.hidden_dim)

        self.atom_feat_embedding = nn.Linear(
            config.feat_dim, config.hidden_dim, bias=False
        )


        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder = load_encoder(config, "encoder")

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim, 1],
            activation=activation_loader(config.mlp_act),
        )

        if config.parallel_training.enable:
            self.scaler_mlp = MultiLayerPerceptron(
                config.hidden_dim,
                [config.hidden_dim, config.hidden_dim, 1],
                activation=activation_loader(config.parallel_training.model.mlp_act_of_scaler),
            )

        
        """
        Incorporate parameters together
        """
        self.model_embedding = nn.ModuleList(
            [
                self.atom_embedding,
                self.atom_feat_embedding,
            ]
        )
        self.model = nn.ModuleList(
            [self.edge_encoder, self.encoder, self.grad_dist_mlp]
        )

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        # variances
        alphas = (1.0 - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        self.num_bond_types = len(BOND_TYPES)
        self.edge_cat = torch.nn.Sequential(
            torch.nn.Linear(
                self.edge_encoder.out_channels,
                self.edge_encoder.out_channels,
            ),
            activation_loader(config.edge_cat_act),
            torch.nn.Linear(
                self.edge_encoder.out_channels,
                self.edge_encoder.out_channels,
            ),
        )

    def _extend_condensed_graph_edge(self, pos, bond_index, bond_type, batch, **kwargs):
        N = pos.size(0)
        cutoff = kwargs.get("cutoff", self.config.edge_cutoff)
        edge_order = kwargs.get("edge_order", self.config.edge_order)

        _g_ext = extend_ts_graph_order_radius
        out = _g_ext(
            N, pos, bond_index, bond_type, batch, order=edge_order, cutoff=cutoff
        )
        edge_index_global, edge_index_local, edge_type= out
        # local index             : (i, j) pairs which are edge of R or P.
        # edge_type_r/edge_type_p : 0, 1, 2, ... 23, 24, ...
        #                           0 -> no edge (bond)
        #                           1, 2, 3 ..-> bond type
        #                           23, 24 -> meaning no bond, but higher order edge. (2-hop or 3-hop)
        # global index            : atom pairs (i, j) which are closer than cutoff
        #                           are added to local_index.
        #

        edge_type_global = torch.zeros_like(edge_index_global[0]) - 1
        adj_global = to_dense_adj(
            edge_index_global, edge_attr=edge_type_global, max_num_nodes=N
        )
        adj_local = to_dense_adj(
            edge_index_local, edge_attr=edge_type, max_num_nodes=N
        )

        adj_global = torch.where(adj_local != 0, adj_local, adj_global)

        edge_index_global, edge_type_global = dense_to_sparse(adj_global)

        edge_type_global[edge_type_global < 0] = 0
        edge_index_global = edge_index_global

        return edge_index_global, edge_index_local, edge_type_global

    def _condensed_edge_embedding(self, edge_length, edge_type, edge_attr=None, emb_type="bond_w_d"):

        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]
        _enc = self.edge_encoder
        _cat_fn = self.edge_cat

        if emb_type == "bond_wo_d":
            edge_attr = _enc.bond_emb(edge_type)
            edge_attr = _cat_fn(edge_attr) 

        elif emb_type == "bond_w_d":
            edge_attr = _enc(edge_length, edge_type)  # Embed edges
            edge_attr = _cat_fn(edge_attr) 

        elif emb_type == "add_d":
            edge_attr = _enc.mlp(edge_length) * edge_attr

        return edge_attr

    def forward_(self, atom_type, feat, pos, bond_index, bond_type, batch, mode, **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        _g_ext = self._extend_condensed_graph_edge
        _e_emb = self._condensed_edge_embedding
        _a_emb = self.atom_embedding
        _af_emb = self.atom_feat_embedding
        _enc = self.encoder

        # condensed atom embedding
        a_emb = _a_emb(atom_type)
        # print(11111111)
        # print(np.shape(atom_type))
        # print(np.shape(feat))
        af_emb = _af_emb(feat.float())
        z1 = a_emb + af_emb
        # z = torch.cat([z1, z1], dim=-1)
        z=z1

        # edge extension
        edge_index, _, edge_type = _g_ext(
            pos,
            bond_index,
            bond_type,
            batch,
        )
        edge_length = get_distance(pos, edge_index).unsqueeze(-1) # (E, 1)

        # edge embedding
        edge_attr = _e_emb(
            edge_length,
            edge_type
        )

        # encoding geometric graph and atom-pair
        node_attr = _enc(z, edge_index, edge_length, edge_attr=edge_attr)

        edge_ord4inp = self.config.edge_order
        edge_ord4out = self.config.pred_edge_order
        if edge_ord4inp != edge_ord4out:
            edge_index, _, edge_type = _g_ext(
                pos,
                bond_index,
                bond_type,
                batch,
                edge_order=edge_ord4out,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
            edge_attr = _e_emb(
                edge_length,
                edge_type
            )

        if mode == "edge":
            h_pair = assemble_atom_pair_feature(node_attr, edge_index, edge_attr)  # (E, 2H)
            edge_inv = self.grad_dist_mlp(h_pair)  # (E, 1)
            return edge_inv, edge_index, edge_length
        elif mode == "scaler":
            # Aggregate node features to graph-level features
            graph_attr = global_mean_pool(node_attr, batch)  # (G, H)
            scaler_inv = self.scaler_mlp(graph_attr)  # (G, 1)
            return scaler_inv
        else:
            raise ValueError(f"Invalid option: {mode}. Supported options are 'edge' and 'scaler'.")

    def forward(self, atom_type, feat, pos, bond_index, bond_type, batch,
                time_step, mode, return_edges=True,  **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        out = self.forward_(
            atom_type,
            feat,
            pos,
            bond_index,
            bond_type,
            batch,
            mode,
        )
        
        if mode == "edge":
            edge_inv, edge_index, edge_length = out

            if return_edges:
                return edge_inv, edge_index, edge_length
            else:
                return edge_inv
        elif mode == "scaler":
            edge_inv = out
            return edge_inv

    @staticmethod
    def softmin_loss(loss1, loss2, gamma=1.0):
        """
        Compute a differentiable minimum between two element-wise losses using softmin.
        :param loss1: Tensor of losses for the original positions
        :param loss2: Tensor of losses for the mirrored positions
        :param gamma: Smoothing factor for softmin
        :return: Element-wise differentiable minimum loss
        """
        stacked_losses = torch.stack([-loss1, -loss2], dim=0)  # Shape: (2, N)
        weights = torch.softmax(stacked_losses / gamma, dim=0)  # Compute softmin weights
        softmin_result = weights[0] * loss1 + weights[1] * loss2  # Element-wise weighted sum
        return softmin_result
    
    
    def get_loss(
        self,
        atom_type,
        feat,
        pos,
        aromatic_rings,
        force_planarity, 
        bond_index,
        bond_type,
        batch,
        num_nodes_per_graph,
        num_graphs,
        extend_order=True,
        extend_radius=True,
        lambda_gp=0,
        sinkhorn_knopp=None,
        mode = "edge",
        prop = None,
        hydrogen_weight_method=1.0
        
    ):
        sinkhorn_knopp_enable=sinkhorn_knopp.enable
        sinkhorn_loss=sinkhorn_knopp.sinkhorn_loss
        kabsch=sinkhorn_knopp.kabsch.enable
        kabsch_ratio=sinkhorn_knopp.kabsch.time_ratio
        time_ratio=sinkhorn_knopp.time_ratio
        epsilon=sinkhorn_knopp.epsilon

        if mode == "edge":
            loss = self.get_loss_edge(
                atom_type,
                feat,
                pos,
                bond_index,
                bond_type,
                batch,
                num_nodes_per_graph,
                num_graphs,
                aromatic_rings=aromatic_rings,
                force_planarity=force_planarity,
                extend_order=extend_order,
                extend_radius=extend_radius,
                lambda_gp=lambda_gp,
                sinkhorn_knopp=sinkhorn_knopp_enable,
                sinkhorn_loss=sinkhorn_loss,
                kabsch=kabsch,
                kabsch_ratio=kabsch_ratio,
                time_ratio=time_ratio,
                epsilon=epsilon,
                hydrogen_weight_method=hydrogen_weight_method,
            )
        elif mode == "scaler":
            loss = self.get_loss_scaler(
                atom_type,
                feat,
                pos,
                bond_index,
                bond_type,
                batch,
                num_graphs,
                prop,
                extend_order=True,
                extend_radius=True,
            )
        return loss

    def get_loss_edge(
        self,
        atom_type,
        feat,
        pos,
        bond_index,
        bond_type,
        batch,
        num_nodes_per_graph,
        num_graphs,
        aromatic_rings=None,
        force_planarity=None,
        extend_order=True,
        extend_radius=True,
        lambda_gp=0,
        sinkhorn_knopp=False,
        sinkhorn_loss='cost_matrix',
        kabsch=True,
        kabsch_ratio=0.1,
        time_ratio=0.25,
        epsilon=1,
        hydrogen_weight_method=1.0,
    ):
        node2graph = batch
        assert atom_type.size(0) == node2graph.size(0), \
        f"Mismatch in atom_type and node2graph sizes: {atom_type.size(0)} != {node2graph.size(0)}"

        dev = pos.device
        # set time step and noise level
        t0 = self.config.get("t0", 0)
        t1 = self.config.get("t1", self.num_timesteps)

        # Vectorized time step generation
        sz = num_graphs // 2 + 1
        half_1 = torch.randint(t0, t1, size=(sz,), device=dev)
        half_2 = t0 + t1 - 1 - half_1
        time_step = torch.cat([half_1, half_2], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)  # (G, )

        # Perturb pos - vectorized
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        pos_noise = torch.randn(size=pos.size(), device=dev)
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        # prediction
        edge_inv, edge_index, edge_length = self(
            atom_type, feat, pos_perturbed, bond_index, bond_type,
            batch, time_step, return_edges=True, extend_order=extend_order,
            extend_radius=extend_radius, 
            mode = "edge",
        )  # (E, 1)
        
        node_eq = eq_transform(
            edge_inv, pos_perturbed, edge_index, edge_length
        )  # chain rule (re-parametrization, distance to position)

        # setup for target - vectorized
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # compute distances - vectorized
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_target = (d_gt - edge_length) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        pos_target = eq_transform(
            d_target, pos_perturbed, edge_index, edge_length
        )

        # Pre-compute thresholds and masks
        threshold_kabsch = int(t1 * kabsch_ratio)
        threshold_sinkhorn = int(t1 * time_ratio)
        time_mask_kabsch = time_step < threshold_kabsch
        time_mask_sinkhorn = time_step < threshold_sinkhorn

        # Initialize temperature scheduler outside the loop
        temperature_scheduler = TemperatureScheduler(
            initial_temperature=0.5,
            min_temperature=0.05,
            max_steps=int(threshold_kabsch),
            strategy="linear"
        )

        # Initialize node losses with zeros
        node_losses = torch.zeros(node2graph.size(0), 1, device=dev)

        if not sinkhorn_knopp:
            # Simple MSE loss for non-sinkhorn case
            node_losses = torch.sum((node_eq - pos_target) ** 2, dim=-1, keepdim=True)
        else:
            # Pre-compute graph masks for all graphs
            graph_masks = [(node2graph == i).nonzero(as_tuple=True)[0] for i in range(num_graphs)]
            
            # Process each graph based on its time step
            for graph_id in range(num_graphs):
                graph_mask = graph_masks[graph_id]
                pos_pred = node_eq[graph_mask]
                pos_tgt = pos_target[graph_mask]
                curr_atom_types = atom_type[graph_mask]
                curr_temperature = temperature_scheduler.get_temperature(time_step[graph_id])
                
                if time_mask_kabsch[graph_id] and kabsch:
                    # First create mirrored version of the entire molecule
                    pos_pred_mirror = pos_pred * torch.tensor([1, 1, -1], device=dev)
                    
                    # Initialize matched positions for both original and mirrored
                    matched_pos_pred = torch.zeros_like(pos_pred)
                    matched_pos_pred_mirror = torch.zeros_like(pos_pred)
                    
                    # Process each atom type separately for matching
                    unique_types = torch.unique(curr_atom_types)
                    for atomtype in unique_types:
                        type_mask = curr_atom_types == atomtype
                        if not type_mask.any():
                            continue
                            
                        pos_pred_type = pos_pred[type_mask]
                        pos_tgt_type = pos_tgt[type_mask]
                        pos_pred_mirror_type = pos_pred_mirror[type_mask]
                        
                        # Compute cost matrices and transport matrices
                        cost_original = torch.cdist(pos_pred_type, pos_tgt_type, p=2) + 1e-7
                        cost_mirror = torch.cdist(pos_pred_mirror_type, pos_tgt_type, p=2) + 1e-7
                        
                        # Apply Sinkhorn and gumbel_softmax_hard
                        transport_original = gumbel_softmax_hard(
                            sinkhorn_algorithm(cost_original, epsilon=epsilon),
                            temperature=curr_temperature
                        )
                        transport_mirror = gumbel_softmax_hard(
                            sinkhorn_algorithm(cost_mirror, epsilon=epsilon),
                            temperature=curr_temperature
                        )
                        
                        # Store matched positions
                        matched_pos_pred[type_mask] = torch.matmul(transport_original, pos_tgt_type)
                        matched_pos_pred_mirror[type_mask] = torch.matmul(transport_mirror, pos_tgt_type)
                    
                    # Apply Kabsch algorithm to the entire molecule
                    weights = torch.ones(pos_tgt.size(0), 1, device=dev)
                    weights[curr_atom_types == 1] = 0.75  # Lower weight for hydrogens
                    
                    R_original, t_original = weighted_kabsch_algorithm(matched_pos_pred, pos_tgt, weights)
                    R_mirror, t_mirror = weighted_kabsch_algorithm(matched_pos_pred_mirror, pos_tgt, weights)
                    
                    aligned_original = torch.matmul(matched_pos_pred, R_original.T) + t_original
                    aligned_mirror = torch.matmul(matched_pos_pred_mirror, R_mirror.T) + t_mirror
                    
                    loss_original = torch.sum((aligned_original - pos_tgt) ** 2, dim=1, keepdim=True)
                    loss_mirror = torch.sum((aligned_mirror - pos_tgt) ** 2, dim=1, keepdim=True)
                    
                    node_losses[graph_mask] = -torch.logsumexp(
                        torch.stack([-loss_original, -loss_mirror], dim=0),
                        dim=0
                    )
                    
                elif time_mask_sinkhorn[graph_id]:
                    # First create mirrored version of the entire molecule
                    pos_pred_mirror = pos_pred * torch.tensor([1, 1, -1], device=dev)
                    
                    # Sinkhorn weighted loss with pre-computed unique types
                    unique_types = torch.unique(curr_atom_types)
                    total_loss_original = torch.zeros_like(pos_pred[:, 0:1])
                    total_loss_mirror = torch.zeros_like(pos_pred[:, 0:1])
                    
                    for atomtype in unique_types:
                        type_mask = curr_atom_types == atomtype
                        if not type_mask.any():
                            continue
                            
                        pos_pred_type = pos_pred[type_mask]
                        pos_tgt_type = pos_tgt[type_mask]
                        pos_pred_mirror_type = pos_pred_mirror[type_mask]
                        
                        # Compute and clamp cost matrices
                        cost_original = torch.cdist(pos_pred_type, pos_tgt_type, p=2) + 1e-7
                        cost_mirror = torch.cdist(pos_pred_mirror_type, pos_tgt_type, p=2) + 1e-7
                        
                        # Apply Sinkhorn algorithm
                        match_original = sinkhorn_algorithm(cost_original, epsilon=epsilon)
                        match_mirror = sinkhorn_algorithm(cost_mirror, epsilon=epsilon)
                        
                        if sinkhorn_loss == 'cost_matrix':
                            loss_original = torch.sum(match_original * cost_original, dim=1, keepdim=True)
                            loss_mirror = torch.sum(match_mirror * cost_mirror, dim=1, keepdim=True)
                        else:
                            target_aligned_original = match_original.T @ pos_tgt_type
                            target_aligned_mirror = match_mirror.T @ pos_tgt_type
                            
                            if sinkhorn_loss == 'deviation':
                                loss_original = torch.norm(pos_pred_type - target_aligned_original, dim=1, keepdim=True)
                                loss_mirror = torch.norm(pos_pred_mirror_type - target_aligned_mirror, dim=1, keepdim=True)
                            else:  # rmsd
                                loss_original = torch.sum((pos_pred_type - target_aligned_original)**2, dim=1, keepdim=True)
                                loss_mirror = torch.sum((pos_pred_mirror_type - target_aligned_mirror)**2, dim=1, keepdim=True)
                        
                        total_loss_original[type_mask] = loss_original
                        total_loss_mirror[type_mask] = loss_mirror
                    
                    # Apply softmin loss for the entire molecule
                    node_losses[graph_mask] = self.softmin_loss(total_loss_original, total_loss_mirror)
                else:
                    # Direct MSE loss
                    node_losses[graph_mask] = torch.sum((pos_pred - pos_tgt) ** 2, dim=1, keepdim=True)

        # Compute weights for hydrogen atoms - vectorized
        weights = torch.ones_like(node_losses, device=dev)
        hydrogen_weight_method = int(hydrogen_weight_method)  # Convert to int as in original
        if hydrogen_weight_method != 0 and sinkhorn_knopp:
            hydrogen_mask = atom_type == 1
            if hydrogen_mask.any():
                hydrogen_indices = torch.nonzero(hydrogen_mask, as_tuple=True)[0]
                hydrogen_graph_ids = node2graph[hydrogen_indices]
                
                h_weights = torch.full((hydrogen_indices.size(0), 1), 0.5, device=dev)
                
                # Compute conditions once
                condition1 = time_step[hydrogen_graph_ids] < threshold_kabsch if kabsch else torch.zeros_like(time_step[hydrogen_graph_ids], dtype=torch.bool)
                condition2 = time_step[hydrogen_graph_ids] < threshold_sinkhorn
                mask_between = (~condition1 & ~condition2).unsqueeze(1)
                
                # Update weights based on conditions
                h_weights = torch.where(
                    condition1.unsqueeze(1),
                    torch.tensor(1.0, device=dev).unsqueeze(0),
                    h_weights
                )
                
                if hydrogen_weight_method == 1:
                    # Linear interpolation for hydrogen_weight == 1
                    between_weight = (0.5 + 1.0) / 2
                    h_weights = torch.where(
                        mask_between,
                        torch.tensor(between_weight, device=dev).unsqueeze(0),
                        h_weights
                    )
                elif hydrogen_weight_method == 2:
                    # Linear interpolation for hydrogen_weight == 2
                    linear_interp = (time_step[hydrogen_graph_ids] - threshold_kabsch).float() / (threshold_sinkhorn - threshold_kabsch)
                    h_weights = torch.where(
                        mask_between,
                        1.0 + (0.5 - 1.0) * linear_interp.unsqueeze(1),
                        h_weights
                    )
                elif hydrogen_weight_method == 3:
                    # Nonlinear interpolation for hydrogen_weight == 3
                    norm_time = (time_step[hydrogen_graph_ids] - threshold_kabsch).float() / (threshold_sinkhorn - threshold_kabsch)
                    nonlinear_interp = 1 - (1 - norm_time) ** 2
                    h_weights = torch.where(
                        mask_between,
                        1.0 + (0.5 - 1.0) * nonlinear_interp.unsqueeze(1),
                        h_weights
                    )
                elif hydrogen_weight_method == 0.5:
                    h_weights = torch.full_like(time_step[hydrogen_graph_ids].float(), 0.5, device=dev).unsqueeze(1)
                else:
                    raise ValueError('The value of hydrogen_weight can only be selected from 0-3')
                
                weights[hydrogen_indices] = h_weights

        # Apply weights and compute final loss
        loss = node_losses * weights

        # Calculate aromatic ring planarity loss
        if force_planarity.enable and aromatic_rings is not None:
            #!
            force_planarity.ratio=0.01
            # force_planarity.weight=1
            threshold_force_planarity = int(t1 * force_planarity.ratio)
            time_mask_planarity = time_step < threshold_force_planarity
            
            # Only proceed if there are any graphs in the time threshold
            if time_mask_planarity.any():
                # Group rings by their corresponding graph in the batch
                graph_rings = [[] for _ in range(num_graphs)]
                for graph_idx, rings in enumerate(aromatic_rings):
                    if time_mask_planarity[graph_idx] and rings:  # Only add rings for graphs within time threshold
                        graph_rings[graph_idx].extend(rings)
                
                # Calculate planarity loss for each graph
                graph_planarity_losses = [0.0] * num_graphs  # Initialize with zeros for all graphs
                for graph_idx, rings in enumerate(graph_rings):
                    if not rings:  # Skip if no aromatic rings in this graph
                        continue
                        
                    graph_loss = 0.0
                    for ring in rings:
                        ring_pos = pos[ring] # Get positions of atoms in the ring
                        # First, center the points
                        center = ring_pos.mean(dim=0)
                        centered_pos = ring_pos - center
                        # Perform SVD to find the normal vector
                        U, S, Vh = torch.linalg.svd(centered_pos)
                        normal = Vh[-1]  # The last right singular vector is the normal
                        # Calculate distances from each point to the plane
                        distances = torch.abs(torch.matmul(centered_pos, normal))
                        # Scale the distances by the ring size to make it more comparable
                        distances = distances * len(ring)
                        # Add to graph loss
                        graph_loss += torch.mean(distances)
                    # Add the total loss for this graph
                    graph_planarity_losses[graph_idx] = graph_loss
                
                if any(graph_planarity_losses):  # If there are any aromatic rings in any graph
                    # Create a tensor of planarity losses for each node
                    node_planarity_losses = torch.zeros(node2graph.size(0), 1, device=dev)
                    
                    # Distribute the planarity loss only to nodes in the rings
                    for graph_idx, rings in enumerate(graph_rings):
                        if not rings:  # Skip if no rings in this graph
                            continue
                        graph_loss = graph_planarity_losses[graph_idx]
                        # Add loss to all nodes in the rings of this graph
                        for ring in rings:
                            node_planarity_losses[ring] = graph_loss
                            
                    loss = loss + force_planarity.weight * node_planarity_losses


        # Gradient penalty if needed
        if lambda_gp > 0:
            params_with_grad = [p for p in self.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                outputs=loss.sum(), inputs=params_with_grad,
                create_graph=True, retain_graph=True
            )
            grad_norm = torch.cat([g.view(-1) for g in grads if g is not None]).norm(2)
            # Create a tensor of gradient penalties for each node
            node_gp_losses = torch.full_like(loss, lambda_gp * (grad_norm ** 2) / loss.size(0))
            loss = loss + node_gp_losses

        return loss
      
    def get_loss_scaler(
            self,
            atom_type,
            feat,
            pos,
            bond_index,
            bond_type,
            batch,
            num_graphs,
            prop,
            extend_order=True,
            extend_radius=True,
        ):
        
            node2graph = batch
            assert atom_type.size(0) == node2graph.size(0), \
            f"Mismatch in atom_type and node2graph sizes: {atom_type.size(0)} != {node2graph.size(0)}"

            dev = pos.device
            # set time step and noise level
            t0 = self.config.get("t0", 0)

            sz = num_graphs // 2 + 1
            time_step = torch.full((2*sz,), t0, device=dev, dtype=torch.int)[:num_graphs]
            a = self.alphas.index_select(0, time_step)  # (G, )

            # Perterb pos
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
            pos_noise = torch.randn(size=pos.size(), device=dev)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

            # prediction
            scalers = self(
                atom_type, feat, pos_perturbed, bond_index, bond_type,
                batch, time_step, return_edges=True, extend_order=extend_order,
                extend_radius=extend_radius,
                mode="scaler",
            )  # (N, 1)

            # Calculate loss
            assert scalers.size(0) == prop.size(0), \
            f"Mismatch in scalers and prop sizes: {scalers.size(0)} != {prop.size(0)}"

            loss = ((scalers - prop) ** 2) # Mean Squared Error loss

            return loss
