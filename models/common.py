# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils.chem import BOND_TYPES
from utils import activation_loader


class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            # self.activation = getattr(F, activation)
            self.activation = activation_loader(activation)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation is None:
                    print("No activation in MultiLayerPerceptron")
                else:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x

def index_set_subtraction(index1:torch.LongTensor, index2:torch.LongTensor, max_num_nodes=None):
    dev = index1.device
    adj1 = to_dense_adj(
            index1, 
            edge_attr=torch.arange(1, index1.shape[1]+1).to(dev), 
            max_num_nodes=max_num_nodes
            )
    adj2 = to_dense_adj(
            index2, 
            edge_attr=torch.ones(index2.shape[1]).to(dev), 
            max_num_nodes=max_num_nodes
            )
    
    adj = adj1 - adj2 * (index1.shape[1] + 1)
    mask = (adj > 0)
    index = adj1[mask] - 1
    #return index

    mask = torch.zeros(index1.size(1), device=dev)
    mask[index] = 1
    mask = mask.bool()
    return mask

def extend_ts_graph_order_radius(
    num_nodes,
    pos,
    edge_index,
    edge_type,
    batch,
    order=3,
    cutoff=10.0,
):

    edge_index_local, edge_type = _extend_ts_graph_order(
        num_nodes, edge_index, edge_type, batch, order=order
    )

    edge_index_global, _ = _extend_to_radius_graph(
        pos, edge_index_local, edge_type, cutoff, batch
    )

    return edge_index_global, edge_index_local, edge_type


def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (E, 2H)
    return h_pair

def _extend_ts_graph_order(num_nodes, edge_index, edge_type, batch, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
        ]

        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index,max_num_nodes=N).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type, max_num_nodes=N).squeeze(0)  # (N, N)
    type_highorder = torch.where(
        adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order)
    )
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)

    # data.bond_edge_index = data.edge_index  # Save original edges
    new_edge_index, new_edge_type = coalesce(
        new_edge_index, new_edge_type.long(), N, N
    )  # modify data

    # [Note] This is not necessary
    # data.is_bond = (data.edge_type < num_types)

    # [Note] In earlier versions, `edge_order` attribute will be added.
    #         However, it doesn't seem to be necessary anymore so I removed it.
    # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
    # assert (data.edge_index == edge_index_1).all()

    return new_edge_index, new_edge_type


def _extend_to_radius_graph(
    pos,
    edge_index,
    edge_type,
    cutoff,
    batch,
    unspecified_type_number=0
):

    assert edge_type.dim() == 1
    N = pos.size(0)

    # bgraph_adj = torch.sparse.LongTensor(edge_index, edge_type, torch.Size([N, N]))
    bgraph_adj = torch.sparse_coo_tensor(edge_index, edge_type, torch.Size([N, N]),dtype=torch.long)

    rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)  # (2, E_r)

    # rgraph_adj = torch.sparse.LongTensor(
    #     rgraph_edge_index,
    #     torch.ones(rgraph_edge_index.size(1)).long().to(pos.device)
    #     * unspecified_type_number,
    #     torch.Size([N, N]),
    # )
    rgraph_adj = torch.sparse_coo_tensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device)
        * unspecified_type_number,
        torch.Size([N, N]),
        dtype=torch.long
    )
    

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


