import os
import pickle
import copy
import json
from collections import defaultdict
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
from collections import Counter

import torch
from torch_geometric.data import Data, Dataset, Batch

from torch_geometric.utils import to_networkx
from torch_scatter import scatter

# from torch.utils.data import Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger


from ase.data import atomic_numbers

# import sidechainnet as scn
RDLogger.DisableLog("rdApp.*")

from .chem import BOND_TYPES, mol_to_smiles


def read_xyz_block(xyz_block):
    sxyz = xyz_block.split("\n")[2:]
    if sxyz[-1]:
        pass
    else:
        sxyz = sxyz[:-1]

    symbols = []
    pos = []
    for line in sxyz:
        li = line.split()
        symbols.append(li[0])
        pos.append([float(u) for u in li[1:]])

    symbols = np.array(symbols)
    pos = np.array(pos)
    return symbols, pos


def _atoms_order():
    desired_order = ["C", "N", "O", "F", "S"]
    symbol_priority = {symbol: i for i, symbol in enumerate(desired_order)}
    symbol_priority["H"] = 999
    return symbol_priority

def _get_priority(symbol_priority, atom):
    return symbol_priority.get(atom.GetSymbol(), 999)

def order_mol(mol):
    symbol_priority = _atoms_order()
    indices = sorted(range(mol.GetNumAtoms()), key=lambda idx: _get_priority(symbol_priority, mol.GetAtomWithIdx(idx)))
    mol = Chem.RenumberAtoms(mol, indices)
    return mol


def center_pos(coords: torch.Tensor) -> torch.Tensor:
    center = coords.mean(dim=0)
    centered_coords = coords - center
    return centered_coords

def generate_data(
    input_smi,
    xyz_block,
    data_cls=Data,
    feat_dict={},
    remove_H=None,
):
    assert isinstance(input_smi, str)
    mol = Chem.MolFromSmiles(input_smi)
    if not remove_H:
        mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)

    mol = order_mol(mol)
    
    N = mol.GetNumAtoms()

    atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    if xyz_block is not None:
        if isinstance(xyz_block, str):
            symbol_xyz, pos = read_xyz_block(xyz_block)
            numeric_list = [atomic_numbers[symbol] for symbol in symbol_xyz]
            assert atomic_number == numeric_list, 'Error atom order'
        else:
            pos = xyz_block
        pos = torch.Tensor(pos)
        assert len(pos) == N
    else:
        pos = torch.zeros(N,3)

    #CoM!    
    pos = center_pos(pos)

    mol_feat = []
    for atom in np.array(mol.GetAtoms()):
        atomic_feat = []
        for k, v in feat_dict.items():
            feat = getattr(atom, k)()
            if feat not in v:
                v.update({feat: len(v)})
            atomic_feat.append(v[feat])
        mol_feat.append(atomic_feat)

    # Extract aromatic ring atom indices
    aromatic_rings = []
    for ring in mol.GetRingInfo().AtomRings():
        # Check if the ring is aromatic
        is_aromatic = True
        for atom_idx in ring:
            if not mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                is_aromatic = False
                break
        if is_aromatic:
            aromatic_rings.append(ring)

    z = torch.tensor(atomic_number, dtype=torch.long)
    mol_feat = torch.tensor(mol_feat, dtype=torch.long)
    mol_adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    row, col = mol_adj.nonzero()

    _nonbond = 0
    mol_edge_type = [
        BOND_TYPES[mol.GetBondBetweenAtoms(int(i), int(j)).GetBondType()] 
        if mol.GetBondBetweenAtoms(int(i), int(j)) is not None 
        else _nonbond
        for i, j in zip(row, col)
    ]

    # mol_edge_type = []
    # for i, j in zip(row, col):
    #     b = mol.GetBondBetweenAtoms(int(i), int(j))
    #     if b is not None:
    #         mol_edge_type.append(BOND_TYPES[b.GetBondType()])
    #     else:
    #         mol_edge_type.append(_nonbond)

    # ! Modify beacuse of the torch version
    # edge_index = torch.tensor([row, col], dtype=torch.long)
    
    edge_index = torch.stack([torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long)])
    
    mol_edge_type = torch.tensor(mol_edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    mol_edge_type = mol_edge_type[perm]

    row, col = edge_index
    # hs = (z == 1).to(torch.float32)
    # num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

    data = data_cls(
        atom_type=z,
        feat=mol_feat,
        pos=pos,
        edge_index=edge_index,
        edge_type=mol_edge_type,
        rdmol=(copy.deepcopy(mol)),
        smiles=input_smi,
        aromatic_rings=aromatic_rings,
    )
    return data, feat_dict



class TSDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform

    def get(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx])
        else:
            return self.data[idx]

    def len(self):
        return len(self.data)


class ConformationDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)


class SidechainConformationDataset(ConformationDataset):
    def __init__(
        self, path, transform=None, cutoff=10.0, max_residue=5000, fix_subgraph=False
    ):
        super().__init__(path, transform)
        self.cutoff = cutoff
        self.max_residue = max_residue
        self.fix_subgraph = fix_subgraph

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        """ Subgraph sampling
            1. sampling an atom from the backbone (residue)
            2. Find all neighboring atoms within a cutoff
            3. extend atoms to ensure the completeness of each residue
            4. remap the index for subgraph
        """
        is_sidechain = data.is_sidechain
        pos = data.pos
        edge_index = data.edge_index
        atom2res = data.atom2res
        dummy_index = torch.arange(pos.size(0))
        backbone_index = dummy_index[~is_sidechain]

        # stop=False
        # while not stop:
        # step 1
        if self.fix_subgraph:
            center_atom_index = backbone_index[backbone_index.size(0) // 2].view(
                1,
            )
        else:
            center_atom_index = backbone_index[
                torch.randint(low=0, high=backbone_index.size(0), size=(1,))
            ]  # (1, )
        pos_center_atom = pos[center_atom_index]  # (1, 3)
        # step 2
        distance = (pos_center_atom - pos).norm(dim=-1)
        mask = distance <= self.cutoff
        # step 3
        is_keep_residue = scatter(
            mask, atom2res, dim=-1, dim_size=self.max_residue, reduce="sum"
        )  # (max_residue, )
        is_keep_atom = is_keep_residue[atom2res]
        is_keep_edge = (is_keep_atom[edge_index[0]]) & (is_keep_atom[edge_index[1]])
        # step 4
        mapping = -torch.ones(pos.size(0), dtype=torch.long)
        keep_index = dummy_index[is_keep_atom]
        mapping[keep_index] = torch.arange(keep_index.size(0))
        if (data.is_sidechain[is_keep_atom]).sum().item() == 0:
            # stop = True
            return None

        # return subgraph data
        subgraph_data = Data(
            atom_type=data.atom_type[is_keep_atom],
            pos=data.pos[is_keep_atom],
            edge_index=mapping[data.edge_index[:, is_keep_edge]],
            edge_type=data.edge_type[is_keep_edge],
            is_sidechain=data.is_sidechain[is_keep_atom],
            atom2res=data.atom2res[is_keep_atom],
        )

        if self.transform is not None:
            subgraph_data = self.transform(subgraph_data)
        return subgraph_data

    @staticmethod
    def collate_fn(data):

        batch = [_ for _ in data if _ is not None]
        return Batch.from_data_list(batch)


def accumulate_grad_from_subgraph(
    model,
    atom_type,
    pos,
    bond_index,
    bond_type,
    batch,
    atom2res,
    batch_size=8,
    device="cuda:0",
    is_sidechain=None,
    is_alpha=None,
    pos_gt=None,
    cutoff=10.0,
    max_residue=5000,
    transform=None,
):
    """
    1. decompose the protein to subgraphs
    2. evaluate subgraphs using trained models
    3. accumulate atom-wise grads
    4. return grads
    """

    accumulated_grad = torch.zeros_like(pos)
    accumulated_time = torch.zeros(pos.size(0), device=pos.deivce)

    all_subgraphs = []
    dummy_index = torch.arange(pos.size(0))

    # prepare subgraphs
    is_covered = torch.zeros(pos.size(0), device=pos.deivce).bool()
    is_alpha_and_uncovered = is_alpha & (~is_covered)
    while is_alpha_and_uncovered.sum().item() != 0:

        alpha_index = dummy_index[is_alpha_and_uncovered]
        center_atom_index = alpha_index[
            torch.randint(low=0, high=alpha_index.size(0), size=(1,))
        ]  # (1, )
        pos_center_atom = pos[center_atom_index]  # (1, 3)

        distance = (pos_center_atom - pos).norm(dim=-1)
        mask = distance <= cutoff

        is_keep_residue = scatter(
            mask, atom2res, dim=-1, dim_size=max_residue, reduce="sum"
        )  # (max_residue, )
        is_keep_atom = is_keep_residue[atom2res]
        is_keep_edge = (is_keep_atom[bond_index[0]]) & (is_keep_atom[bond_index[1]])

        mapping = -torch.ones(pos.size(0), dtype=torch.long)
        keep_index = dummy_index[is_keep_atom]
        mapping[keep_index] = torch.arange(keep_index.size(0))

        is_covered |= is_keep_atom
        is_alpha_and_uncovered = is_alpha & (~is_covered)

        if (is_sidechain[is_keep_atom]).sum().item() == 0:
            continue

        subgraph = Data(
            atom_type=atom_type[is_keep_atom],
            pos=pos[is_keep_atom],
            edge_index=mapping[bond_index[:, is_keep_edge]],
            edge_type=bond_type[is_keep_edge],
            is_sidechain=is_sidechain[is_keep_atom],
            atom2res=atom2res[is_keep_atom],
            mapping=keep_index,
        )
        if transform is not None:
            subgraph = transform(subgraph)
        all_subgraphs.append(subgraph)

    # run model
    tot_iters = (len(all_subgraphs) + batch_size - 1) // batch_size
    for it in range(tot_iters):
        batch = Batch.from_data_list(
            all_subgraphs[it * batch_size, (it + 1) * batch_size]
        ).to(device)


class PackedConformationDataset(ConformationDataset):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)
        # k:v = idx: data_obj
        self._pack_data_by_mol()

    def _pack_data_by_mol(self):
        """
        pack confs with same mol into a single data object
        """
        self._packed_data = defaultdict(list)
        if hasattr(self.data, "idx"):
            for i in range(len(self.data)):
                self._packed_data[self.data[i].idx.item()].append(self.data[i])
        else:
            for i in range(len(self.data)):
                self._packed_data[self.data[i].smiles].append(self.data[i])
        print(
            "[Packed] %d Molecules, %d Conformations."
            % (len(self._packed_data), len(self.data))
        )

        new_data = []
        # logic
        # save graph structure for each mol once, but store all confs
        cnt = 0
        for k, v in self._packed_data.items():
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                all_pos.append(v[i].pos)
            data.pos_ref = torch.cat(all_pos, 0)  # (num_conf*num_node, 3)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            # del data.pos

            if hasattr(data, "totalenergy"):
                del data.totalenergy
            if hasattr(data, "boltzmannweight"):
                del data.boltzmannweight
            new_data.append(data)
        self.new_data = new_data

    def __getitem__(self, idx):

        data = self.new_data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.new_data)
