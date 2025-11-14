"""Module to check flatness of ligand substructures."""


from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
from numpy import ndarray as Array

from rdkit import Chem

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmarts
from rdkit.Chem.rdmolops import SanitizeMol

def check_flatness_main(mol_list):
    mask = []
    for mol in mol_list:
        if mol:
            resluts_and_details, FLAT = _check_flatness(mol)
            mask.append(FLAT)
        else:
            mask.append(False)
    print(mask)
    all_flat = all(mask)        
    return mask, all_flat

def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    # Add a conformer if it doesn't exist
    if mol.GetNumConformers() == 0:
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    set_rdmol_positions_(mol, pos)
    return mol

def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol

def check_planarity(data, threshold_flatness=0.1, flat_system=None):
    smi = data.smiles
    coord = data.pos_gen
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)
    mol_pred = set_rdmol_positions(mol, coord)
    _, FLAT = _check_flatness(mol_pred, threshold_flatness = threshold_flatness, flat_systems = flat_system)
    return FLAT

def _check_flatness(
    mol_pred, threshold_flatness = 0.1, flat_systems = None
):
    """Check whether substructures of molecule are flat.
    """
    if not flat_systems or not isinstance(flat_systems, dict):
        flat_systems = {
            "aromatic_6_membered_rings_sp2": "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1",
            "trigonal_planar_double_bonds": "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)",
        }
    mol = deepcopy(mol_pred)

    # if mol cannot be sanitized, then rdkit may not find substructures
    assert mol_pred.GetNumConformers() > 0, "Molecule does not have a conformer."
    flags = SanitizeMol(mol)
    if flags != 0:
        return None, None

    planar_groups = []
    types = []
    for flat_system, smarts in flat_systems.items():
        match = MolFromSmarts(smarts)
        atom_groups = list(mol.GetSubstructMatches(match))
        planar_groups += atom_groups
        types += [flat_system] * len(atom_groups)

    # calculate distances to plane and check threshold
    coords = [_get_coords(mol, group) for group in planar_groups]
    max_distances = [float(_get_distances_to_plane(X).max()) for X in coords]
    flatness_passes = [bool(d <= threshold_flatness) for d in max_distances]
    details = {
        "type": types,
        "planar_group": planar_groups,
        "max_distance": max_distances,
        "flatness_passes": flatness_passes,
    }

    results = {
        "num_systems_checked": len(planar_groups),
        "num_systems_passed": sum(flatness_passes),
        "max_distance": max(max_distances) if max_distances else np.nan,
        "flatness_passes": all(flatness_passes) if len(flatness_passes) > 0 else True,
    }
    # print(details)
    return {"results": results, "details": details}, results['flatness_passes']


def _get_distances_to_plane(X):
    """Get distances of points X to their common plane."""
    # center points X in R^(n x 3)
    X = X - X.mean(axis=0)
    # singular value decomposition
    _, _, V = np.linalg.svd(X)
    # last vector in V is normal vector to plane
    n = V[-1]
    # distances to plane are projections onto normal
    d = np.dot(X, n)
    return d


def _get_coords(mol, indices):
    return np.array([mol.GetConformer().GetAtomPosition(i) for i in indices])

