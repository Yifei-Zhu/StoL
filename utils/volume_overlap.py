
"""
This code is based on the original code available at: https://github.com/maabuu/posebusters.git,
BSD 3-Clause License
Copyright (c) 2023, Martin Buttenschoen
All rights reserved.
Module to check volume overlap between docked ligand and protein.
"""

import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdShapeHelpers import ShapeTverskyIndex
from rdkit.Chem.rdchem import Mol, RWMol
from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform
from rdkit.Chem.rdmolops import AddHs, RemoveHs, RemoveStereochemistry, RenumberAtoms, SanitizeMol


def check_volume_overlap(  # noqa: PLR0913
    mol_1,
    mol_2,
    clash_cutoff= 0.05,
    vdw_scale= 0.8,
    ignore_Hs=False,
    search_distance=6.0,
):
    """Check volume overlap between mol_1 and mol_2
    Args:
        mol_1: RDKit molecule
        mol_2: RDKit molecule
        clash_cutoff: Cutoff distance for volume overlap
        vdw_scale: VDW scale factor for volume overlap
        ignore_Hs: Whether to ignore hydrogens
        search_distance: Distance to search for volume overlap
    """
    # filter by atom types
    if ignore_Hs:
        ignore_types={"hydrogens"}
        keep_mask = np.array(get_atom_type_mask(mol_1, ignore_types))
        mol_1 = _filter_by_mask(mol_1, keep_mask)
        keep_mask = np.array(get_atom_type_mask(mol_2, ignore_types))
        mol_2 = _filter_by_mask(mol_2, keep_mask)
        
    # filter by distance --> this is slowing this function down
    distances = _pairwise_distance(mol_1.GetConformer().GetPositions(), mol_2.GetConformer().GetPositions())
    keep_mask = distances.min(axis=0) <= search_distance * vdw_scale
    mol_2 = _filter_by_mask(mol_2, keep_mask)
    if mol_2.GetNumAtoms() == 0:
        return {"results": {"volume_overlap": np.nan, "no_volume_clash": True}}, False

    ignore_hydrogens = "hydrogens" in ignore_types
    overlap = ShapeTverskyIndex(mol_1, mol_2, alpha=1, beta=0, vdwScale=vdw_scale, ignoreHs=ignore_hydrogens)

    results = {
        "volume_overlap": overlap,
        "no_volume_clash": overlap <= clash_cutoff,
    }

    return {"results": results}, True


def _pairwise_distance(x, y):
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def _filter_by_mask(mol, mask):
    if mask.sum() < len(mask):
        mol = delete_atoms(mol, np.where(~mask)[0].tolist())
    return mol

def delete_atoms(mol, indices):
    """Delete atoms from molecule.

    Args:
        mol: Molecule to delete atoms from.

    Returns:
        Molecule without atoms.
    """
    # delete in reverse order to avoid reindexing issues
    indices = sorted(indices, reverse=True)
    if len(indices) == 0:
        return mol
    mol = RWMol(mol)
    for index in indices:
        mol.RemoveAtom(index)
    return Mol(mol)

def get_atom_type_mask(mol, ignore_types) -> list[bool]:
    """Get mask for atoms to keep."""
    ignore_types = set(ignore_types)
    ignore_h = "hydrogens" in ignore_types
    return [
        _keep_atom(a, ignore_h) for a in mol.GetAtoms()
    ]
    
def _keep_atom(  # noqa: PLR0913, PLR0911
    atom, ignore_h) -> bool:
    """Whether to keep atom for given ignore flags."""
    symbol = atom.GetSymbol()
    if ignore_h and symbol == "H":
        return False
    return True