import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_ordered_structure(smiles, atom_types):
    """
    Generate a molecular structure with atom order consistent with the given atom types.

    Parameters:
    - smiles: str, SMILES string representing the molecule.
    - atom_types: list of str, list of atom types in desired order.

    Returns:
    - structure: numpy array of shape (n_atoms, 3), where n_atoms is the number of atoms in the molecule.
    """
    # Generate RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # Get atom coordinates
    conformer = mol.GetConformer()
    coords = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    atom_coords = [(mol.GetAtomWithIdx(i).GetSymbol(), [coords[i].x, coords[i].y, coords[i].z]) for i in range(mol.GetNumAtoms())]
    
    # Reorder atoms to match `atom_types`
    ordered_coords = []
    for atom_type in atom_types:
        for i, (symbol, coord) in enumerate(atom_coords):
            if symbol == atom_type:
                ordered_coords.append(coord)
                atom_coords.pop(i)
                break
    
    # Convert to numpy array and return
    return np.array(ordered_coords)



def generate_confs(smiles, atom_types, num_confs=20):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    conformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
    rmslist=[]
    AllChem.AlignMolConformers(mol, RMSlist=rmslist)

    all_confs_coords = []

    for conf_id in conformers:
        conformer = mol.GetConformer(conf_id)
        coords = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        atom_coords = [
            (mol.GetAtomWithIdx(i).GetSymbol(), [coords[i].x, coords[i].y, coords[i].z])
            for i in range(mol.GetNumAtoms())
        ]

        tmp_atom_coords = atom_coords[:]
        ordered_coords = []
        for atom_type in atom_types:
            for i, (symbol, coord) in enumerate(tmp_atom_coords):
                if symbol == atom_type:
                    ordered_coords.append(coord)
                    tmp_atom_coords.pop(i)

        all_confs_coords.append(np.array(ordered_coords))

    return all_confs_coords



if __name__=='_main__':
# Example usage
    ref = {
        "smiles": "CCO",
        "atom_type": ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    }

    ordered_structure = generate_ordered_structure(ref['smiles'], ref['atom_type'])
    for atom, coord in ordered_structure:
        print(f"{atom}: {coord}")
