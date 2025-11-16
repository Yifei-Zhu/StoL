import os
import subprocess as sub
import tempfile
from rdkit import Chem
import re
import csv
import copy
def load_molecule(mol_path):
    """
    Load a MOL file using RDKit and explicitly add hydrogen atoms
    """
    mol = Chem.MolFromMolFile(mol_path, removeHs=True)
    if mol is None:
        raise ValueError(f"Failed to load MOL file: {mol_path}. Please check the file format.")
    return mol

def convert_xyz_to_mol(xyz_content, mol_path):
    """
    Use the Open Babel command-line tool to convert an XYZ content to a MOL file
    """
    try:
        # Write the single XYZ content to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xyz') as tmp_xyz:
            tmp_xyz.write(xyz_content)
            tmp_xyz_path = tmp_xyz.name
        # Call Open Babel to convert the temporary XYZ file to MOL, suppressing output
        command = f'obabel {tmp_xyz_path} -O {mol_path} >/dev/null 2>&1'
        sub.check_call(command, shell=True)
    except sub.CalledProcessError as e:
        raise RuntimeError(f"Open Babel conversion failed: {e}")
    finally:
        if os.path.exists(tmp_xyz_path):
            os.remove(tmp_xyz_path)

def generate_smiles_with_mapping(mol):
    """
    Generate SMILES string with atom mapping
    """

    # Add atom map numbers based on atom indices (assuming order is preserved)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # Atom indices start at 0

    smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, allHsExplicit=True)
    return smiles

def parse_smiles_mapping(smiles):
    """
    Parse the SMILES string to extract atom mappings
    Returns a dictionary mapping SMILES atom indices to XYZ atom line numbers
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Failed to parse SMILES string.")
    mapping = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        atom_idx = atom.GetIdx()
        if map_num != 0:
            mapping[atom_idx] = map_num - 1  # XYZ line indices start at 0
    return mapping


def get_related_H_dict(mol_path):
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    h_dict = {}

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()

        if atomic_num != 1:
            connected_hs = []
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    connected_hs.append(neighbor.GetIdx())
            h_dict[atom_idx] = connected_hs

    return h_dict


def get_mapping_between_xyz_and_smiles(xyz_file):
    smiles_list, mapping_list = [], []
    related_h_list=[]

    # Step 1: Read and split XYZ file into individual molecules
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    molecules = []
    i = 0
    while i < len(lines):
        num_atoms_line = lines[i].strip()
        if not num_atoms_line:
            i += 1
            continue  # Skip empty lines
        num_atoms = int(num_atoms_line)
        comment = lines[i+1].strip()
        atom_lines = lines[i+2:i+2+num_atoms]
        if len(atom_lines) < num_atoms:
            raise ValueError(f"Expected {num_atoms} atom lines, but found {len(atom_lines)}.")
        molecules.append(atom_lines)
        i += 2 + num_atoms

    # Step 2: Process each molecule
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, mol_xyz in enumerate(molecules):
            # Prepare XYZ content
            xyz_content = f"{len(mol_xyz)}\nMolecule {idx+1}\n" + ''.join(mol_xyz)
            mol_file = os.path.join(tmpdir, f"temp_{idx}.mol")
            try:
                # Convert XYZ to MOL
                convert_xyz_to_mol(xyz_content, mol_file)
                # Load molecule with RDKit
                mol  = load_molecule(mol_file)
                # Generate SMILES with mapping
                smiles_with_mapping = generate_smiles_with_mapping(mol)
                smiles_list.append(smiles_with_mapping)
                # Parse mappings
                mapping = parse_smiles_mapping(smiles_with_mapping)
                mapping_list.append(mapping)

                # get related index of H in xyz
                # {0: [], 1: [12], 2: [13], 3: [14], 4: [], 5: [15], 6: [], 7: [10, 11], 8: [], 9: [16]} means that the atom in line 10 is connected to H atom in line 16
                h_dict = get_related_H_dict(mol_file)
                related_h_list.append(h_dict)
            except Exception as e:
                # print(f"Error processing molecule {idx+1}: {e}")
                smiles_list.append(None)
                mapping_list.append(None)
                related_h_list.append(None)

    return smiles_list, mapping_list, related_h_list

def map_atoms_to_xyz(smiles_list, mapping_list, related_h_list):
    """
    Generate a complete atom correspondence list between SMILES atoms and XYZ atoms.

    Parameters:
        smiles_list (list of str): List of SMILES strings with atom mappings.
        mapping_list (list of dict): List of mappings from SMILES atom indices to XYZ atom line indices.

    Returns:
        list of list of tuples: For each molecule, a list of tuples containing (SMILES_atom_idx, XYZ_atom_line_idx).
    """
    correspondence_all = []
    for mol_idx, (smiles, mapping) in enumerate(zip(smiles_list, mapping_list)):
        if smiles is None or mapping is None:
            correspondence_all.append(None)
            continue
        correspondence = {}
        for smiles_atom_idx, xyz_line_idx in mapping.items():
            correspondence[smiles_atom_idx] =  xyz_line_idx
        # Sort by SMILES atom index for consistency
        # correspondence_sorted = sorted(correspondence, key=lambda x: x[0])
        correspondence_all.append(correspondence)
    return correspondence_all

def save_correspondence_to_csv(correspondence_all, output_file='atom_correspondence.csv'):
    """
    Save the atom correspondence to a CSV file.

    Parameters:
        correspondence_all (list of list of tuples): Complete atom correspondences for all molecules.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Molecule_Index', 'SMILES_Atom_Index', 'XYZ_Atom_Line'])
        for mol_idx, correspondence in enumerate(correspondence_all):
            if correspondence is None:
                writer.writerow([mol_idx+1, 'Error', 'Error'])
                continue
            for smiles_atom_idx, xyz_line_idx in correspondence:
                writer.writerow([mol_idx+1, smiles_atom_idx, xyz_line_idx])
    print(f"Atom correspondence has been saved to {output_file}")


def get_mapping_between_xyz_and_smiles_main(xyz_path):
    smiles, mappings, related_h_list = get_mapping_between_xyz_and_smiles(xyz_path)
    correspondence_all = map_atoms_to_xyz(smiles, mappings, related_h_list)
    return smiles, correspondence_all, related_h_list


if __name__ == "__main__":

    xyz_path = "1.xyz"

    smiles, mappings = get_mapping_between_xyz_and_smiles(xyz_path)

    for mol_idx, (smi, mapping) in enumerate(zip(smiles, mappings)):
        print(f"\nMolecule {mol_idx+1}:")
        print(f"SMILES: {smi}")
        if mapping is not None:
            print("Atom Mapping (SMILES_atom_idx -> XYZ_atom_line_idx):")
            for smiles_atom_idx, xyz_line_idx in sorted(mapping.items()):
                print(f"  {smiles_atom_idx} -> {xyz_line_idx}")
        else:
            print("  Mapping: None due to previous errors.")

    correspondence_all = map_atoms_to_xyz(smiles, mappings)

    for mol_idx, correspondence in enumerate(correspondence_all):
        print(f"\nMolecule {mol_idx+1} Atom Correspondence:")
        if correspondence is None:
            print("  Error in mapping.")
            continue
        for smiles_atom_idx, xyz_line_idx in correspondence:
            print(f"  SMILES Atom {smiles_atom_idx} <--> XYZ Line {xyz_line_idx}")


    # save_correspondence_to_csv(correspondence_all)
