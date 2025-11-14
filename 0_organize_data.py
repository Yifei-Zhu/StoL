"""
zhuyf
2024.11.19
"""
import os
import numpy as np
import pandas as pd
import yaml
import pickle
from tqdm import tqdm
from easydict import EasyDict
from collections import Counter

import rdkit as rd
from rdkit import Chem
from ase.data import chemical_symbols

from qcdge_related.extract_data import extractData as etd
from qcdge_related.props import PropsDict

def extract_from_hdf5(config, hdf5_file, csv_file, save_smiles, save_xyz, order=1, prop_list=[], other_props=[], save_pkl=None):
            
    remove_H = config.Data.remove_H
    heavy_atom_threshold = config.Data.extract_data.heavy_atom_threshold
    element_input = config.Data.extract_data.element_input
    remove_reference = config.Data.other_props.remove_reference.enable
    
    qcdge = PropsDict()

    extractor = etd(order=order, prop_list = prop_list+other_props, hdf5_file = hdf5_file, csv_file = csv_file, heavy_atom_threshold=heavy_atom_threshold, element_input=element_input)
    data_dict = extractor.read_from_hdf5()

    atom_types = np.array(data_dict[qcdge.get_key('1')], dtype=object)
    coordinates = np.array(data_dict[qcdge.get_key('2')], dtype=object)
    smiles_can = np.array(data_dict[qcdge.get_key('105')])
    mol_list = np.array(data_dict['mol'])

    props_variables = {}

    if other_props:
        for prop in other_props:
            variable_name = qcdge.get_key(str(prop))
            props_variables[variable_name] = np.array(data_dict[variable_name])
    if order != 3:
        smiles_mask = smiles_can == '1'
        coordinates = coordinates[~smiles_mask]
        atom_types = atom_types[~smiles_mask]
        smiles_can = smiles_can[~smiles_mask]
        mol_list = mol_list[~smiles_mask]
        smiles_can = add_hydrogens_to_smiles(smiles_can)
        
        for key in props_variables.keys():
            props_variables[key] = props_variables[key][~smiles_mask]

    # Save smiles and coords separately
    write_smiles_to_csv(smiles_can, save_smiles)
    write_coords_to_xyz(atom_types, coordinates, mol_list, smiles_can, save_xyz, remove_H=remove_H)
    
    # Modify props_variables format
    for key in props_variables.keys():
        props_variables[key] = [item[0][0] for item in props_variables[key]]
    
    # ! current version only consider total GS energy
    if 3 in other_props and remove_reference:
        props_list=[]
        atom_e_dict = qcdge.reference_atom_energy(level=int(config.Data.other_props.remove_reference.level), unit=int(config.Data.other_props.remove_reference.unit))
        for p,smi in zip(props_variables[qcdge.get_key(str(3))], smiles_can):
            energy = cal_reference_energy(smi, atom_e_dict)
            e = float(p)-energy
            props_list.append(e)
        props_variables[qcdge.get_key(str(3))] = props_list
        
    # Save prop dict to pkl file
    if other_props:
        f_save = open(save_pkl, 'wb')
        pickle.dump(props_variables, f_save)
        f_save.close()

def cal_reference_energy(smi, atom_e_dict):       
    atom_counts = count_atoms_in_smiles(smi)
    total_energy = 0.0
    total_energy = sum(atom_counts[atom] * atom_e_dict[atom] for atom in atom_counts)
    return total_energy

def count_atoms_in_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Invalid SMILES!")
    
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_counts = dict(Counter(atoms))
    return atom_counts

def write_smiles_to_csv(smiles_can, save_smiles):
    df = pd.DataFrame(smiles_can, columns=["SMILES_RDKIT_CAN"])
    df.to_csv(save_smiles, index=False)

def write_coords_to_xyz(atom_types, coordinates, mol_list, smiles_can, save_xyz, remove_H):

    with open(save_xyz, 'w') as file:
        for i, (atoms, coords) in enumerate(zip(atom_types, coordinates)):

            if remove_H:
                count_of_ones = atoms[0].count(1)
            else:
                count_of_ones = 0

            file.write(f"{len(atoms[0])-count_of_ones}\n")
            file.write(f"{mol_list[i]}\t{smiles_can[i]}\n")
            labels = [chemical_symbols[int(number)] for number in atoms[0]]
            for label, coord in zip(labels, coords):
                if remove_H and label == "H":
                    continue
                x, y, z = coord
                file.write(f"{label} {x:.6f} {y:.6f} {z:.6f}\n")

def add_hydrogens_to_smiles(smiles_list):
    smiles_with_h = []
    for smiles in tqdm(smiles_list, desc="Adding hydrogens to SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        assert mol != None
        mol_with_h = Chem.AddHs(mol)
        smiles_with_h.append(Chem.MolToSmiles(mol_with_h))
    return smiles_with_h

if __name__ == '__main__':
    config_path = config_path = os.path.join(os.getcwd(), "configs/data_config.yaml")

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    
    hdf5_file, csv_file, save_smiles, save_xyz = (
        config.Data.extract_data.hdf5_file, 
        config.Data.extract_data.csv_file, 
        config.Data.extract_data.save_smiles, 
        config.Data.extract_data.save_xyz
    )

    prop_list = config.Data.extract_data.prop_list.split(',')
    prop_list = [int(x) for x in prop_list]
    
    other_props = config.Data.other_props.enable
    if other_props:
        other_props = config.Data.other_props.props.split(',')
        other_props = [int(x) for x in other_props]
        save_pkl = config.Data.other_props.save_pkl
    else:
        other_props=[]
        save_pkl=None
   
    extract_from_hdf5(config, hdf5_file, csv_file, save_smiles, save_xyz, order=3, prop_list=prop_list, other_props=other_props, save_pkl=save_pkl)

