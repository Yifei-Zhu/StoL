import argparse
from pathlib import Path
import pickle
from easydict import EasyDict
from matplotlib import pyplot as plt
import yaml

from rdkit import Chem

from utils.misc import get_logger
from utils.split_mol_to_fragments import get_redundant_H, draw_molecule_with_atom_indices, get_severable_bond, check_coverage, plot_fragments,save_fragments_to_csv

def read_smi_from_file(smi_file):
    smis=[]
    with open(smi_file) as f1:
        for line in f1:
            smi = line.strip()
            smis.append(smi)
    return smis

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def interactive_get_atom_of_fragments():
    number_of_fragments=input("Please input the number of fragments: ")
    atom_of_fragments=[]
    for i in range(int(number_of_fragments)):
        atom_nu_in_each_frag = input(f"Please input the atom number of the {i+1}th fragment (comma separated): ").strip().split(',')
        atom_of_fragments.append([int(x) for x in atom_nu_in_each_frag])
    return atom_of_fragments

def read_file_of_atom_of_fragments(file_path):
    """
    Read fragment information from a file and return a dictionary.
    
    Args:
        file_path (str): Path to the file containing fragment information
        
    Returns:
        dict: Dictionary with molecule index as key and list of fragment atoms as value
        
    File format:
        molecule_index
        atom_numbers_in_fragment_1
        atom_numbers_in_fragment_2
        ...
        [blank line]
        
        e.g.:
        10
        1,2,3,4,5
        3,4,5,6,7,8
        6,7,8,9,10,11,12,13
        
        15
        1,2,3,4,5,6
        4,5,6,7,8
    """
    fragments_dict = {}
    current_mol_idx = None
    current_fragments = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                if current_mol_idx is not None:
                    fragments_dict[current_mol_idx] = current_fragments
                    current_mol_idx = None
                    current_fragments = []
                continue
                
            if current_mol_idx is None:
                # First line of a molecule block is the molecule index
                current_mol_idx = int(line)
                current_fragments = []
            else:
                # Convert comma-separated string to list of integers
                atoms = [int(x) for x in line.split(',')]
                current_fragments.append(atoms)
    
    # Don't forget to add the last molecule if the file doesn't end with a blank line
    if current_mol_idx is not None:
        fragments_dict[current_mol_idx] = current_fragments
    
    return fragments_dict


def split_with_input(smiles, atom_of_fragments, save_path=None, print_fn=print):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Convert to canonical SMILES form without isotope/stereochemistry information
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    
    # 为原子添加映射编号以便于跟踪
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    # 生成带有原子映射的SMILES
    mol_smiles_mapped = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
    draw_molecule_with_atom_indices(mol, fig_path=f'{save_path}/figures_of_mol.png')

    severable_bonds = get_severable_bond(mol)
    
    # 生成每个片段的SMILES
    redundant_H_idx = []
    fragments = []
    for frag_atoms in atom_of_fragments:
        # 创建一个包含指定原子的子结构
        submol = Chem.RWMol(mol)
        # 删除不在片段中的原子
        atoms_to_remove = []
        for atom in submol.GetAtoms():
            if atom.GetIdx() not in frag_atoms:
                atoms_to_remove.append(atom.GetIdx())
        # 从后向前删除原子，避免索引变化
        for idx in sorted(atoms_to_remove, reverse=True):
            submol.RemoveAtom(idx)
        
        # 清理分子
        Chem.SanitizeMol(submol)
        
        # 获取片段的SMILES
        frag_smiles = Chem.MolToSmiles(submol, isomericSmiles=True, kekuleSmiles=True)
        fragments.append((frag_smiles, frag_atoms))

        # 记录需要添加的H原子
        redundant_H = []
        for atom in submol.GetAtoms():
            if atom.GetNumExplicitHs() > 0:
                redundant_H.append(atom.GetIdx())
        redundant_H_idx.append(redundant_H)
    
    frags_smi = [frag[0] for frag in fragments]
    frag_i = [frag[1] for frag in fragments]
    
    check_frag = check_coverage(mol, fragments)
    if not check_frag:
        print_fn(f'Failure! Molecule {smiles} fails to be split into fragments.')
        return False, False, False, False

    print_fn(f'Success! Molecule {smiles} was split into {len(fragments)} fragments.')
    plot_fragments(frags_smi+[smiles], fig_path=f'{save_path}/figures_of_frag.png')
    ref_pos_list_in_first, redundant_pos_list_in_second = get_redundant_H(severable_bonds, frag_i)
    csv_file = save_path / 'fragments.csv'
    save_fragments_to_csv(frags_smi, frag_i, ref_pos_list_in_first, redundant_pos_list_in_second, csv_file)
    
    return mol_smiles_mapped, frags_smi, frag_i, redundant_H_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split molecules into fragments manually or using prepared files.')
    
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--save_dir', type=str, help="The directory to save the result.", default='./')
    input_group.add_argument("--smis", type=str, help="SMILES strings of molecules to be split", nargs="+", default=None)
    input_group.add_argument("--smis_file", type=str, help="File containing SMILES strings (one per line)", default=None)
    input_group.add_argument("--failed_file", type=str, help="Pickle file containing failed molecules to process")
    input_group.add_argument('--atom_of_frags_file', type=str, help="File containing fragment atom information")

    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--mode', type=str, choices=['1', '2', '99'], 
                          help="1: Interactive mode\n2: Non-interactive mode\n99: Visualization mode",
                          required=True)
    

    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument(
        "--add_already_split",
        type=str2bool,
        help="If True, it will add the already split fragments to the list of fragments.",
        default=False
    )
    
    args = parser.parse_args()

    if not any([args.smis, args.smis_file, args.failed_file]):
        parser.error("At least one of --smis, --smis_file, or --failed_file must be provided")
    
    if args.mode == '2' and not args.atom_of_frags_file:
        parser.error("--atom_of_frags_file is required for non-interactive mode (mode 2)")
    
    save_dir = args.save_dir

    # Read smiles from file
    try:
        if args.failed_file:
            failed_dict = pickle.load(open(args.failed_file, 'rb'))
            nu_list = list(failed_dict.keys())
            smiles_list = list(failed_dict.values())
        else:
            smiles_list = args.smis if args.smis else read_smi_from_file(args.smis_file)
            nu_list = [i for i in range(len(smiles_list))]
    except Exception as e:
        print(f"Error: Failed to load the smiles list: {str(e)}")
        exit(1)

    if args.mode == "99":
        # Visualization mode
        for smi in smiles_list:
            plot_fragments([smi])
    else:
        success_dict = {}
        failed_dict = {}
        frag_smi_dict = {}
        sorted_fragments_dict = {}
        redundant_H_dict = {}

        if args.mode == "1":
            # Interactive mode
            for nu, smi in zip(nu_list, smiles_list):        
                save_path = Path(save_dir) / f'{nu}_mol'
                save_path.mkdir(parents=True, exist_ok=True)
                atom_of_fragments = interactive_get_atom_of_fragments()
                smiles_mapped, sorted_frag_smi, sorted_fragments, redundant_H_idx = split_with_input(smi, atom_of_fragments, save_path)

                success_dict[nu] = len(sorted_frag_smi)
                frag_smi_dict[nu] = sorted_frag_smi
                sorted_fragments_dict[nu] = sorted_fragments
                redundant_H_dict[nu] = redundant_H_idx
                
        elif args.mode == "2":
            # Non-interactive mode
            fragments_dict = read_file_of_atom_of_fragments(args.atom_of_frags_file)
            for nu, smi in zip(nu_list, smiles_list):        
                if nu not in fragments_dict:
                    print(f"Warning: No fragment information found for molecule {nu}, skipping...")
                    continue
                    
                save_path = Path(save_dir) / f'{nu}_mol'
                save_path.mkdir(parents=True, exist_ok=True)
                smiles_mapped, sorted_frag_smi, sorted_fragments, redundant_H_idx = split_with_input(smi, fragments_dict[nu], save_path)

                
                success_dict[nu] = len(sorted_frag_smi)
                frag_smi_dict[nu] = sorted_frag_smi
                sorted_fragments_dict[nu] = sorted_fragments
                redundant_H_dict[nu] = redundant_H_idx
        
        # Add already split fragments to the list of fragments
        if args.add_already_split:
            success_dict_old = pickle.load(open(f'{save_dir}/success_split.pkl', 'rb'))
            frag_smi_dict_old = pickle.load(open(f'{save_dir}/frag_smi_dict.pkl', 'rb'))
            sorted_fragments_dict_old = pickle.load(open(f'{save_dir}/sorted_fragments_dict.pkl', 'rb'))
            redundant_H_dict_old = pickle.load(open(f'{save_dir}/redundant_H_dict.pkl', 'rb'))
            
            success_dict = {**success_dict_old, **success_dict}
            frag_smi_dict = {**frag_smi_dict_old, **frag_smi_dict}
            sorted_fragments_dict = {**sorted_fragments_dict_old, **sorted_fragments_dict}
            redundant_H_dict = {**redundant_H_dict_old, **redundant_H_dict}
        
        # Save the result dict of fragmentation
        success_pkl = f'{save_dir}/success_split.pkl'
        f_save = open(success_pkl, 'wb')
        pickle.dump(success_dict, f_save)
        f_save.close()

        
        frag_smi_dict_pkl = f'{save_dir}/frag_smi_dict.pkl'
        f_save = open(frag_smi_dict_pkl, 'wb')
        pickle.dump(frag_smi_dict, f_save)
        f_save.close()
        
        sorted_fragments_dict_pkl = f'{save_dir}/sorted_fragments_dict.pkl'
        f_save = open(sorted_fragments_dict_pkl, 'wb')
        pickle.dump(sorted_fragments_dict, f_save)
        f_save.close()
        
        redundant_H_dict_pkl = f'{save_dir}/redundant_H_dict.pkl'
        f_save = open(redundant_H_dict_pkl, 'wb')
        pickle.dump(redundant_H_dict, f_save)
        f_save.close()
