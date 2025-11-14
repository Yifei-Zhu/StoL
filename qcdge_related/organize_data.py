import os
import pickle
import numpy as np
import h5py
from tqdm import tqdm
from ase.data import chemical_symbols

def extract_data(hdf5_file, target_dict, prefix, mol_list=None, generateInput=False):
    # Ensure the data directory exists
    os.makedirs('./data', exist_ok=True)

    atoms_list, coords_list = [], []
    targets_list = []
    print('Start extracting data!\n')
    with h5py.File(hdf5_file, 'r') as f:
        for key in tqdm(mol_list, desc="Extracting data"):
            # Check if specific target states are provided
            if target_dict:
                for state, target_list in target_dict.items():
                    all_data = f[key][state]
                    if 'Info_of_AllExcitedStates' not in target_list:
                        targets = [all_data[target_name][()] for target_name in target_list]
                        targets_list.append(np.array(targets))
                    else:
                        targets = [all_data[target_name][()] for target_name in target_list if target_name != 'Info_of_AllExcitedStates']
                        excited_e = get_excited_state_contribution(f[key][state]['Info_of_AllExcitedStates'])
                        targets.append(excited_e)
                        targets_list.append(np.array(targets))
            else:
                all_data = f[key]['ground_state']

            if generateInput:
                atoms = all_data['labels'][()][0]
                labels = np.asarray([chemical_symbols[int(num)] for num in atoms])
                atoms_list.append(labels)
                coords = all_data['coords'][()]
                coords_list.append(coords)

    # Save data only if generateInput is True
    if generateInput:
        with open(f'./data/{prefix}_atoms.pkl', 'wb') as af:
            pickle.dump(atoms_list, af)
        with open(f'./data/{prefix}_coords.pkl', 'wb') as cf:
            pickle.dump(coords_list, cf)

    # Always save targets data
    with open(f'./data/{prefix}_targets.pkl', 'wb') as tf:
        pickle.dump(targets_list, tf)
    print(targets_list)
    print("Data extraction complete.\n")


def get_excited_state_contribution(info):
    info_dict=eval(info[()][0])
    for w in range(1,21):
        if info_dict[str(w)]['state_type'] == 'Singlet':
            first_e1=float(info_dict[str(w)]['excitation_e_eV'].split()[0])
            break
    return first_e1



if __name__ == '__main__':
    # for hdf5_file in ['./9_CNOF_pub_molecules.hdf5', './10_CNOF_pub_molecules.hdf5','./qm9_props.hdf5', './gdb11_props.hdf5']:
    #     prefix=hdf5_file.split('/')[-1].split('_')[0]
    #     target_dict = {'excited_state':['Info_of_AllExcitedStates']}
    #     extract_data(hdf5_file, target_dict, prefix, generateInput=False)

    hdf5_file = '/data/home/zhuyf/dataset_work/database/checkOptedGeoms/final_all.hdf5'
    prefix=hdf5_file.split('/')[-1].split('_')[0]
    print(prefix)
    target_dict = {'excited_state':['Info_of_AllExcitedStates']}
    extract_data(hdf5_file, target_dict, prefix, generateInput=False,mol_list=['Ba000010932','Ba000010932'])