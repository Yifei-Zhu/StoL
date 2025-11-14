from pathlib import Path
import pandas as pd
import pickle
import os
import numpy as np
import torch
import random
import tqdm
import argparse
import pickle

from utils.datasets import generate_data, read_xyz_block, _atoms_order
from utils.parse_xyz import parse_xyz_corpus
from typing import List
import yaml
from easydict import EasyDict

from utils.misc import seed_all, get_logger
from qcdge_related.props import PropsDict
from qcdge_related.scale_prop import PropDataScale

def random_split(data_list: List, train: float = 0.8, valid: float = 0.1, seed: int = 1234):
    """
    Randomly split a dataset into non-overlapping train/valid/test set.
    args :
        data_list (list): a list of data
        train (float): ratio of train data
        valid (float): ratio of valid data
        seed (int): random seed
    return :
        train_data (list): a list of train data
        valid_data (list): a list of valid data
        test_data (list): a list of test data
    """
    assert train + valid < 1
    random.seed(seed)
    random.shuffle(data_list)
    N = len(data_list)
    n_train = int(N * train)
    n_valid = int(N * valid)
    train_data = data_list[:n_train]
    valid_data = data_list[n_train: n_train + n_valid]
    test_data = data_list[n_train + n_valid:]

    return train_data, valid_data, test_data


def index_split(num_data: int, train: float = 0.8, valid: float = 0.1, seed: int = 1234):
    """
    Generate randomly splitted index of data into non-overlapping train/valid/test set.
    This function assume that the data is augmented so that original samples are placed in even index
    and the corresponding augmented samples are placed in the next index.
    args :
        num_data (int): the number of data of original samples
        train (float): ratio of train data
        valid (float): ratio of valid data
        seed (int): random seed
    return :
        train_index (list): a list of train index of original and augmented samples
        valid_index (list): a list of valid index of original and augmented samples
        test_index (list): a list of test index of original and augmented samples
    """
    assert train + valid < 1
    random.seed(seed)
    index_list = list(range(num_data))
    random.shuffle(index_list)

    n_train = int(num_data * train)
    n_valid = int(num_data * valid)
    train_index = np.array(index_list[:n_train])
    valid_index = np.array(index_list[n_train: n_train + n_valid])
    test_index = np.array(index_list[n_train + n_valid:])

    train_index = list(np.concatenate((train_index * 2, train_index * 2 + 1)))
    valid_index = list(np.concatenate((valid_index * 2, valid_index * 2 + 1)))
    test_index = list(np.concatenate((test_index * 2, test_index * 2 + 1)))

    train_index.sort()
    valid_index.sort()
    test_index.sort()
    return train_index, valid_index, test_index


def check_dir(dir_name):
    """
    Check the directory exists or not
    If not, make the directory
    Check wheather train_data.pkl, valid_data.pkl, test_data.pkl are exist or not.
    If exist, raise error.

    args :
       dir_name (str): directory name
    return :
         None
    """
    os.makedirs(dir_name, exist_ok=True)
    if os.path.isfile(os.path.join(dir_name, "train_data.pkl")):
        raise ValueError("train_data.pkl is already exist.")
    if os.path.isfile(os.path.join(dir_name, "valid_data.pkl")):
        raise ValueError("valid_data.pkl is already exist.")
    if os.path.isfile(os.path.join(dir_name, "test_data.pkl")):
        raise ValueError("test_data.pkl is already exist.")


def _get_priority_symbol(symbol_priority, atom):
    return symbol_priority.get(atom, 999)

def reorder_xyz_block(xyz_block):
    lines = xyz_block.strip("\n").split("\n")
    num_atoms = int(lines[0].strip())
    comment = lines[1]
    atomic_lines = lines[2:]
    if not atomic_lines[-1].strip():
        atomic_lines = atomic_lines[:-1]
    symbols = []
    pos = []
    for line in atomic_lines:
        parts = line.split()
        symbols.append(parts[0])
        coords = [float(x) for x in parts[1:4]]
        pos.append(coords)

    symbols = np.array(symbols)
    pos = np.array(pos)

    symbol_priority = _atoms_order()
    indices = sorted(range(len(symbols)), key=lambda i: _get_priority_symbol(symbol_priority, symbols[i]))
    symbols_new = symbols[indices]
    pos_new = pos[indices]
    new_lines = []
    new_lines.append(str(len(symbols_new)))
    new_lines.append(comment)
    for s, p in zip(symbols_new, pos_new):
        new_lines.append(f"{s:2s}{'':7s}{p[0]:10.6f}{p[1]:11.6f}{p[2]:11.6f}")

    new_xyz_block = "\n".join(new_lines)

    # print(new_xyz_block)
    return new_xyz_block

if __name__ == "__main__":

    config_path = os.path.join(os.getcwd(), "configs/data_config.yaml")
    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f)).Data
    remove_H = config.remove_H
    seed = config.prepreocessing.seed
    train = config.prepreocessing.train
    valid = config.prepreocessing.valid
    feat_dict = config.prepreocessing.feat_dict
    save_dir = config.prepreocessing.save_dir
    xyz_data = config.prepreocessing.xyz_data
    smiles_file = config.prepreocessing.smiles_file
    ban_index = config.prepreocessing.ban_index
    
 
    other_props = config.other_props.enable
    if other_props:
        qcdge = PropsDict()
        other_props = config.other_props.props.split(',')
        other_props = [int(x) for x in other_props]
        save_pkl = config.other_props.save_pkl
        f_read = open(save_pkl, 'rb')
        prop_dict = pickle.load(f_read)
        f_read.close()

        # ! current version only consider total GS energy
        if 3 in other_props and config.other_props.normalize_in_preprocessing.enable:
            prop_list = prop_dict['Etot']
            prop_list = np.abs(prop_list).tolist() # becase energy is negative
            
            # initialize PropDataScale Class
            scales = PropDataScale(prop_list)
            normalization_methods = {
                1: scales.min_max_normalization,
                2: scales.standardization,
                3: scales.robust_scaling,
                4: scales.log_transformation,
                5: scales.inverse_transform
            }
            scale_method = normalization_methods.get(config.other_props.normalize_in_preprocessing.scale)
            if scale_method:
                scaled_list = scale_method()
            else:
                raise ValueError(f"Invalid scale method: {config.other_props.normalize_in_preprocessing.scale}")

            prop_dict['Etot'] = scaled_list


 #1 - min_max_normalization 2- standardization 3- robust_scaling 4- log_transformation 5- inverse_transform

    #geometry data of xyz format.
    xyz_blocks = parse_xyz_corpus(xyz_data)

    # reaction smiles data of csv format.
    df = pd.read_csv(smiles_file)
    smiles_list = df.SMILES_RDKIT_CAN

    # set index of source data to be excluded
    if ban_index:
        ban_index = ban_index
    else:
        ban_index = []
        
    # set feature types
    # if there exist pre-defined feat_dict, load the feat_dict
    if os.path.isfile(feat_dict):
        feat_dict = pickle.load(open(feat_dict, "rb"))
    else:
        print(feat_dict, "is not exist. Use default feat_dict.")
        feat_dict = {
            "GetAtomicNum":{},
            "GetIsAromatic": {},
            "GetFormalCharge": {},
            "GetHybridization": {},
            "GetTotalNumHs": {},
            "GetNumImplicitHs":{},
            "GetTotalValence": {},
            "GetTotalDegree": {},
            "GetChiralTag": {},
            "IsInRing": {},
        }

    # generate torch_geometric.data.Data instance
    data_list = []
    for idx, (smiles, xyz_block) in tqdm.tqdm(enumerate(zip(smiles_list, xyz_blocks))):
        # print(xyz_block)
        xyz_block =  reorder_xyz_block(xyz_block)
        
        input_smi = smiles
        data, feat_dict = generate_data(input_smi, xyz_block, feat_dict = feat_dict, remove_H = remove_H)
        
        if other_props:
            add_props = {}
            for prop in other_props:
                variable_name = qcdge.get_key(str(prop))
                add_props[variable_name] = prop_dict[variable_name][idx]
            for key, value in add_props.items():
                setattr(data, key, value) 
    
        data_list.append(data)

    # convert features to one-hot encoding
    num_cls = [len(v) for k, v in feat_dict.items()]
    for data in data_list:
        feat_onehot = []
        feats = data.feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.feat = torch.cat(feat_onehot, dim=-1)

    train_index, valid_index, test_index = index_split(
        int(len(data_list) / 2),
        train=train,
        valid=valid,
        seed=seed
    )
    train_index = [i for i in train_index if i not in ban_index]
    valid_index = [i for i in valid_index if i not in ban_index]
    test_index = [i for i in test_index if i not in ban_index]

    train_data = [data_list[i] for i in train_index]
    valid_data = [data_list[i] for i in valid_index]
    test_data = [data_list[i] for i in test_index]
    index_dict = {
        "train_index": train_index,
        "valid_index": valid_index,
        "test_index": test_index,
    }

    check_dir(save_dir)

    # save the data, feat_dict, index_dict at the save_dir with pickle format. (.pkl)
    with open(os.path.join(save_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f, protocol=4)
    with open(os.path.join(save_dir, "valid_data.pkl"), "wb") as f:
        pickle.dump(valid_data, f, protocol=4)
    with open(os.path.join(save_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f, protocol=4)
    with open(os.path.join(save_dir, "feat_dict.pkl"), "wb") as f:
        pickle.dump(feat_dict, f, protocol=4)
    with open(os.path.join(save_dir, "index_dict.pkl"), "wb") as f:
        pickle.dump(index_dict, f, protocol=4)

    # Logging
    log_dir = save_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger("test", log_dir)
    logger.info(config)
