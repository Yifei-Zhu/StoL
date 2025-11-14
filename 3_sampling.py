
import sys
import numpy as np
import argparse
import pickle
import torch
import yaml
from tqdm.auto import tqdm
from easydict import EasyDict
torch.serialization.add_safe_globals([EasyDict])
from pathlib import Path

from models import sampler
from torch_geometric.transforms import Compose
from models.epsnet import get_model
from utils.datasets import generate_data
from utils.transforms import CountNodesPerGraph
from utils.misc import seed_all, get_logger
from torch_geometric.data import Batch



def repeat(iterable, num):
    new_iterable = []
    for idx, x in enumerate(iterable):
        for _ in range(num):
            new_item = x.clone()
            new_item.mol_id = idx  # 为每个分子添加一个唯一的 ID
            new_iterable.append(new_item)
    return new_iterable


def batching(iterable, batch_size, repeat_num=1):
    cnt = 0
    iterable = repeat(iterable, repeat_num)
    while cnt < len(iterable):
        if cnt + batch_size <= len(iterable):
            yield iterable[cnt: cnt + batch_size]
            cnt += batch_size
        else:
            yield iterable[cnt:]
            cnt += batch_size


def preprocessing(smiles_list, feat_dict_path="feat_dict.pkl"):
    feat_dict = pickle.load(open(feat_dict_path, "rb"))

    data_list = []
    for smi in smiles_list:
        data, _ = generate_data(smi, None, feat_dict=feat_dict)
        data_list.append(data)
    num_cls = [len(v) for k, v in feat_dict.items()]
    for data in data_list:
        feat_onehot = []
        feats = data.feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.feat = torch.cat(feat_onehot, dim=-1)

    return data_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, help="path for loading the checkpoint", nargs="+")
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--sampling_set", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config_path = Path.cwd() / "configs" / "sampling_config.yaml"

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = config_path.stem

    # Model parameters
    ckpt = args.ckpt if args.ckpt else [checkpoint.strip() for checkpoint in config.sampling.model.checkpoint.split(',')]

    device = config.sampling.model.device
    batch_size = config.sampling.model.batch_size

    # IO parameters
    save_traj = config.io.save_traj
    save_dir = args.save_dir if args.save_dir else config.io.save_dir 
    
    save_individual = config.io.save_individual
    repeats = config.io.repeats
    if not save_individual:
        assert repeats == 1, "If you want sample several confs for each mol, you should enable save_individual."

    # Test data parameters
    feat_dict = config.sampling.sampling_data.feat_dict
    sampling_set = args.sampling_set if args.sampling_set else config.sampling.sampling_data.sampling_set
    start_idx = args.start_idx if args.start_idx else config.sampling.sampling_data.start_idx
    end_idx = args.end_idx if args.end_idx else config.sampling.sampling_data.end_idx

    # Sampling parameters
    clip = float(config.sampling.sampling.clip)
    n_steps = int(config.sampling.sampling.n_steps)

    # Parameters for DDPM
    sampling_type = config.sampling.parameters.sampling_type
    eta = float(config.sampling.parameters.eta)
    step_lr = float(config.sampling.parameters.step_lr)
    seed = int(args.seed) if args.seed else config.sampling.parameters.seed

    # Logging
    # log_dir = save_dir
    log_dir = save_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger("test", log_dir)
    logger.info(args)

    # Load checkpoint
    logger.info("Loading model...")
    ckpts = [torch.load(x) for x in ckpt]
    models = []

    # Check if all checkpoint files exist
    for checkpoint in ckpt:
        if not Path(checkpoint).exists():
            logger.error(f"Checkpoint file {checkpoint} does not exist!")
            sys.exit(1)


    for i, (ckpt, ckpt_path) in enumerate(zip(ckpts, ckpt)):
        # print(ckpt["config"].model)
        logger.info(f"load {i+1}/{len(ckpts)} model from {ckpt_path}")
        model = get_model(ckpt["config"].model).to(device)
        model.load_state_dict(ckpt["model"])
        models.append(model)
        
    model = sampler.EnsembleSampler(models).to(device)
    seed_all(seed)

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = Compose([CountNodesPerGraph(), ])

    # load data
    if ".dat" in sampling_set or ".txt" in sampling_set or ".pck" in sampling_set or ".pkl" in sampling_set:
        if not Path(sampling_set).is_file():
            logger.info(f"!!!Sampling file {sampling_set} is not found!!!\n" * 3)
            exit()
        elif ".txt" in sampling_set or ".dat" in sampling_set:
            logger.info(f"Sampling file from {sampling_set}.\n Processing smiles...")
            smiles_list = open(sampling_set, "r").read().strip().split("\n")
            sampling_set = preprocessing(smiles_list, feat_dict_path=feat_dict)
        else:
            logger.info(f"Sampling file from {sampling_set}.\n Loading dataset...")
            sampling_set = pickle.load(open(sampling_set, "rb"))
    else:
        logger.info(f"Test smiles : {sampling_set}.\n Processing smiles...")
        smiles_list = [sampling_set]
        sampling_set = preprocessing(smiles_list, feat_dict_path=feat_dict)

    sampling_set_selected = []
    nu_list = []
    for i, data in enumerate(sampling_set):
        if not (start_idx <= i < end_idx):
            continue
        sampling_set_selected.append(data)
        nu_list.append(i)


    results = []
    all_molecule_conformations = {}

    for i, batch in tqdm(enumerate(batching(sampling_set_selected, batch_size, repeat_num=repeats))):
        batch = Batch.from_data_list(batch).to(device)
        for _ in range(2):  # Maximum number of retry
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(device)

                pos_gen, pos_gen_traj = model.dynamic_sampling(
                    atom_type=batch.atom_type,
                    feat=batch.feat,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=True,  # Done in transforms.
                    n_steps=n_steps,
                    step_lr=step_lr,
                    clip=clip,
                    sampling_type=sampling_type,
                    eta=eta,
                    mode="edge",
                )
                alphas = model.alphas.detach()

                alphas = alphas[model.num_timesteps - n_steps: model.num_timesteps]
                alphas = alphas.flip(0).view(-1, 1, 1)
                pos_gen_traj_ = torch.stack(pos_gen_traj) * alphas.sqrt().cpu()

                for j, data in enumerate(batch.to_data_list()):
                    mask = batch.batch == j
                    if save_traj:
                        data.pos_gen = pos_gen_traj_[:, mask]
                    else:
                        data.pos_gen = pos_gen[mask]

                    data = data.to("cpu")
                    if save_individual:
                        mol_id = int(data.mol_id)
                        if mol_id not in all_molecule_conformations:
                            all_molecule_conformations[mol_id] = []
                        all_molecule_conformations[mol_id].append(data)
                    else:
                        results.append(data)
                break
            except FloatingPointError:
                clip = 20
                logger.warning("Retrying with clipping thresh 20.")

    if save_individual:

        processed_molecules = set() # Record the mol_id that has been processed 

        for mol_id, conformations in all_molecule_conformations.items():
            mol_id = int(mol_id)
            save_path = Path(log_dir) / f'mol{nu_list[mol_id]}' / f"molecule_{nu_list[mol_id]}_samples.pkl"
            
            # Delete old files only on first processing
            if mol_id not in processed_molecules:
                if save_path.exists():
                    save_path.unlink()
                    # logger.info(f"Deleted existing file: {save_path}")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                processed_molecules.add(mol_id)

            conformations_list = list(conformations)

            with open(save_path, "wb") as f:
                pickle.dump(conformations_list, f)
            logger.info(f"Saved samples for molecule {mol_id} to: {save_path}")
        
        nu_path = Path(log_dir) / 'nu_list.dat'  
        np.savetxt(nu_path, np.array(nu_list))

    else:
        save_path = Path(log_dir) /"samples_all.pkl"
        if save_path.exists():
            save_path.unlink()
            # logger.info(f"Deleted file: {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Saved all samples to: {save_path}")
