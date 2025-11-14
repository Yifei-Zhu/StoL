import numpy as np

import argparse
import pickle
import shutil
import yaml
import subprocess as sub
from tqdm.auto import tqdm
from easydict import EasyDict
from pathlib import Path
import signal
import time
from functools import wraps
import multiprocessing
from contextlib import contextmanager

from utils.split_mol_to_fragments import split_main_with_index
from utils.mapping_from_origianl_smiles_to_xyz import get_xyz_overlap_indices
from utils.misc import get_logger
from utils.frags_matching import assemble_frags

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # 注册SIGALRM信号处理器
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    # 设置闹钟
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 恢复原来的信号处理器
        signal.signal(signal.SIGALRM, original_handler)
        # 取消闹钟
        signal.alarm(0)

def retry_with_timeout(max_retries=5, timeout_seconds=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    with timeout(timeout_seconds):
                        result = func(*args, **kwargs)
                        return result
                except TimeoutException:
                    if attempt < max_retries - 1:
                        kwargs['print_fn'](f"Attempt {attempt + 1} timed out after {timeout_seconds}s, retrying...")
                        time.sleep(1)  # 短暂等待后重试
                    else:
                        kwargs['print_fn'](f"All {max_retries} attempts timed out")
                        return False
                except Exception as e:
                    kwargs['print_fn'](f"Error occurred: {str(e)}")
                    if attempt < max_retries - 1:
                        kwargs['print_fn']("Retrying...")
                        time.sleep(1)
                    else:
                        kwargs['print_fn']("Max retries reached")
                        return False
            return False
        return wrapper
    return decorator

# 使用装饰器包装split_main_with_index函数
@retry_with_timeout(max_retries=5, timeout_seconds=10)
def split_main_with_index_with_retry(*args, **kwargs):
    from utils.split_mol_to_fragments import split_main_with_index
    return split_main_with_index(*args, **kwargs)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--smis", type=str, help="smiles of mol needed to be splited", nargs="+", default=None)
    parser.add_argument("--smi_file", type=str, help="file of smiles of mol needed to be splited", default=None)
    parser.add_argument(
        "--from_scratch",
        type=str2bool,
        help="If True, it will start by splitting the original SMILES.",
        required=True
    )
    parser.add_argument("--save_dir", type=str, help="The directory to save the results")
    parser.add_argument("--ckpt", type=str, help="path for loading the checkpoint", nargs="+")

    #! The parameters are used only for testing
    parser.add_argument("--start_nu", type=int, help="If start_nu=2 and end_nu=4, means 2 3 4  ", default=None)
    parser.add_argument("--end_nu", type=int, help="If start_nu=2 and end_nu=4, means 2 3 4 ", default=None)

    args = parser.parse_args()

    config_path = Path.cwd() / "configs" / "sampling_config.yaml"

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = config_path.stem

    save_dir = config.io.save_dir if not args.save_dir else args.save_dir

    log_dir = save_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger("split_mol", log_dir)

    max_fragment_num = config.split_mol.max_fragment_num

    if args.from_scratch:
        smi_file = args.smi_file if args.smi_file else config.split_mol.smi_file
        smiles_list = args.smis if args.smis else read_smi_from_file(smi_file)
        
        if args.start_nu and args.end_nu:
            smiles_list=smiles_list[args.start_nu-1:args.end_nu]

        logger.info("Splitting mol to frags ...")
        success_dict={}
        failed_dict={}
        frag_smi_dict={}
        sorted_fragments_dict={}
        redundant_H_dict={}

        for i, smi in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Processing molecules"):
            # print(i)
            save_path=Path(save_dir) / f'{i}_mol'
            save_path.mkdir(parents=True, exist_ok=True)

            smiles_mapped, sorted_frag_smi, sorted_fragments, redundant_H_idx = split_main_with_index_with_retry(smi, save_path, max_fragment_num=max_fragment_num, print_fn=logger.info)

            if sorted_frag_smi:
                success_dict[i] = len(sorted_frag_smi)
                frag_smi_dict[i] = sorted_frag_smi
                sorted_fragments_dict[i] = sorted_fragments
                redundant_H_dict[i] = redundant_H_idx
            else:
                # logger.info(f"Molecule {smi} failed to be split into fragments.")
                failed_dict[i] = smi

        # In success_dict, key is the number of the mol and value is the number of fragments of the mol.
        success_pkl = f'{save_dir}/success_split.pkl'
        f_save = open(success_pkl, 'wb')
        pickle.dump(success_dict, f_save)
        f_save.close()
        logger.info(f"The result dict of fragmentation is saved to {success_pkl}")

        failed_pkl = f'{save_dir}/failed_split.pkl'
        f_save = open(failed_pkl, 'wb')
        pickle.dump(failed_dict, f_save)
        f_save.close()
        logger.info(f"The result dict of fragmentation is saved to {failed_pkl}")

        frag_smi_dict_pkl = f'{save_dir}/frag_smi_dict.pkl'
        f_save = open(frag_smi_dict_pkl, 'wb')
        pickle.dump(frag_smi_dict, f_save)
        f_save.close()
        logger.info(f"The result dict of fragmentation is saved to {frag_smi_dict_pkl}")

        sorted_fragments_dict_pkl = f'{save_dir}/sorted_fragments_dict.pkl'
        f_save = open(sorted_fragments_dict_pkl, 'wb')
        pickle.dump(sorted_fragments_dict, f_save)
        f_save.close()
        logger.info(f"The result dict of fragmentation is saved to {sorted_fragments_dict_pkl}")

        redundant_H_dict_pkl = f'{save_dir}/redundant_H_dict.pkl'
        f_save = open(redundant_H_dict_pkl, 'wb')
        pickle.dump(redundant_H_dict, f_save)
        f_save.close()
        logger.info(f"The result dict of fragmentation is saved to {redundant_H_dict_pkl}")

    else:
        # Load splited fragments
        success_pkl = f'{save_dir}/success_split.pkl'
        with open(success_pkl, 'rb') as f:
            success_dict = pickle.load(f)

        frag_smi_dict_pkl = f'{save_dir}/frag_smi_dict.pkl'
        with open(frag_smi_dict_pkl, 'rb') as f:
            frag_smi_dict = pickle.load(f)

        sorted_fragments_dict_pkl = f'{save_dir}/sorted_fragments_dict.pkl'
        with open(sorted_fragments_dict_pkl, 'rb') as f:
            sorted_fragments_dict = pickle.load(f)

        redundant_H_dict_pkl = f'{save_dir}/redundant_H_dict.pkl'
        with open(redundant_H_dict_pkl, 'rb') as f:
            redundant_H_dict = pickle.load(f)

    if args.start_nu and args.end_nu:        
        sampling_mols = list(success_dict.items())[args.start_nu-1:args.end_nu]
    else:
        sampling_mols = success_dict.items()

    skip_list = []
    for key, val in sampling_mols:
        if val > 4:
            skip_list.append(key)
            continue
        
        start_nu = 0
        end_nu = val
        sub_save_dir = Path(save_dir) / f'{int(key)}_mol'
        sampling_set = Path(sub_save_dir) / 'frag_smiles.dat'

        # Sampling for frags of each mol
        logger.info(f"Sampling for frags of {key} mol ...")
        cmd = f'python 3_sampling.py \
            --start_idx {start_nu} \
            --end_idx {end_nu} \
            --sampling_set {sampling_set} \
            --save_dir {sub_save_dir}'
        if args.ckpt:
            ckpt_str = ' '.join(str(x) for x in args.ckpt)
            cmd += f' --ckpt {ckpt_str}'
        sub.call(cmd, shell=True)
        logger.info(f'{key} mol frags sampling was completed.')

        # Post-sampling process
        logger.info(f"Post-sampling process for frags of {key} mol ...")
        # sub_save_dir = Path(save_dir) / f'{int(key)}_mol'
        sub.call(
            f'python 4_post_sampling.py \
                --save_dir {sub_save_dir}', shell=True
            )

    logger.info(f'These mol have more than 4 frags:\n{skip_list}')

    logger.info('The molecular fragment assembling is about to begin')

    if int(config.post_sampling.save_individual.clustering.enable) == 0:
        clustering_method_name = None
    else:
        clustering_method = int(config.post_sampling.save_individual.clustering.method)
        clustering_methods = {
            1: 'kmeans',
            2: 'hierarchical',
            }
        clustering_method_name = clustering_methods[clustering_method]


    # Fragment assembling
    # Iterate over the successfully fragmented molecule
    assemble_success_list = []
    error_list = []
    skip_list=[]
    for key, val in sampling_mols:
        if val > 4:
            skip_list.append(key)
            continue
        logger.info(f"Getting XYZ indies of fragmentations for {int(key)}_mol...")
        sub_save_dir = Path(save_dir) / f'{int(key)}_mol'

        try:
            # Get corresponding info
            sorted_fragments_i = sorted_fragments_dict[key]
            frag_smi_i = frag_smi_dict[key]
            redundant_H_idx = redundant_H_dict[key]

            overlap_dict = get_xyz_overlap_indices(sub_save_dir, sorted_fragments_i, frag_smi_i, redundant_H_idx, clustering_method_name, logger)

            overlap_pkl = f'{sub_save_dir}/overlap_dict.pkl'

            f_save = open(overlap_pkl, 'wb')
            pickle.dump(overlap_dict, f_save)
            f_save.close()
            logger.info(f"The XYZ indies of fragmentations of {int(key)}_mol is saved to {overlap_pkl}")

            logger.info(f"Assembling fragmentations of {int(key)}_mol according to common part...")

            assemble = assemble_frags(overlap_dict, sub_save_dir, clustering_method_name, threshold=0.2, print_fn=logger.info)
            if assemble:
                assemble_success_list.append(key)
        except:
            error_list.append(key)

    # The list of successfully assembled mols
    assemble_success_npy = f'{save_dir}/assemble_success.npy'
    np.save(assemble_success_npy, assemble_success_list)
    logger.info(f"The result dict of assemble is saved to {assemble_success_npy}")


    #!TEST
    print(f'TEST TEST TEST TEST: {error_list}')
    print(f'skip_list: {skip_list}')

