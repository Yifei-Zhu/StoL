import argparse
import pickle
import shutil
import yaml
import numpy as np
from tqdm.auto import tqdm
from easydict import EasyDict
from pathlib import Path

from qcdge_related.extract_data import extractData as etd

from utils.torch_geometric_to_xyz import pkl_to_xyz
from utils.check_generated_strus import check_from_pkl
from utils.dimensionality_reduction import dimensionality_reduction_main
from utils.clustering_all import hierarchical_clustering_main, kmeans_main, save_structures_by_cluster

from utils.misc import (
    seed_all,
    get_logger,
)

from utils.generated_stru_from_smiles import generate_ordered_structure, generate_confs


def read_multi_xyz(filename):
    """
    读取包含多个结构的XYZ文件，返回结构字典列表    
    返回:
        list: 包含多个结构字典的列表，每个字典格式为:
              {'atom_type': atom_labels, 'positions': positions}
    """
    structures = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            num_atoms = int(line.strip())
            comment = f.readline()
            atom_labels = []
            positions = []
            for _ in range(num_atoms):
                parts = f.readline().split()
                atom_labels.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
                
            positions_array = np.array(positions, dtype=np.float64)
 
            structures.append({
                'atom_type': atom_labels,
                'positions': positions_array
            })
    
    return structures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config_path = Path.cwd() / "configs" / "sampling_config.yaml"

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = config_path.stem

    # IO parameters
    save_dir = args.save_dir if args.save_dir else config.io.save_dir

    nu_path = Path(save_dir) / 'nu_list.dat'
    nu_list = np.loadtxt(nu_path)
    nu_list=[int(nu) for nu in nu_list]

    save_individual = config.io.save_individual
    repeats = config.io.repeats
    if not save_individual:
        assert repeats == 1, "If you want sample several confs for each mol, you should enable save_individual."

    # option of post-sampling-process
    tol = config.post_sampling.bond_filter_tolerance

    transform_to_xyz = int(config.post_sampling.transform_to_xyz)
    transform_mapping = {1: False, 2: True}  # Map transform_to_xyz values to all_transform

    dm_type_option = config.post_sampling.save_individual.dimensionality_reduction.method
    dm_types = {
                1:'tsne',
                2:'umap',
                }
    dm_type=dm_types[int(dm_type_option)]

    clustering_enable = int(config.post_sampling.save_individual.clustering.enable)
    clustering_enable = int(clustering_enable) if clustering_enable != 0 else False
    if clustering_enable:
        clustering_method = int(config.post_sampling.save_individual.clustering.method)
        clustering_methods = {
                                1: 'kmeans',
                                2: 'hierarchical',
                                }

        if clustering_method not in clustering_methods:
            raise ValueError(f"Invalid clustering method {clustering_method}. Valid methods are {list(clustering_methods.keys())}.")
        clustering_method_name = clustering_methods[clustering_method]

        if not hasattr(config.post_sampling.save_individual.clustering.parameters, clustering_method_name):
            raise AttributeError(f"Clustering parameters for {clustering_method_name} not found in the configuration.")
        clustering_params = getattr(config.post_sampling.save_individual.clustering.parameters, clustering_method_name)

    log_dir=save_dir
    logger = get_logger("post_sampling", log_dir)
    logger.info("Filtering strus ...")

    #Checking its chemical validity by bond matrices
    if save_individual:
        logger.info("Filtering in individual floder.")
        # invalid_nu = []

        masks = {}

        # check for each frags
        for nu in nu_list:
            pkl_file = Path(save_dir) / f'mol{nu}' / f"molecule_{nu}_samples.pkl"
            mask, nums = check_from_pkl(pkl_file, tolerance=tol)
            masks[nu] = mask
            valid_nu = sum(mask)
            # if valid_nu < max(nums*0.2, 20):
            #     invalid_nu.append(nu)

            logger.info(f"valid strus of mol {nu}: {valid_nu}/{nums}={valid_nu/nums:.3f}")
        # if invalid_nu:
        #     logger.info(f"!!! Invalid mol: {invalid_nu}")

        # Transfrom PyG data to xyz file
        # 0 - do not read stru from pkl 1 - all confs in one xyz 2 - one conf in one xyz
        strus_dict_list=[]
        for nu in nu_list:
            pkl_file = Path(save_dir) / f'mol{nu}' / f"molecule_{nu}_samples.pkl"
            strus = pkl_to_xyz(pkl_file, nu, masks[nu], all_transform=transform_mapping[transform_to_xyz])
            logger.info(f"Confs of mol {nu} are saved to xyz file.")

            ref = strus[list(strus.keys())[0]]
            ref_atom = ref['atom_type']
            # ref_structure = ref['positions']
            ref_structure = generate_ordered_structure(ref['smiles'], ref_atom)
            # print(ref['smiles'])
            # T-SNE dimentionality reduction
            atom_type_list=[]
            strus_list = []
            for _, dict_i in strus.items():
                atom_type_list.append(dict_i['atom_type'])
                strus_list.append(dict_i['positions'])

            #Test: compare with rdkit
            if config.post_sampling.save_individual.test.compare_with_rdkit:
                nu_cof = len(strus_list)
                rdkit_list = generate_confs(ref['smiles'], ref_atom, num_confs=nu_cof)
                strus_list += rdkit_list
                atom_type_list += atom_type_list
                compare = nu_cof
            else:
                compare = config.post_sampling.save_individual.test.compare_with_rdkit
            try:
                fig_path = Path(save_dir) / f'mol{nu}' / f"{dm_type}_molecule_{nu}.png"
                embedding, dis_m = dimensionality_reduction_main(ref_structure, strus_list, ref_atom, atom_type_list, fig_path, compare = compare, method = dm_type)
                low_dim_file = Path(save_dir) / f'mol{nu}' / f"{dm_type}_low_dim.npy"
                np.save(low_dim_file, embedding)
                dis_m_file = Path(save_dir) / f'mol{nu}' / f"{dm_type}_distance_m.npy"
                np.save(dis_m_file, dis_m)

                if not isinstance(compare, bool):
                    logger.info(f'{dm_type} method was performed upon diffusion and rdkit samples')
                else:
                    logger.info(f'{dm_type} method was performed upon diffusion samples')
                    if clustering_enable:
                        distance_threshold = clustering_params.get('distance_threshold', None)
                        n_clusters = clustering_params.get('n_clusters', None)

                        fig_path = Path(save_dir) / f'mol{nu}' / f"clustering_{nu}_{clustering_method_name}.png"
                        label_path = Path(save_dir) / f'mol{nu}' / f"label_clustering_{nu}_{clustering_method_name}.npy"
                        centroids_xyz_path = Path(save_dir) / f'mol{nu}' / f"centroids_{nu}_{clustering_method_name}.xyz"

                        dimensionlity_reduction = {1: embedding, 2: False}.get(clustering_enable)
                        assert dimensionlity_reduction is not None, 'clustering_enable must be 1 or 2' 

                        if clustering_method == 1:
                            labels, cluster_counts = kmeans_main(
                                ref_structure, strus_list, ref_atom, atom_type_list, n_clusters, fig_path, label_path, masks[nu], centroids_xyz_path=centroids_xyz_path,dimensionlity_reduction=dimensionlity_reduction
                                )

                        elif clustering_method == 2:
                            labels, cluster_counts = hierarchical_clustering_main(
                                ref_structure, strus_list, ref_atom, atom_type_list, fig_path, label_path, masks[nu], n_clusters=n_clusters, thresh=distance_threshold,centroids_xyz_path=centroids_xyz_path,  pidimensionlity_reduction=dimensionlity_reduction
                                )

                        if isinstance(dimensionlity_reduction, np.ndarray):
                            target_str = f'low_dim embedding from {dm_type}'
                        else:
                            target_str = f'original strus'

                        logger.info(f'Clustering method: {clustering_method_name}, Target Mol: {nu}, Data: {target_str}')
                        logger.info(f"Total number of clusters: {len(cluster_counts)}")
                        logger.info(f"Number of mol {nu} in each cluster:")
                        # for cluster_id, count in cluster_counts.items():
                        #     logger.info(f"  Cluster {cluster_id}: {count} molecules")
                        logger.info(f'Expanded labels saved to {label_path}')
                        logger.info(f"Clustering figure of mol {nu} is saved to {fig_path}.")

                        output_dir = Path(save_dir) / f'mol{nu}' /'cluster_xyz'
                        if output_dir.exists() and output_dir.is_dir():
                            shutil.rmtree(output_dir)
                        output_dir.mkdir(exist_ok=True)
                        save_structures_by_cluster(labels, strus_list, atom_type_list, output_dir)
                        logger.info(f"Strus of each cluster of mol {nu} is saved to {output_dir}.")
            except:
                logger.info(f'{dm_type} method failed.')
                continue
    else:
        # Filtering mols
        logger.info("Filtering in main floder.")
        pkl_file = Path(save_dir) / "samples_all.pkl"
        tol=0.3
        masks, nums = check_from_pkl(pkl_file, tolerance=tol)
        valid_nu = sum(masks)
        logger.info(f"Valid strus of : {valid_nu}/{nums}={valid_nu/nums:.3f}")

        # Save strus to one xyz file
        nu = 'all'
        pkl_file = Path(save_dir) / "samples_all.pkl"
        if transform_to_xyz !=0 :
            pkl_to_xyz(pkl_file, nu, masks, all_transform =transform_mapping.get(transform_to_xyz, False))
            logger.info(f"Strus are saved to {pkl_file}.")
            print(f"XYZ file is saved.")
        logger.info('Only confs saved in a individual folder for each mol can perform clustering.')

