import numpy as np
import math

import umap
from sklearn.manifold import TSNE

from utils.atom_matching import  hungarian_with_constraints
from utils.align_mol import kabsch_algorithm_classical


def align_mol_classical(coords1, coords2, atom_type1, atom_type2):
    reorder_coords2 = hungarian_with_constraints(coords1, coords2, atom_type1, atom_type2)
    aligned_coords2 = kabsch_algorithm_classical(reorder_coords2, coords1)
    return aligned_coords2


from sklearn.manifold import TSNE

def calculate_distance_matrix(structures):
    """
    Calculate pairwise distance matrix for a list of aligned 3D structures.
    
    Args:
        structures (list of list or np.ndarray): A list of 3D positions for structures.
    
    Returns:
        np.ndarray: Pairwise distance matrix of shape (num_structures, num_structures).
    """
    # Ensure all structures are NumPy arrays and have the same dimensionality

    structures = [np.asarray(structure).reshape(-1, 3) for structure in structures]

    num_structures = len(structures)
    dist_matrix = np.zeros((num_structures, num_structures), dtype=np.float32)

    for i in range(num_structures):
        for j in range(i + 1, num_structures):
            # Flatten to compare structures as 1D vectors
            dist = np.linalg.norm(structures[i].flatten() - structures[j].flatten())
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix


import matplotlib.pyplot as plt

def visualize_embedding(embedding, save_path, compare, method):
    plt.figure()
    if not compare:
        plt.scatter(embedding[:, 0], embedding[:, 1])
    elif compare:
        nu = math.ceil(len(embedding)/2)
        plt.scatter(embedding[:nu, 0], embedding[:nu, 1], c='r',label='diff')
        plt.scatter(embedding[nu:, 0], embedding[nu:, 1], c='blue',label='rdkit')
        plt.legend()
        save_path = save_path.parent / f"compare_{save_path.name}"

    plt.title(f'{method} Visualization of Molecular Conformations')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(save_path)
    plt.close()

def dimensionality_reduction_main(ref_structure, structures_list, atom_type1, atom_type2_list, save_path, compare=False, method='tsne'):
    aligned_structures = [
        align_mol_classical(ref_structure, s, atom_type1, atom_type2) 
        for s, atom_type2 in zip(structures_list, atom_type2_list)
    ]
    dis_m = calculate_distance_matrix(aligned_structures)
    if method == 'tsne':
        #T-SNE can only be performed upon all data
        tsne = TSNE(metric='precomputed')
        embedding = tsne.fit_transform(dis_m)
    elif method == 'umap':
        # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        
        if not isinstance(compare, bool):  # Compare specified as number of diffusion structures
            num_diffusion = compare
            embedding_diffusion = reducer.fit_transform(dis_m[:num_diffusion, :num_diffusion])
            embedding_rdkit = reducer.transform(dis_m[num_diffusion:, :num_diffusion])
            embedding = np.vstack((embedding_diffusion, embedding_rdkit))
        else:  # Compare is False or not specified
            embedding = reducer.fit_transform(dis_m)

    visualize_embedding(embedding, save_path, compare, method)
    return embedding, dis_m