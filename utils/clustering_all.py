import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pathlib import Path
from utils.atom_matching import  hungarian_with_constraints
from utils.align_mol import kabsch_algorithm_classical


def align_mol_classical(coords1, coords2, atom_type1, atom_type2):
    reorder_coords2 = hungarian_with_constraints(coords1, coords2, atom_type1, atom_type2)
    aligned_coords2 = kabsch_algorithm_classical(reorder_coords2, coords1)
    return aligned_coords2

def hierarchical_clustering_main(ref_structure, structures_list, atom_type1, atom_type2_list, save_path, label_path, mask, n_clusters=None, thresh=None, centroids_xyz_path=None, dimensionlity_reduction=None):
    """Main function for hierarchical clustering:
        - dimensionlity_reduction: np.ndarray or None
    """

    cluster = ClusteringClass(data=dimensionlity_reduction, distance_matrix=None)

    cluster.calculate_distance_matrix(align_mol_classical, ref_structure, structures_list, atom_type1, atom_type2_list, reduced=dimensionlity_reduction)
    
    labels, linkage_matrix = cluster.hierarchical_clustering(method='ward', thresh=thresh, n_clusters=n_clusters)
    
    if isinstance(dimensionlity_reduction, np.ndarray):
        cluster.visualize(cluster_type='hierarchical', save_file = save_path)

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.savefig(save_path)  
    
    cluster.save_labels(label_path, mask)
    cluster_counts = cluster.summarize_clusters()
    cluster.save_hierarchical_centroids(structures_list, atom_type2_list, filename=centroids_xyz_path)
    return labels, cluster_counts

from sklearn_extra.cluster import KMedoids

def kmeans_main(ref_structure, structures_list, atom_type1, atom_type2_list, n_clusters, save_path, label_path, mask, centroids_xyz_path=None,dimensionlity_reduction=None):
    """Main function for KMeans clustering:
        - dimensionlity_reduction: np.ndarray or None
        
    # ! 2025.7.8 15:52 modify kmeans to KMedoids
    
    """

    if not isinstance(dimensionlity_reduction, np.ndarray):
        data = np.array([
            align_mol_classical(ref_structure, s, atom_type1, atom_type2).flatten()
            for s, atom_type2 in zip(structures_list, atom_type2_list)
        ])
    else:
        data = dimensionlity_reduction
    
    cluster = ClusteringClass(data=data,distance_matrix=None)
    labels = cluster.kmeans_clustering(n_clusters=n_clusters)
    
    if isinstance(dimensionlity_reduction, np.ndarray):
        cluster.visualize(cluster_type='kmeans', save_file = save_path)
        
    cluster.save_labels(label_path, mask)
    cluster_counts = cluster.summarize_clusters()
    cluster.save_kmeans_centroids_as_xyz(structures_list, atom_type2_list, centroids_xyz_path)
    return labels, cluster_counts

class ClusteringClass:
    def __init__(self, data=None, distance_matrix=None):
        """
        Initialize the class with data or a precomputed distance matrix.

        Parameters:
        - data: ndarray, original data points.
        - distance_matrix: ndarray, precomputed distance matrix.
        """
        self.data = data
        self.distance_matrix = distance_matrix
        self.labels = None


    def calculate_distance_matrix(self, align_function, ref_structure, structures_list, atom_type1, atom_type2_list,reduced):
        """
        Calculate a custom distance matrix using the provided alignment function.

        Parameters:
        - align_function: function, function to align molecules and compute distance.
        - ref_structure: array, reference structure.
        - structures_list: list, list of structures to align.
        - atom_type1: array, atom types for reference structure.
        - atom_type2_list: list, list of atom types for other structures.

        Returns:
        - distance_matrix: ndarray, pairwise distances.
        """
        if isinstance(reduced, np.ndarray):
            # Directly compute distance if data is already reduced
            self.distance_matrix = squareform(pdist(self.data))
        elif reduced == False:
            aligned_structures = np.array([
                align_function(ref_structure, s, atom_type1, atom_type2).flatten()
                for s, atom_type2 in zip(structures_list, atom_type2_list)
            ])
            self.distance_matrix = squareform(pdist(aligned_structures))
 
        else:
            raise ValueError("Invalid value for 'reduced'. Must be an ndarray or False.")
        
        return self.distance_matrix

    def kmeans_clustering(self, n_clusters):
        """
        Perform KMeans clustering.

        Parameters:
        - n_clusters: int, number of clusters.

        Returns:
        - labels: array, cluster labels for each data point.
        """
        if self.data is None:
            raise ValueError("Original data must be provided for KMeans clustering.")

        # self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        # self.labels = self.kmeans.fit_predict(self.data)
        # return self.labels

        self.kmedoids = KMedoids(
            n_clusters=n_clusters,
            random_state=0
        )
        self.labels = self.kmedoids.fit_predict(self.data)

        return self.labels

    def hierarchical_clustering(self, method='ward', n_clusters=None , thresh=None):
        """
        Perform hierarchical clustering.

        Parameters:
        - method: str, linkage method (e.g., 'ward', 'single', 'complete').
        - n_clusters: int, number of clusters.

        Returns:
        - labels: array, cluster labels for each data point.
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix must be provided for hierarchical clustering.")

        linkage_matrix = linkage(squareform(self.distance_matrix), method=method)
        if n_clusters:
            self.labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        else:
            self.labels = fcluster(linkage_matrix, t=thresh, criterion='distance')

        return self.labels, linkage_matrix
    
    def save_kmeans_centroids_as_xyz(self, structures_list, atom_type2_list, centroids_xyz_path):
        """
        Save KMeans cluster centroids as .xyz file using corresponding structures.

        Parameters:
        - kmeans_model: sklearn KMeans model after fitting.
        - structures_list: list, list of structures corresponding to original data.
        - atom_type2_list: list, list of atom types for the structures.
        - centroids_xyz_path: str, path to save the .xyz file.
        """
        # centroids_indices = self.kmeans.predict(self.kmeans.cluster_centers_)  # Find closest points

        centroids_indices = self.kmedoids.medoid_indices_ 

        with open(centroids_xyz_path, 'w') as xyz_file:
            for idx in centroids_indices:
                structure = structures_list[idx]
                atom_types = atom_type2_list[idx]
                num_atoms = len(atom_types)
                xyz_file.write(f"{num_atoms}\nCluster Center Structure {idx}\n")
                for atom_type, coords in zip(atom_types, structure):
                    xyz_file.write(f"{atom_type} {coords[0]} {coords[1]} {coords[2]}\n")

    def save_hierarchical_centroids(self, structures_list, atom_type2_list, filename="hierarchical_centroids.xyz"):
        """
        Save the hierarchical clustering cluster centers' corresponding structures to an .xyz file.

        Parameters:
        - structures_list: list, list of structures corresponding to the input data.
        - atom_type2_list: list, list of atom types for each structure.
        - filename: str, name of the .xyz file to save.
        """
        if self.data is None or self.labels is None:
            raise ValueError("Hierarchical clustering must be performed before saving centroids.")

        unique_labels = np.unique(self.labels)
        centroids_indices = []

        for label in unique_labels:
            # Get indices of points in the current cluster
            cluster_indices = np.where(self.labels == label)[0]
            
            # Calculate the centroid of the cluster
            cluster_points = self.data[cluster_indices]
            cluster_centroid = cluster_points.mean(axis=0)
            
            # Find the structure closest to the centroid
            distances = np.linalg.norm(cluster_points - cluster_centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            centroids_indices.append(closest_idx)

        # Write the closest structures to an .xyz file
        with open(filename, 'w') as xyz_file:
            for idx in centroids_indices:
                structure = structures_list[idx]
                atom_types = atom_type2_list[idx]
                num_atoms = len(atom_types)
                xyz_file.write(f"{num_atoms}\nCluster Center Structure {idx}\n")
                for atom_type, coords in zip(atom_types, structure):
                    xyz_file.write(f"{atom_type} {coords[0]} {coords[1]} {coords[2]}\n")


    def visualize(self, cluster_type='kmeans',save_file=None,):
        plt.figure()
        if cluster_type == 'kmeans':
            if self.data is None or not hasattr(self, 'labels'):
                raise ValueError("K-means visualization requires labels and raw data.")
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', s=50)
            # centers = self.kmeans.cluster_centers_
            centers = self.data[self.kmedoids.medoid_indices_]  
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=1)
            plt.title("K-means Clustering")
        elif cluster_type == 'hierarchical':
            if not hasattr(self, 'labels'):
                raise ValueError("Hierarchical clustering visualization requires labels.")
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', s=50)
            plt.title("Hierarchical Clustering")
            
        else:
            raise ValueError("Invalid cluster_type. Choose 'kmeans' or 'hierarchical'.")
        plt.savefig(save_file)
        plt.close()

    def save_labels(self, filename, mask):
        if not hasattr(self, 'labels'):
            raise ValueError("Labels are not available to save. Perform clustering first.")

        expanded_labels = np.full(len(mask), None, dtype=object)
        used_indices = np.where(mask)[0]  # Indices of used structures
        for i, idx in enumerate(used_indices):
            expanded_labels[idx] = self.labels[i]

        # Save expanded labels to .npy
        np.save(filename, expanded_labels)

        

    def summarize_clusters(self):
        # 统计每个类的分子数量
        cluster_counts = Counter(self.labels)
        return cluster_counts


def save_structures_by_cluster(labels, structures_list, atom_type2_list, output_dir):
    """
    Save structures belonging to each cluster into separate .xyz files.

    Parameters:
    - labels: list or ndarray, cluster labels for each structure.
    - structures_list: list, list of structures corresponding to original data.
    - atom_type2_list: list, list of atom types for the structures.
    - output_dir: str, directory to save the .xyz files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        centroids_xyz_path = os.path.join(output_dir, f"cluster_{cluster}.xyz")
        
        with open(centroids_xyz_path, 'w') as xyz_file:
            for idx in cluster_indices:
                structure = structures_list[idx]
                atom_types = atom_type2_list[idx]
                num_atoms = len(atom_types)
                xyz_file.write(f"{num_atoms}\nCluster {cluster} Structure {idx}\n")
                for atom_type, coords in zip(atom_types, structure):
                    xyz_file.write(f"{atom_type} {coords[0]} {coords[1]} {coords[2]}\n")

# 示例使用
if __name__ == "__main__":
    # 示例数据
    data = np.random.rand(100, 2)

    # 初始化聚类类
    clustering = Clustering(data=data)

    # K-means聚类
    labels_kmeans = clustering.kmeans(n_clusters=3)
    print("K-means labels:", labels_kmeans)
    clustering.visualize(cluster_type='kmeans')

    # 层级聚类
    labels_hierarchical = clustering.hierarchical(method='ward', n_clusters=3)
    print("Hierarchical labels:", labels_hierarchical)
    clustering.visualize(cluster_type='hierarchical')

    # 保存类名单
    clustering.save_labels("cluster_labels.csv")
