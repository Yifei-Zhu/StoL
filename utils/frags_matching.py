import os 
from pathlib import Path
import math
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

def read_xyz(fileA):
    with open(fileA, 'r') as f1:
        n_atom = int(f1.readline())
        atom=[]
        coor = np.empty((n_atom, 3), dtype=np.float64)
        line_nu = -1
        for line in f1:
            if line.split() and line.split()[0].isalpha():
                line_nu += 1
                atom.append(line.split()[0])
                for i in range(3):
                    # print(line.split()[i+1])
                    coor[line_nu, i] = line.split()[i+1]
    return coor, atom

def centroid(coords):
    """计算坐标的重心。"""
    return np.mean(coords, axis=0)

def kabsch_rotate(P, Q):
    """
    使用Kabsch算法计算最佳旋转矩阵，将Q对齐到P。
    
    参数:
        P: 目标点集 (N x 3)
        Q: 需要旋转的点集 (N x 3)
    
    返回:
        R: 旋转矩阵 (3 x 3)
        Q_rot: 旋转后的Q点集 (N x 3)
    """
    # 计算重心
    centroid_P = centroid(P)
    centroid_Q = centroid(Q)
    
    # 中心化
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 计算协方差矩阵
    C = np.dot(Q_centered.T, P_centered)
    
    # SVD
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    
    # 旋转矩阵
    R = np.dot(V, Wt)
    
    # 应用旋转
    Q_rot = np.dot(Q_centered, R)
    
    return R, Q_rot

def cal_rmsd(P, Q):
    """计算两个点集之间的均方根偏差(RMSD)。"""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))

def to_xyz(file1, file3, P):
    with open(file1,'r')as f1, open(file3,'w') as f2:
        f2.write(f'{f1.readline()}')
        f2.write(f'{f1.readline()}')

        line_nu=0
        for line in f1:
            try:
                _ = line.split()[0]
            except:
                break
            if line.split()[0].isalpha():
                f2.write(f'{line.split()[0]}\t')
                f2.write(f'{str(P[line_nu]).replace("[", "").replace("]", "")}\n')
                line_nu+=1



def read_multi_molecule_xyz(xyz_file_path):
    structures = []
    with open(xyz_file_path, 'r') as file:
        lines = file.readlines()
    i = 0
    total_lines = len(lines)
    while i < total_lines:
        num_atoms = int(lines[i].strip())
        i += 1
        comment = lines[i].strip()
        i += 1
        elements = []
        coordinates = []
        
        for _ in range(num_atoms):
            parts = lines[i].strip().split()        
            element = parts[0]
            x, y, z = map(float, parts[1:4])
            elements.append(element)
            coordinates.append([x, y, z])
            i += 1
        coordinates = np.array(coordinates)
        structures.append({
            'elements': elements,
            'coordinates': coordinates,
            'comment': comment,
        })
    return structures

def generate_new_xyz(overlap, stru):
    rotable_part=stru[overlap]
    mask = np.ones(stru.shape[0], dtype=bool)
    mask[overlap] = False
    other_part=stru[mask]
    reconstructed = np.empty_like(stru)
    reconstructed[mask] = other_part
    reconstructed[overlap] = rotable_part
    return reconstructed, rotable_part, other_part


def mirror_structure_axis(stru, axis='z'):
    mirrored = np.copy(stru)
    mirrored[:, 2] = -mirrored[:, 2]
    return mirrored

def kabsch_calculation(ref_all, ref_part, overlap1, overlap2, stru2):
    """
    Perform Kabsch alignment of a structure to a reference structure.

    This function aligns `stru2` to the reference structure (`ref_all`) using the Kabsch algorithm
    based on overlapping atoms defined by `overlap1` and `overlap2`. It returns the RMSD of the
    alignment and the newly aligned coordinates.

    Parameters:
    ----------
    ref_all : np.ndarray
        All atomic coordinates of the reference structure. Shape: (N_ref, 3)
    ref_part : np.ndarray
        Coordinates of the overlapping subset of atoms in the reference structure.
        Shape: (N_overlap, 3)
    overlap1 : list or array-like
        Indices of the overlapping atoms in `ref_all`.
    overlap2 : list or array-like
        Indices of the overlapping atoms in `stru2`.
    stru2 : np.ndarray
        All atomic coordinates of the structure to be aligned. Shape: (N_struct, 3)

    Returns:
    -------
    rmsd : float
        Root Mean Square Deviation (RMSD) between the aligned overlapping atoms.
    new_xyz : np.ndarray
        Aligned atomic coordinates of `stru2`. Shape: (N_struct, 3)
    """
    
    # Generate subsets of the target structure:
    # - all2: All atomic coordinates in stru2
    # - xyz2: Coordinates of the overlapping atoms in stru2
    # - others2: Coordinates of the non-overlapping atoms in stru2
    all2, xyz2, others2 = generate_new_xyz(overlap2, stru2)
    
    # Calculate the centroid of the overlapping atoms in stru2
    xyz2_center = centroid(xyz2)
    
    # Translate overlapping atoms to center them at the origin
    xyz2 -= xyz2_center

    all2 -= xyz2_center
    others2 -= xyz2_center
    
    # Compute the optimal rotation matrix using the Kabsch algorithm
    # This aligns the overlapping atoms of stru2 (xyz2) to the reference overlapping atoms (ref_part)
    rotator, _ = kabsch_rotate(ref_part, xyz2)
    
    # Apply the rotation matrix to the overlapping atoms
    xyz2 = np.dot(xyz2, rotator)
    
    # Apply the same rotation matrix to the non-overlapping atoms to maintain structural integrity
    others2 = np.dot(others2, rotator)
    
    # Initialize an empty array to hold the reconstructed aligned structure
    reconstructed = np.empty_like(all2)
    
    # Insert the rotated overlapping atoms back into their original positions
    reconstructed[overlap2] = xyz2
    
    # Create a boolean mask to identify non-overlapping atom positions
    mask = np.ones(all2.shape[0], dtype=bool)
    mask[overlap2] = False
    
    # Insert the rotated non-overlapping atoms into their original positions
    reconstructed[mask] = others2
    
    # The fully reconstructed aligned structure
    new_xyz = reconstructed
    
    # Calculate RMSD between the reference overlapping atoms and the aligned overlapping atoms
    rmsd = cal_rmsd(ref_all[overlap1], new_xyz[overlap2])
    
    # Return the calculated RMSD and the aligned coordinates
    return rmsd, new_xyz

def remove_common_part(ref_stru, atoms, related_h_dict, overlaps):
    remove_list = []
    for i in overlaps:
        remove_list.append(int(i))
        h_list = related_h_dict[int(i)]
        remove_list += [int(v) for v in h_list]
        
    rows_to_delete = np.array(remove_list)
    mask = np.ones(len(atoms), dtype=bool)
    mask[rows_to_delete] = False
    new_atom = np.array(atoms)[mask].tolist()
    new_atom_check = np.array(atoms)[~mask].tolist()
    
    new_redundant = []
    deleted_indices = np.where(~mask)[0]
    for i, atom in zip(deleted_indices, new_atom_check):
        if atom == 'R':
            for key, val in related_h_dict.items():
                if i in val:
                    new_redundant.append(overlaps.index(key))
                    break

    new_stru = np.delete(ref_stru, rows_to_delete, 0)

    return new_stru, new_atom, new_redundant


def index_nonadjacent_common_part(atoms, related_h_dict, overlaps):
    """
    Mark the part that overlaps with non adjacent frags in the second frag
    """
    index_list = []
    for i in overlaps:
        index_list.append(int(i))
        h_list = related_h_dict[int(i)]
        index_list += [int(v) for v in h_list]
    for j in index_list:
        atoms[j] = 'R'
    return atoms


def remove_redundant_H(xyz, atom2, related_h_dict, ref_atom_redundant_H_coord, atom_redundant_H_i):
    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    # There may be multiple broken site here.
    all_h_list=[]
    all_h_list += [related_h_dict[i] for i in atom_redundant_H_i]
    for i, h_list in enumerate(all_h_list):
        atom_with_redundant_coord = xyz[atom_redundant_H_i][i]
        similarity = -99999
        for h_idx in h_list:
            h_coord =  xyz[h_idx]
            vec1 = ref_atom_redundant_H_coord[i] - atom_with_redundant_coord
            vec2 = h_coord - atom_with_redundant_coord
            similarity_H = cosine_similarity(vec1, vec2)
            if abs(similarity_H) > similarity:
                similarity = similarity_H
                if atom2[h_idx] != 'R':
                    remove_idx = h_idx
        atom2[remove_idx] = 'R'

    return atom2

def align_xyz_single(overlap1, stru1, overlap_second, cur_strus, related_h_first, related_h_second_list, ref_atom_redundant_H_i,atom_redundant_H,need_remove_after_combine_i):
    atom1=stru1['elements']
    stru1=stru1['coordinates']

    ref_all, ref_part, _ = generate_new_xyz(overlap1, stru1)
    center = centroid(ref_part)
    ref_all -= center
    ref_part -= center

    assembles = []
    rmsds=[]
    for idx, overlap2 in enumerate(overlap_second):
        if overlap2:

            # Must copy!!!
            stru2 = cur_strus[idx]['coordinates'].copy()
            atom2 = cur_strus[idx]['elements'].copy()
            # print(atom2.count('R'))

            rmsd1, new_xyz1= kabsch_calculation(ref_all, ref_part, overlap1, overlap2, stru2)

            # Compare mirror one
            stru2_mirrored = mirror_structure_axis(stru2)
            rmsd2, new_xyz2 = kabsch_calculation(ref_all, ref_part, overlap1, overlap2, stru2_mirrored)
            rmsd = np.minimum(rmsd1, rmsd2)
            new_xyz = new_xyz2 if rmsd1 > rmsd2 else new_xyz1

            # remove heavy atoms in common part
            new_ref_all, new_atom1, new_redundant = remove_common_part(ref_all, atom1, related_h_first, overlap1)

            # remove duplicate H atoms of common part
            ref_atom_redundant_H_coord = ref_all[ref_atom_redundant_H_i]

            try:
                new_atom2 = remove_redundant_H(new_xyz, atom2, related_h_second_list[idx], ref_atom_redundant_H_coord, atom_redundant_H[idx])
                
                if new_redundant:
                    new_atom2 = remove_redundant_H(new_xyz, new_atom2, related_h_second_list[idx], ref_atom_redundant_H_coord, new_redundant)

                new_atom2 = index_nonadjacent_common_part(new_atom2, related_h_second_list[idx], need_remove_after_combine_i)

                # assemble two parts
                assemble_xyz = np.vstack((new_xyz, new_ref_all))
                assemble_atoms = new_atom2 + new_atom1

            except:
                assembles.append(None)
                rmsds.append(None)
                continue

            assembles.append({'elements':assemble_atoms, 'coordinates':assemble_xyz})
            rmsds.append(rmsd)

            ## !TEST: print assembled xyz
            # print(len(assemble_xyz))
            # print('\n')
            # for i,j in zip(assemble_atoms, assemble_xyz):
            #     print(f'{i}  {j[0]}  {j[1]} {j[2]}')
            # print(rmsd)
            # os._exit()

            # except:
            #     assembles.append(None)
            #     rmsds.append(None)
        else:
            assembles.append(None)
            rmsds.append(None)
    return assembles, rmsds

def align_xyz_main(overlap_first, overlap_second, aligned_strus, cur_strus, related_h_first_list, related_h_second_list, ref_atom_redundant_H,atom_redundant_H, need_remove_after_combine, threshold=0.2):
    assemble_strus = []
    assemble_rmsd = []
    # Repeat cur_strus
    repeat_times = math.ceil(len(aligned_strus) / len(cur_strus))
    overlap_first = overlap_first * repeat_times
    related_h_first_list = related_h_first_list * repeat_times
    ref_atom_redundant_H = ref_atom_redundant_H* repeat_times
    need_remove_after_combine = need_remove_after_combine * repeat_times

    # for i, (overlap1, strus1) in enumerate(zip(overlap_first, aligned_strus)):
    for i, (overlap1, strus1) in tqdm(enumerate(zip(overlap_first, aligned_strus)), total=len(overlap_first), desc="Processing Overlaps", unit="pair"):
        
        related_h_first = related_h_first_list[i]
        ref_atom_redundant_H_i = ref_atom_redundant_H[i]
        need_remove_after_combine_i = need_remove_after_combine[i]
        if overlap1 and strus1:
            strus, rmsds = align_xyz_single(overlap1, strus1, overlap_second, cur_strus, related_h_first, related_h_second_list, ref_atom_redundant_H_i,atom_redundant_H,need_remove_after_combine_i)
            assemble_strus += strus
            assemble_rmsd += rmsds

    for i, val in enumerate(assemble_rmsd):
        if val is None or val > threshold:
            assemble_strus[i] = None

    return assemble_strus

def assemble_frags(overlap_dict, sub_save_dir, clustering_method_name, threshold=0.2, print_fn=print):
    if clustering_method_name:
        xyz_path_first = Path(sub_save_dir) / f'mol0' / f"centroids_0_{clustering_method_name}.xyz"
    else:
        xyz_path_first = Path(sub_save_dir) / f'mol0' / f"all_structures.xyz"

    aligned_strus = read_multi_molecule_xyz(xyz_path_first)
    for i in range(len(overlap_dict.keys())):

        # if i == 3:
        #     for stru in aligned_strus:
        #         if stru:
        #             for i,j in zip(stru['elements'], stru['coordinates']):
        #                 print(f'{i}  {j[0]}  {j[1]} {j[2]}')
        #             os.exit()
        overlap_first, overlap_second, (related_h_first, related_h_second), (ref_atom_redundant_H, atom_redundant_H), need_remove_after_combine = overlap_dict[i]

        if clustering_method_name:
            xyz_path_cur = Path(sub_save_dir) / f'mol{i+1}' / f"centroids_{i+1}_{clustering_method_name}.xyz"
        else:
            xyz_path_cur = Path(sub_save_dir) / f'mol{i+1}' / f"all_structures.xyz"

        cur_strus = read_multi_molecule_xyz(xyz_path_cur)

        aligned_strus = align_xyz_main(overlap_first, overlap_second, aligned_strus, cur_strus, related_h_first, related_h_second, ref_atom_redundant_H, atom_redundant_H, need_remove_after_combine, threshold=threshold)
        
        # # ! TEST
        # i=0
        # for stru in aligned_strus:    
        #     if stru:
        #         i+=1
        #         if i < 1:
        #             continue
        #         print(len(stru['elements']))
        #         print('\n')
        #         for i,j in zip(stru['elements'], stru['coordinates']):
        #             print(f'{i}  {j[0]}  {j[1]} {j[2]}')
        #         break
        
    aligned_strus = remove_redundant_H_atoms_all(aligned_strus)

    output_path = Path(sub_save_dir) / 'assembled_strus.xyz'
    write_list_to_xyz(aligned_strus, output_path)
    count = len([x for x in aligned_strus if x is not None])
    
    if count > 0:
        print_fn(f'{sub_save_dir.name}: {count} assembled strus were saved to {output_path}.')
        return True
    else:
        print_fn(f'{sub_save_dir.name}: {0} strus were assembled.')
        return False  

def write_list_to_xyz(molecule_list, output_path):
    output_path = Path(output_path)
    with output_path.open('w') as f:
        for idx, molecule in enumerate(molecule_list, start=1):
            if molecule is not None:
                elements = molecule.get('elements', [])
                coordinates = molecule.get('coordinates', [])

                num_atoms = len(elements)
                f.write(f"{num_atoms}\n")
                f.write(f"Assembled Molecule {idx}\n")

                for elem, coord in zip(elements, coordinates):
                    x, y, z = coord
                    f.write(f"{elem} {x:.6f} {y:.6f} {z:.6f}\n")

def remove_redundant_H_atoms_all(strus):
    new_strus=[]
    for stru in strus:
        if stru:
            new_stru = remove_R_atoms(stru)
            new_strus.append(new_stru)
            # for i,j in zip(stru['elements'], stru['coordinates']):
            #     print(f'{i}  {j[0]}  {j[1]} {j[2]}')
            # os.exit()
    return new_strus

def remove_R_atoms(data):
    """
    删除标记为 'R' 的原子及其对应的坐标。

    参数:
        data (dict): 包含 'elements' 和 'coordinates' 的字典。

    返回:
        dict: 更新后的数据字典，不包含 'R' 元素及其坐标。
    """
    elements = data.get('elements', [])
    coordinates = data.get('coordinates', np.array([]))

    if not elements or coordinates.size == 0:
        print("数据中缺少 'elements' 或 'coordinates'。")
        return data

    # 确保 coordinates 是一个 NumPy 数组
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)

    # 找到所有非 'R' 元素的索引
    non_R_indices = [i for i, elem in enumerate(elements) if elem != 'R']

    # 使用这些索引过滤元素和坐标
    filtered_elements = [elements[i] for i in non_R_indices]
    filtered_coordinates = coordinates[non_R_indices]

    # 返回更新后的数据
    return {
        'elements': filtered_elements,
        'coordinates': filtered_coordinates
    }   

if __name__ == '__main__':
    overlap1 = [2, 7, 3, 5, 4, 1, 6]
    overlap2 = [[6, 8, 1, 3, 2, 0, 4], [2, 7, 3, 5, 4, 1, 6]]
    stru1, atom1 = read_xyz('0.xyz')
    stru2, atom2 = read_xyz('1.xyz') 
    align_xyz_single(overlap1, stru1, overlap2, [stru2, stru1], atom1, [atom2,atom1])