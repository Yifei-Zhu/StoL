import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from utils.xyz_to_generated_smiles_mapping import get_mapping_between_xyz_and_smiles_main

def get_frags_overlaps(sorted_fragments, redundant_H_idx):
    common_elements = []
    common_positions = {}  # Dictionary to store common elements and their positions

    ref_pos_list = redundant_H_idx[0]
    redundant_pos_list = redundant_H_idx[1]

    all_common=[]
    for i in range(len(sorted_fragments) - 1):
        set_next = set(sorted_fragments[i + 1])
        common = [x for x in sorted_fragments[i] if x in set_next]
        
        #!TEST
        # ref_pos_list_in_first = ref_pos_list[i]
        # redundant_pos_list_in_second = redundant_pos_list[i+1]

        ref_pos_list_in_first = [ref_pos_list[i]] if isinstance(ref_pos_list[i], int) else ref_pos_list[i]
        redundant_pos_list_in_second = [redundant_pos_list[i+1]] if isinstance(redundant_pos_list[i+1], int) else redundant_pos_list[i+1]
        
        all_common+=common
        
        # Store common elements and their positions in the fragments
        common_positions[i] = {
            "common_elements": common,
            "positions_in_cur": [sorted_fragments[i].index(x) for x in common],
            "positions_in_next": [sorted_fragments[i + 1].index(x) for x in common],
            #!TEST
            # "ref_pos_list_in_first": sorted_fragments[i].index(ref_pos_list_in_first),
            # "redundant_pos_list_in_second": sorted_fragments[i+1].index(redundant_pos_list_in_second),

            # NOTE: current version support two adjacent fragment have one or more broken site
            "ref_pos_list_in_first": [sorted_fragments[i].index(x) for x in ref_pos_list_in_first],
            "redundant_pos_list_in_second": [sorted_fragments[i+1].index(x) for x in redundant_pos_list_in_second],
        }

        common_elements.append(common)

    # 0524 FEAT: in some cases, non-adjacemt frags may also overlap, so we should detect these overlap
    common_part_not_adjacent = {i: [] for i in range(len(sorted_fragments)-1)}
    sets = [set(lst) for lst in sorted_fragments]
    for i, lst in enumerate(sorted_fragments):
        if i==0:
            continue
        for num in lst:
            found_in_other = False
            for j, s in enumerate(sets):
                if j != i and num in s:
                    found_in_other = True
            if found_in_other:
                if num not in all_common:
                    common_part_not_adjacent[i-1].append(num)    
    for key in common_positions.keys():
        common_positions[key]['common_part_not_adjacent_need_remove'] = common_part_not_adjacent[key]

    return common_elements, common_positions


def check_equivalence_of_two_smiles(mol_1, mol_2):
    """
    检查两个 RDKit 分子对象对应的 SMILES 字符串是否等价。
    等价性的判断基于去除特定字符后的排序结果是否相同。

    参数:
        mol_1 (rdkit.Chem.Mol): 第一个分子对象。
        mol_2 (rdkit.Chem.Mol): 第二个分子对象。

    返回:
        bool: 如果两个 SMILES 等价，返回 True；否则，返回 False。
    """
    # 生成规范化的 SMILES 字符串
    def preprocess_mol(mol):
        # 移除所有氢原子
        mol = Chem.RemoveHs(mol, implicitOnly=False)
        # 移除所有原子映射编号
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    # 预处理分子
    mol_1 = preprocess_mol(mol_1)
    mol_2 = preprocess_mol(mol_2)

    canonical_smiles_1 = Chem.MolToSmiles(mol_1, canonical=True, isomericSmiles=False, kekuleSmiles=False)
    canonical_smiles_2 = Chem.MolToSmiles(mol_2, canonical=True, isomericSmiles=False, kekuleSmiles=False)

    # 清理 SMILES 字符串
    # 去除 :数字 和 ()[] 等字符
    pattern = r':\d+|[()\[\]]'
    clean_smiles_1 = re.sub(pattern, '', canonical_smiles_1)
    clean_smiles_2 = re.sub(pattern, '', canonical_smiles_2)
    clean_smiles_1 = clean_smiles_1.replace('H', '').upper()
    clean_smiles_2 = clean_smiles_2.replace('H', '').upper()
    # 对清理后的字符串进行排序
    sorted_smiles_1 = ''.join(sorted(clean_smiles_1))
    sorted_smiles_2 = ''.join(sorted(clean_smiles_2))

    # 比较排序后的字符串是否相同
    are_same = sorted_smiles_1 == sorted_smiles_2
    return are_same



def parse_smiles_with_mapping(smiles):
    """
    解析带有原子映射号的 SMILES，并返回 RDKit 分子对象。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    return mol

def build_atom_map(mol):
    """
    构建一个从原子映射号到原子索引的字典。
    """
    atom_map = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num != 0:
            atom_map[map_num] = atom.GetIdx()
    return atom_map

def find_atom_correspondence(mol1, mol2, atom_map1, atom_map2):
    """
    根据原子映射号建立两个分子之间的原子对应关系。
    """
    correspondence = {}
    for map_num in atom_map1:
        if map_num in atom_map2:
            correspondence[atom_map1[map_num]] = atom_map2[map_num]
        else:
            pass
    return correspondence

def map_atoms_via_mcs(mol1, mol2):
    """
    Establish an atomic correspondence between two molecules using the Maximum Common Substructure (MCS) method.

    Parameters:
        mol1 (rdkit.Chem.Mol): The first molecule.
        mol2 (rdkit.Chem.Mol): The second molecule.

    Returns:
        dict: A dictionary mapping atom indices from mol1 to mol2.

    Raises:
        ValueError: If MCS search is canceled, no matching substructure is found, or the number of matched atoms 
                    does not equal the total number of atoms in either molecule.
    """
    # Perform MCS search with specific comparison criteria
    mcs_result = rdFMCS.FindMCS(
        [mol1, mol2],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        matchValences=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=20  # Set timeout to prevent long computations
    )
    from rdkit.Chem import Draw
    img = Draw.MolsToGridImage([mol1, mol2], 
                          molsPerRow=2, 
                          subImgSize=(300, 300),
                          legends=['Mol1: Ethanol', 'Mol2: Acetic Acid'])

    img.save('molecules.png')

    # Check if MCS search was canceled or no SMARTS string was found
    if mcs_result.canceled or not mcs_result.smartsString:
        raise ValueError("MCS search was canceled or no matching substructure was found.")
    
    # Obtain the SMARTS string of the MCS
    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    # Check if the SMARTS string was successfully parsed
    if mcs_mol is None:
        raise ValueError("Failed to parse the SMARTS string of the MCS.")
    
    # Get substructure matches in both molecules based on the MCS
    matches1 = mol1.GetSubstructMatches(mcs_mol)
    matches2 = mol2.GetSubstructMatches(mcs_mol)
    
    # Check if matches were found in both molecules
    if not matches1 or not matches2:
        raise ValueError("No substructure match found in one or both molecules.")
    
    # Assume the first match corresponds
    match1 = matches1[0]
    match2 = matches2[0]
    # Establish atomic correspondence by zipping the matched atom indices
    atom_corr = dict(zip(match1, match2))
    # Additional Check: Ensure the number of matched atoms equals the total number of atoms in both molecules
    if len(atom_corr) != mol1.GetNumAtoms():
        raise ValueError(f"The number of matched atoms ({len(atom_corr)}) does not equal the total number of atoms in mol1 ({mol1.GetNumAtoms()}).")
    
    if len(atom_corr) != mol2.GetNumAtoms():
        raise ValueError(f"The number of matched atoms ({len(atom_corr)}) does not equal the total number of atoms in mol2 ({mol2.GetNumAtoms()}).")
    
    return atom_corr

def get_atom_correspondence(smiles_a, smiles_b):
    """
    获取两个 SMILES 中所有原子的对应关系。
    """
    # 解析 SMILES

    mol_a = parse_smiles_with_mapping(smiles_a)
    mol_b = parse_smiles_with_mapping(smiles_b)

    if check_equivalence_of_two_smiles(mol_a, mol_b):
        try:
            # 构建映射字典
            mapping_a = build_atom_map(mol_a)
            mapping_b = build_atom_map(mol_b)
            
            # 建立原子对应关系
            correspondence = find_atom_correspondence(mol_a, mol_b, mapping_a, mapping_b)
            
            # 使用 MCS 方法进一步完善对应关系
            mcs_correspondence = map_atoms_via_mcs(mol_a, mol_b)
            # 更新 correspondence，优先使用 MCS 对应关系
            correspondence.update(mcs_correspondence)
            # for idx_mol_1, idx_mol_2 in correspondence.items():
            #     print(f"mol_1 atom {mol_a.GetAtomWithIdx(idx_mol_1).GetSymbol()} ({idx_mol_1}) -> mol_2 atom {mol_b.GetAtomWithIdx(idx_mol_2).GetSymbol()} ({idx_mol_2})")
            return correspondence
        except:
            return None

def get_mapping_of_smi1_to_smi2list(smi1, smi2_list):

    all_mappings=[]
    mol_1 = Chem.MolFromSmiles(smi1)
    for smi2 in smi2_list:
        if not smi2:
            all_mappings.append(False)
            continue 
        mol_2 = Chem.MolFromSmiles(smi2)
        # if check_equivalence_of_two_smiles(mol_1, mol_2):
        if mol_1 and mol_2:
            if check_equivalence_of_two_smiles(mol_1, mol_2):
                # list of tuple (index_mol_1, index_mol_2)
                mapping = get_atom_correspondence(smi1,smi2)
                all_mappings.append(mapping)
                
            else:
                all_mappings.append(False)
        else:
            all_mappings.append(False)
    return all_mappings


def get_all_overlap_indices(mappings, g_mapping, overlaps, broken_idx):
    """
    Retrieve the XYZ indices of overlapping parts for all molecules.

    Parameters:
        mappings (list of dict): List of mappings from SMILES to GSMI.
        g_mapping (list of dict): List of mappings from GSMI to XYZ.
        overlaps (iterable): List of identifiers for overlapping parts.

    Returns:
        list of list: A list where each element corresponds to a molecule and contains 
                      the XYZ indices of its overlapping parts.
    """
    correspondence = []
    broken_bond_idx_list = []
    for i in range(len(mappings)):
        if mappings[i] and g_mapping[i]:
            # try:
            # Map each overlap identifier to its corresponding XYZ index
            overlap_indices = [g_mapping[i][mappings[i][j]] for j in overlaps]
            #!TEST
            # broken_bond_idx = g_mapping[i][mappings[i][broken_idx]]
            broken_bond_idx = [g_mapping[i][mappings[i][b]] for b in broken_idx]
            # except (KeyError, IndexError):
            #     # If a mapping fails, assign an empty list for this molecule
            #     overlap_indices = []
            #     broken_bond_idx = []
        else:
            # If either mapping is missing, assign an empty list for this molecule
            overlap_indices = []
            broken_bond_idx = []
        correspondence.append(overlap_indices)
        broken_bond_idx_list.append(broken_bond_idx)
    return correspondence, broken_bond_idx_list



def get_nonadjacent_ovelap_indices(mappings, g_mapping, overlaps):
    correspondence = []
    for i in range(len(mappings)):
        if mappings[i] and g_mapping[i]:
            overlap_indices = [g_mapping[i][mappings[i][j]] for j in overlaps]

        else:
            overlap_indices = []
        correspondence.append(overlap_indices)
    return correspondence


def get_xyz_overlap_indices(sub_save_dir, sorted_frags, sorted_frag_smi, redundant_H_idx, clustering_method_name, logger):
    '''
    Example:s
    sorted_frags: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9, 10, 11], [9, 10, 11, 12, 13, 14, 15, 16, 17]]
    
    sorted_frag_smi: ['O=[C:1]([OH:2])[CH2:3][N:4]1[CH2:5][CH2:6][CH2:7][CH:8]1[CH3:9]', '[CH3:3][N:4]1[CH2:5][CH2:6][CH2:7][CH:8]1[CH2:9][CH2:10][CH3:11]', '[CH3:9][CH2:10][CH2:11][c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1']
    
    overlap_dict = {0: {'common_elements': [3, 4, 5, 6, 7, 8, 9], 'positions_in_cur': [3, 4, 5, 6, 7, 8, 9], 'positions_in_next': [0, 1, 2, 3, 4, 5, 6], 'ref_pos_list_in_first': [1], 'redundant_pos_list_in_second': [3]}, 1: {'common_elements': [9, 10, 11], 'positions_in_cur': [6, 7, 8], 'positions_in_next': [0, 1, 2], 'ref_pos_list_in_first': [8], 'redundant_pos_list_in_second': [9]}}
    '''
    # Get a list of overlaps between neighbouring fragments
    _, overlap_dict = get_frags_overlaps(sorted_frags, redundant_H_idx)

    # overlap_list -> smi1_to_gsmi2list_mappings -> g_smi_to_xyz_mapping_list 
    # ---> FINAL: a list of which the element is xyz index of overlap
    xyz_overlap_part_index_dict={}

    for i in range(len(sorted_frag_smi[1:])):
        if clustering_method_name:
            xyz_path_former = Path(sub_save_dir) / f'mol{i}' / f"centroids_{i}_{clustering_method_name}.xyz"
            xyz_path_cur = Path(sub_save_dir) / f'mol{i+1}' / f"centroids_{i+1}_{clustering_method_name}.xyz"
        else:
            xyz_path_former = Path(sub_save_dir) / f'mol{i}' / f"all_structures.xyz"
            xyz_path_cur = Path(sub_save_dir) / f'mol{i+1}' / f"all_structures.xyz"

        cur_smi = sorted_frag_smi[i]
        next_smi = sorted_frag_smi[i+1]


        # DICT1: Get the mapping dict between the generated SMILES and xyz
        cur_g_smiles_list, cur_g_smi_to_xyz_mapping_list, cur_related_h_mapping = get_mapping_between_xyz_and_smiles_main(xyz_path_former)
        next_g_smiles_list, next_g_smi_to_xyz_mapping_list, next_related_h_mapping = get_mapping_between_xyz_and_smiles_main(xyz_path_cur)

        # DICT2: Get the mapping dict between origin frag SMILES and generated SMILES
        cur_smi_to_gsmi2list_mappings = get_mapping_of_smi1_to_smi2list(cur_smi, cur_g_smiles_list)
        next_smi_to_gsmi2list_mappings = get_mapping_of_smi1_to_smi2list(next_smi, next_g_smiles_list)
        # Get the overlap list of two fragments
        overlap_between_cur_next = overlap_dict[i]

        overlap_in_cur = overlap_between_cur_next['positions_in_cur']
        overlap_in_next = overlap_between_cur_next['positions_in_next']
        cur_ref_atom_index = overlap_between_cur_next['ref_pos_list_in_first']
        next_redundant_H_index = overlap_between_cur_next['redundant_pos_list_in_second']
        need_remove_after_combine = overlap_between_cur_next['common_part_not_adjacent_need_remove']

        # Retrieve the XYZ indices of overlapping parts for current and next fragments. (Connecting two dictionaries)
        # cur_ref_atom_index_mapped: reference atom for removing the redundant H
        #next_redundant_H_index_mapped: the index of heavy atom  with redundant H

        cur_overlap_xyz_index, cur_ref_atom_index_mapped = get_all_overlap_indices(cur_smi_to_gsmi2list_mappings, cur_g_smi_to_xyz_mapping_list, overlap_in_cur, cur_ref_atom_index)

        next_overlap_xyz_index, next_redundant_H_index_mapped = get_all_overlap_indices(next_smi_to_gsmi2list_mappings, next_g_smi_to_xyz_mapping_list, overlap_in_next, next_redundant_H_index)

        # Non-adjacent overlap
        need_remove_after_combine_mapping = get_nonadjacent_ovelap_indices(next_smi_to_gsmi2list_mappings, next_g_smi_to_xyz_mapping_list,  need_remove_after_combine)

        if len(cur_overlap_xyz_index) != 0 and len(next_overlap_xyz_index) != 0:
            xyz_overlap_part_index_dict[i] = (cur_overlap_xyz_index, next_overlap_xyz_index, (cur_related_h_mapping, next_related_h_mapping), (cur_ref_atom_index_mapped, next_redundant_H_index_mapped), need_remove_after_combine_mapping)

        else:
            logger.info(f'Failed when trying to get overlapping Indies between {i} and {i+1} fragments')
            return {}
 
    logger.info(f'Successfully obtained indies of all fragment pairs of {sub_save_dir.name}.')
    return xyz_overlap_part_index_dict
