import copy 
import os
from io import BytesIO
import csv

from itertools import combinations
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import signal
from concurrent.futures import TimeoutError
from functools import partial

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import BRICS, Recap

# from utils.MacFrag import MacFrag
    

def Recap_frag(mol):
    recap_tree = Recap.RecapDecompose(mol)
    recap_fragments = list(recap_tree.children.keys())
    return recap_fragments

def Brics_frag(mol):
    brics_fragments = BRICS.BRICSDecompose(mol)
    return brics_fragments

def MacFrac(mol):
    mac_frag = MacFrag(mol,maxBlocks=6,maxSR=8,asMols=False,minFragAtoms=1)
    return mac_frag 

def nx_to_smiles(subgraph, original_mol):
    """
    Converts a NetworkX subgraph to a SMILES string using RDKit.
    :param subgraph: A NetworkX graph representing a molecular substructure.
    :param original_mol: The original RDKit molecule, for mapping atom and bond properties.
    :return: SMILES string of the substructure.
    """
    # Create a new editable RDKit molecule
    mol = Chem.RWMol()

    # Map NetworkX nodes to RDKit atom indices
    node_mapping = {}
    for node in subgraph.nodes(data=True):
        atom_idx = node[0]
        atom = original_mol.GetAtomWithIdx(atom_idx)  # Get the corresponding atom
        rd_atom = Chem.Atom(atom.GetSymbol())  # Use the atom's symbol (e.g., 'C', 'O')
        rd_atom.SetFormalCharge(atom.GetFormalCharge())  # Copy the atom's charge
        mol_idx = mol.AddAtom(rd_atom)
        node_mapping[atom_idx] = mol_idx

    # Add bonds to the molecule
    for edge in subgraph.edges(data=True):
        start, end = edge[0], edge[1]
        bond = original_mol.GetBondBetweenAtoms(start, end)  # Get the original bond
        bond_type = bond.GetBondType()  # Use the original bond type (SINGLE, DOUBLE, etc.)
        mol.AddBond(node_mapping[start], node_mapping[end], bond_type)

    # Generate SMILES from the RDKit molecule
    mol = mol.GetMol()  # Convert to a non-editable molecule
    smiles = Chem.MolToSmiles(mol)
    return smiles


# def generate_smiles_fragments(mol):
#     # Step 1: Find all severable bonds
#     severable_bonds = get_severable_bond(mol)
#     all_fragments = set()  # Use a set to avoid duplicate SMILES
#     frags=set()
#     # Step 2: Generate all combinations of bonds to break
#     for num_bonds in range(1, len(severable_bonds) + 1):
#         for bond_subset in combinations(severable_bonds, num_bonds):
#             # Create a temporary copy of the molecule
#             temp_mol = Chem.RWMol(mol)

#             # Step 3: Break the bonds in the current subset
#             for bond in bond_subset:
#                 temp_mol.RemoveBond(bond[0], bond[1])

#             # Step 4: Get the resulting fragments
#             fragments = Chem.GetMolFrags(temp_mol, asMols=True)

#             # Step 5: Convert fragments to SMILES and add to the set if heavy atom count matches
#             for frag in fragments:
#                 if 9 <= frag.GetNumHeavyAtoms() <= 10:  # Filter by heavy atom count
#                     smiles = Chem.MolToSmiles(frag)
#                     all_fragments.add(smiles)

#     return list(all_fragments)

def get_severable_bond(mol):
    severable_list = []
    G = nx.Graph()

    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)


    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        G.add_edge(start, end, bond_type=bond_type)

    temp_G = G.copy()


    for e in G.edges():
        if e[0] not in temp_G.nodes() or e[1] not in temp_G.nodes():
            continue

        G2 = copy.deepcopy(temp_G)

        if G2[e[0]][e[1]]['bond_type'] == 1:
            G2.remove_edge(*e)


        if nx.is_connected(G2):
            continue


        components = list(sorted(nx.connected_components(G2), key=len))

        # l1 = components[0]
        # if len(l1) <= 2:
        #     continue

        severable_list.append((e[0], e[1]))
    return severable_list

# ! max_atoms is a implicit hyperparameters 
def process_bond_subset_with_timeout(bond_subset_mol, timeout=10, min_atoms=6, max_atoms=10):
    """
    Filter frags >=min_atoms and <= max_atoms
    """
    bond_subset, mol = bond_subset_mol
    temp_mol = Chem.RWMol(mol)


    for bond in bond_subset:
        temp_mol.RemoveBond(bond[0], bond[1])

    temp_romol = temp_mol.GetMol()


    atom_mapping = []
    mol_fragments = Chem.GetMolFrags(temp_romol, asMols=True, fragsMolAtomMapping=atom_mapping)

    fragments = []
    for frag_mol, indices in zip(mol_fragments, atom_mapping):
        if min_atoms <= frag_mol.GetNumHeavyAtoms() <= max_atoms:
            smiles = Chem.MolToSmiles(frag_mol)
            fragments.append((smiles, tuple(indices)))
    return fragments

def generate_smiles_fragments_with_indices(mol, max_fragment_num=5):

    severable_bonds = get_severable_bond(mol)
    all_fragments = set()

    non_h_atom_indices = set(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H')

    bond_combinations = []
    max_combinations = 1000 
    
    for num_bonds in range(1, min(4, len(severable_bonds) + 1)):
        combinations_at_level = list(combinations(severable_bonds, num_bonds))
        if len(combinations_at_level) > max_combinations // num_bonds:
            combinations_at_level = combinations_at_level[:max_combinations // num_bonds]
        bond_combinations.extend((comb, mol) for comb in combinations_at_level)

    with multiprocessing.Pool(processes=min(16, len(bond_combinations))) as pool:
        try:

            results = []
            for bond_comb in bond_combinations:
                result = pool.apply_async(process_bond_subset_with_timeout, (bond_comb,))
                results.append(result)

            for result in results:
                try:
                    fragment_list = result.get(timeout=10)
                    for fragment in fragment_list:
                        all_fragments.add(fragment)
                except multiprocessing.TimeoutError:
                    continue 
                except Exception as e:
                    continue
        finally:
            pool.close()
            pool.join()

    all_fragments = list(all_fragments)

    unique_fragments = []
    for i, (smiles_i, indices_i) in enumerate(all_fragments):
        is_subset = False
        set_i = set(indices_i)
        for j, (smiles_j, indices_j) in enumerate(all_fragments):
            if i != j and set_i.issubset(indices_j):
                is_subset = True
                break
        if not is_subset:
            unique_fragments.append((smiles_i, list(set_i)))


    valid_combination = find_valid_fragment_combination(mol, unique_fragments, max_fragment_num=max_fragment_num)

    if valid_combination:
        sorted_combination = sort_tuples_by_overlap(valid_combination)
        return sorted_combination, severable_bonds
    else:
        sorted_fragments = sort_tuples_by_overlap(unique_fragments)
        return sorted_fragments, severable_bonds

def find_valid_fragment_combination(mol, fragments, max_fragment_num=5):
    """
    Find combinations of fragments that satisfy the following conditions:
    1. Adjacent fragments must overlap by at least 4 non-hydrogen atoms (increased overlap requirement).
    2. The selected fragments must collectively cover all atoms of the molecule.
    3. The total number of fragments in the combination must be between 2 and 4, inclusive (added constraint on fragment count).

    Parameters:
    - mol: The original molecule.
    - fragments: A list of candidate fragments, where each element is a tuple (smiles, atom_indices).

    Returns:
    - A fragment combination meeting all conditions, or None if no such combination exists.
    """

    if not fragments:
        return None

    non_h_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']
    non_h_atom_set = set(non_h_atom_indices)

    fragments.sort(key=lambda x: len(set(x[1]).intersection(non_h_atom_set)), reverse=True)

    for start_idx in range(min(10, len(fragments))):
        selected_fragments = [fragments[start_idx]]
        remaining_fragments = fragments.copy()
        remaining_fragments.pop(start_idx)

        fragment_atoms = set(fragments[start_idx][1])
        covered_atoms = fragment_atoms.intersection(non_h_atom_set)
        
        while remaining_fragments and len(selected_fragments) < 5:
            best_fragment = None
            best_overlap = 3
            best_idx = -1

            for idx, (smiles, indices) in enumerate(remaining_fragments):
                indices_set = set(indices)

                non_h_indices = indices_set.intersection(non_h_atom_set)
                overlap = len(covered_atoms.intersection(non_h_indices))

                if overlap >= 4 and overlap > best_overlap:
                    new_atoms = len(non_h_indices - covered_atoms)
                    if new_atoms >= 3:
                        best_fragment = (smiles, indices)
                        best_overlap = overlap
                        best_idx = idx

            if best_fragment is None:
                break

            selected_fragments.append(best_fragment)
            fragment_atoms = set(best_fragment[1])
            covered_atoms.update(fragment_atoms.intersection(non_h_atom_set))
            remaining_fragments.pop(best_idx)

            if check_coverage(mol, selected_fragments) and 2 <= len(selected_fragments) <= 4:
                return selected_fragments

    for start_idx in range(min(max_fragment_num, len(fragments))):
        selected_fragments = [fragments[start_idx]]
        remaining_fragments = fragments.copy()
        remaining_fragments.pop(start_idx)
        
        fragment_atoms = set(fragments[start_idx][1])
        covered_atoms = fragment_atoms.intersection(non_h_atom_set)
        
        while remaining_fragments and len(selected_fragments) < 15:
            best_fragment = None
            best_coverage_increase = 0
            best_idx = -1

            for idx, (smiles, indices) in enumerate(remaining_fragments):
                indices_set = set(indices)
                non_h_indices = indices_set.intersection(non_h_atom_set)

                overlap = len(covered_atoms.intersection(non_h_indices))

                if overlap >= 3:
                    new_coverage = len(non_h_indices - covered_atoms)
                    if new_coverage > best_coverage_increase:
                        best_fragment = (smiles, indices)
                        best_coverage_increase = new_coverage
                        best_idx = idx
            
            if best_fragment is None or best_coverage_increase == 0:
                break
                
            selected_fragments.append(best_fragment)
            fragment_atoms = set(best_fragment[1])
            covered_atoms.update(fragment_atoms.intersection(non_h_atom_set))
            remaining_fragments.pop(best_idx)
            
            if check_coverage(mol, selected_fragments) and len(selected_fragments) <= 5:
                return selected_fragments

    return None

def draw_molecule_with_atom_indices(mol,fig_path=None):
    """
    Draws a molecule with atom indices labeled.

    Parameters:
        mol (rdkit.Chem.Mol): The molecule to draw.
    """
    # Generate a 2D depiction of the molecule
    Chem.rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500) 
    drawer.drawOptions().addAtomIndices = True  # Enable atom indices
    
    # Draw the molecule
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert to image format and display
    img = drawer.GetDrawingText()
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(plt.imread(BytesIO(img)))
    plt.axis('off')  # Turn off axes
    if fig_path:
        plt.savefig(fig_path)
    plt.close(fig)  # Explicitly close the figure
    
import re
def clean_smiles(smiles):

    pattern = re.compile(r'\[([^\]:]+):\d+\]')
    cleaned_smiles = pattern.sub(r'[\1]', smiles)
    return cleaned_smiles

def plot_fragments(smiles_list, fig_path=None):
    # Convert SMILES to RDKit Mol objects
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Annotate molecules with atom indices
    for mol in mols:
        if mol is not None:
            Chem.rdDepictor.Compute2DCoords(mol)

    # Draw molecules with high resolution
    images = []
    for mol, smiles in zip(mols, smiles_list):
        if mol is not None:
            drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
            drawer.drawOptions().addAtomIndices = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            img_data = BytesIO(drawer.GetDrawingText())
            images.append(plt.imread(img_data, format='png'))

    # Display images in a grid
    rows = (len(images) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))
    for ax, img, smiles in zip(axes.flatten(), images, smiles_list):
        ax.imshow(img)
        if smiles == smiles_list[-1]:
            ax.set_title(f'Original Mol: {smiles}')
        else:
            mol = Chem.MolFromSmiles(smiles)
            simplified_smiles = Chem.MolToSmiles(mol, canonical=True)
            clean_smiles = re.sub(r':\d+', '', simplified_smiles)
            ax.set_title(clean_smiles)
        ax.axis('off')
    for ax in axes.flatten()[len(images):]:
        ax.axis('off')
    plt.tight_layout()
    
    if fig_path:
        fig.savefig(fig_path)
    else:
        plt.show()
    plt.close(fig)  # Explicitly close the figure


def check_coverage(mol, fragments_with_indices):
    # Step 1: Get the original molecule's atom indices
    original_atoms = set(range(mol.GetNumAtoms()))  # Atom indices in the original molecule

    # Step 2: Collect the atom indices from the fragments
    fragment_atoms = set()
    for _, indices in fragments_with_indices:
        fragment_atoms.update(indices)  # Add atoms from the current fragment

    # Step 3: Compare the sets
    if fragment_atoms == original_atoms:
        # print("The fragments cover all atoms in the original molecule.")
        return True
    else:
        # print("The fragments do NOT cover all atoms in the original molecule.")
        return False

def save_fragments_to_csv(frags_smi, frag_i,  ref_pos_list_in_first, redundant_pos_list_in_second, file_path):
    # Open the CSV file in write mode
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["SMILES", "Atom Indices", 'ref_pos_list_in_first','redundant_pos_list_in_second'])
        # Write the data (each row is a tuple of SMILES and atom indices)
        # for smiles, indices in zip(frags_smi, frag_i):
        for i in range(len(frags_smi)):
            smiles = frags_smi[i]
            indices = frag_i[i]
            ref_pos = ref_pos_list_in_first[i]
            redundant_pos = redundant_pos_list_in_second[i]
            # Convert atom indices list to a string, e.g., '1,2,3'
            indices_str = ",".join(map(str, indices))
            writer.writerow([smiles, indices_str, ref_pos, redundant_pos])

def save_smi_of_fragment_to_dat(frags_smi, file_path):
    with open(file_path, mode='w', newline='') as f1:
        for smiles in frags_smi:
            f1.write(smiles)
            f1.write('\n')    

def calculate_overlap(list1, list2):
    overlap = list(set(list1) & set(list2))
    return len(overlap)


def sort_tuples_by_overlap(data, start_with=0):

    if not data:
        return []

    data = data.copy()

    current = None
    sorted_data = []

    for i, item in enumerate(data):
        if item[1][0] == start_with:
            current = data.pop(i)
            sorted_data.append(current)
            break

    if current is None and data:
        current = data.pop(0)
        sorted_data.append(current)

    if current is None:
        return []

    while data:
        max_overlap = 0
        next_tuple = None
        next_index = -1
        
        for idx, candidate in enumerate(data):
            overlap = calculate_overlap(current[1], candidate[1])
            if overlap > max_overlap:
                max_overlap = overlap
                next_tuple = candidate
                next_index = idx

        if next_tuple and max_overlap > 0:
            sorted_data.append(next_tuple)
            current = next_tuple
            data.pop(next_index)
        else:
            sorted_data.extend(data)
            break
    
    return sorted_data

def split_main_with_index(smiles, save_path, max_fragment_num=5, print_fn=print):
    """
    Split a molecule into fragments while preserving the correspondence between
    fragment atom indices and original molecule atom indices.

    Parameters:
    - smiles (str): The SMILES string of the original molecule.
    - save_path (str or Path): Directory path to save the output files.

    Returns:
    - mol_smiles_mapped
    - frags_smi
    - frag_i
    - redundant_H_pos
    """
    # Ensure save_path is a Path object
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Paths for saving figures
    frag_fig_path = save_path / 'figures_of_frag.png'
    mol_fig_path = save_path / 'figures_of_mol.png'
    
    # Step 1: Load the molecule from SMILES and ensure canonical form
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Convert to canonical SMILES form without isotope/stereochemistry information
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    mol_smiles_mapped = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)

    fragments, severable_bonds = generate_smiles_fragments_with_indices(mol, max_fragment_num=max_fragment_num)

    check_frag = check_coverage(mol, fragments)
    if not check_frag:
        print_fn(f'Failure! Molecule {smiles} fails to be split into fragments.')
        return False, False, False, False

    print_fn(f'Success! Molecule {smiles} was split into {len(fragments)} fragments.')
    frags_smi = [frag[0] for frag in fragments]
    frag_i = [frag[1] for frag in fragments]

    ref_pos_list_in_first, redundant_pos_list_in_second = get_redundant_H(severable_bonds, frag_i)

    csv_file = save_path / 'fragments.csv'
    save_fragments_to_csv(frags_smi, frag_i, ref_pos_list_in_first, redundant_pos_list_in_second, csv_file)

    dat_file = save_path / 'frag_smiles.dat'
    save_smi_of_fragment_to_dat(frags_smi, dat_file)

    plot_fragments(frags_smi+[smiles], fig_path=frag_fig_path)
    # print_fn(f'Saving fragment figures to {frag_fig_path}')

    draw_molecule_with_atom_indices(mol, fig_path=mol_fig_path)
    # print_fn(f'Saving molecule figure to {mol_fig_path}')
    
    redundant_H_pos = (ref_pos_list_in_first, redundant_pos_list_in_second)
    return mol_smiles_mapped, frags_smi, frag_i, redundant_H_pos


def get_redundant_H(severable_bonds, frags):
    redundant_pos_list_in_second=[]
    ref_pos_list_in_first = []
    for i in range(len(frags)-1):
        cur_part = frags[i]
        next_part = frags[i+1]
        redundant_pos = []
        ref_pos = []

        # Get the index of the broken bond in xyz.
        for b1, b2 in severable_bonds:

            if b1 in cur_part and b2 in cur_part:
                if (b1 in next_part and b2 not in next_part):
                    redundant_pos.append(b1)
                    ref_pos.append(b2)
                elif (b2 in next_part and b1 not in next_part):
                    redundant_pos.append(b2)
                    ref_pos.append(b1)

        # assert len(redundant_pos)==len(ref_pos)==1, 'The current version supports only one broken single bond in a common part, so only one H will be redundant.'
        # redundant_pos= redundant_pos[0]
        # ref_pos=ref_pos[0]

        redundant_pos_list_in_second.append(redundant_pos)
        ref_pos_list_in_first.append(ref_pos)

    redundant_pos_list_in_second.insert(0, [])
    ref_pos_list_in_first.append([])

    return ref_pos_list_in_first, redundant_pos_list_in_second

if __name__ == '__main__':
    smiles='O=C(O)CN1CCC[C@@H]1CCCc1ccccc1'
    # smiles='Cn1ccc(C(=O)NC[C@H]2CCCO2)n1'
    # smiles='CC(C)CC(=O)C(O)Cc1ccc(O)cc1'
    # smiles='COc1ccc(C(=O)Nc2cccc(C(=O)O)c2)cc1'
    smiles='CCCCCCCCCCCCCCC[C@@H](O)[C@H](CO)NC(C)=O'
    # smiles='C/C(=N\O)C(C)NCC(C)(C)CNC(C)/C(C)=N/O'
    # smiles='OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F'
    frags = split_main_with_index(smiles, './')
    # frags_smi = frags[0]
    # frags_smi+=[smiles]

    # plot_fragments(frags_smi)
    # print(frag_smi)
    # draw_molecule_with_atom_indices(mol)
    # a = Brics_frag(mol)
    # print(a)
    
    
    
