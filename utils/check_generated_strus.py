import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

from ase.data import chemical_symbols
import networkx as nx

from utils.check_planar import check_planarity

class MolecularGraphCheck:
    STANDARD_BOND_DISTANCES = {
        'H': {'H': 0.74, 'C': 1.09, 'N': 1.01, 'O': 0.96, 'F': 0.92},
        'C': {'H': 1.09, 'C': 1.54, 'N': 1.47, 'O': 1.43, 'F': 1.35},
        'N': {'H': 1.01, 'C': 1.47, 'N': 1.45, 'O': 1.40, 'F': 1.36},
        'O': {'H': 0.96, 'C': 1.43, 'N': 1.40, 'O': 1.48, 'F': 1.42},
        'F': {'H': 0.92, 'C': 1.35, 'N': 1.36, 'O': 1.42, 'F': 0.71},
    }

    def __init__(self, margin=1.2):
        self.margin = margin

    @staticmethod
    def calculate_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def set_graph(self, symbols, positions):
        assert len(symbols) == len(positions), ValueError("Symbols and positions must have the same length.")

        size = len(symbols)
        G = nx.Graph()

        for i, symbol in enumerate(symbols):
            G.add_node(i, symbol=symbol)

        for i in range(size):
            for j in range(i + 1, size):
                symbol1, symbol2 = symbols[i], symbols[j]
                if symbol1 in self.STANDARD_BOND_DISTANCES and symbol2 in self.STANDARD_BOND_DISTANCES[symbol1]:
                    bond_length = self.STANDARD_BOND_DISTANCES[symbol1][symbol2]
                
                else:
                    raise ValueError(f"Bond length data not found for pair ({symbol1}, {symbol2}).")

                distance = self.calculate_distance(positions[i], positions[j])

                if distance <= bond_length * self.margin:
                    G.add_edge(i, j)
        return G

    @staticmethod
    def graph_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        return G

    @staticmethod
    def compare_graphs(graph1, graph2):
        # MolecularGraphCheck.visualize_graph(graph1, "3D Structure Graph")
        # MolecularGraphCheck.visualize_graph(graph2, "SMILES Graph")
        gm = nx.isomorphism.GraphMatcher(graph1, graph2, node_match=lambda n1, n2: n1['symbol'] == n2['symbol'])
        return gm.is_isomorphic()
 
    @staticmethod
    def visualize_graph(graph, title):
        """
        Visualize a molecular graph using NetworkX and Matplotlib.

        :param graph: The NetworkX graph to visualize.
        :param title: Title for the graph plot.
        """
        # Extract node labels
        labels = nx.get_node_attributes(graph, 'symbol')

        # Draw the graph
        pos = nx.spring_layout(graph, seed=42)  # Position nodes using a spring layout
        plt.figure(figsize=(6, 6))
        nx.draw(graph, pos, with_labels=False, node_color='skyblue', node_size=500, edge_color='gray')
        nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color='black')
        plt.title(title)
        plt.show()
    
def check_from_tyg_data(data, tolerance=0.2):
    """
    Validate a single molecule's 3D structure against its SMILES using graph comparison.

    :param data: A data object containing 'smiles', 'pos_gen', and 'atom_type'.
    :param tolerance: Tolerance multiplier for bond length comparison.
    :return: Boolean indicating whether the structure is valid.
    """
    smi = data.smiles
    coord = data.pos_gen
    at = data.atom_type
    at = [chemical_symbols[i] for i in at]
    # Initialize the graph builder
    validator = MolecularGraphCheck(margin=tolerance)
    # Build the graph from 3D coordinates
    graph_3d = validator.set_graph(at, coord)
    
    validator2 = MolecularGraphCheck(margin=tolerance)
    # Build the graph from SMILES
    graph_smiles = validator.graph_from_smiles(smi)
    
    # MolecularGraphCheck.visualize_graph(graph_3d, "3D Structure Graph")
    # MolecularGraphCheck.visualize_graph(graph_smiles, "SMILES Graph")
    # import os
    # os.exit()

    # Compare the two graphs
    are_graphs_equal = validator.compare_graphs(graph_3d, graph_smiles)
    return are_graphs_equal



def check_from_pkl(pkl_file, tolerance=0.2):
    """
    Validate molecules stored in a pickle file and return a mask of results.
    
    :param pkl_file: Path to the pickle file containing molecule data.
    :param tolerance: Tolerance for bond length comparison.
    :return: A list of booleans representing the validation results for each molecule.
    """
    tolerance=1+tolerance
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
        mask = []
        for i, da in enumerate(data):
            is_valid1 = check_from_tyg_data(da, tolerance=tolerance)

            is_valid2 = check_planarity(da, threshold_flatness=0.3)
            # TODO: remove this line@
            is_valid2=is_valid1
            is_valid = is_valid1 and is_valid2
            mask.append(is_valid)
        return mask, i+1


# Example usage
if __name__ == "__main__":
    smiles = "C(CO)N"  # Example molecule
    atom_types = ["C", "C", "O", "N", "H", "H", "H", "H", "H"]  # Atom types
    coordinates = np.array([
        [0.0, 0.0, 0.0],  # C
        [1.54, 0.0, 0.0],  # C
        [2.54, 0.96, 0.0],  # O
        [1.54, -1.47, 0.0],  # N
        [0.0, 0.0, 1.09],  # H
        [1.54, 0.0, 1.09],  # H
        [3.08, 1.43, 0.0],  # H
        [1.54, -1.47, 1.09],  # H
        [0.0, -1.47, 0.0],  # H
    ])

    validator = MoleculeValidator()
    result, bond_matrix = validator.validate_structure(smiles, atom_types, coordinates)