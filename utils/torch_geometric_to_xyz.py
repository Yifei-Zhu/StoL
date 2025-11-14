import pickle

from pathlib import Path

def torch_geometric_to_xyz(data, option='output', comment='', output_file=''):
    """
    Convert torch geometric data to an XYZ format file.

    Args:
        data: Torch geometric data containing atom types and positions.
        option: 'input' or 'output', determines whether to use `data.pos` or `data.pos_gen`.
        comment: A comment to include in the XYZ file.
        output_file: Path to the output file.
    """
    atom_types = data.atom_type.cpu().numpy() if hasattr(data.atom_type, 'cpu') else data.atom_type
    if option == 'input':
        positions = data.pos.cpu().numpy()
        output_file = output_file.parent / f"input{output_file.name}"
    elif option == 'output':
        positions = data.pos_gen.cpu().numpy()

    atom_type_map = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}  # Example mapping for atomic numbers to symbols
    atom_labels = [atom_type_map.get(int(t), "X") for t in atom_types]  # Default to "X" for unknown types

    # Check if append is needed for single file saving
    append = output_file.suffix == '.xyz' and "all_structures" in str(output_file)
    save_one(output_file, comment, atom_types, atom_labels, positions, append=append)
    return atom_labels, positions
    
    
def save_one(output_file, comment, atom_types, atom_labels, positions, append=False):
    """
    Save atomic data to an XYZ format file.
    
    Args:
        output_file: Path to the output XYZ file.
        comment: A comment to include in the XYZ file.
        atom_types: List of atomic types.
        atom_labels: List of atomic labels (e.g., "H", "C").
        positions: Array of atomic positions.
        append: If True, append to the file; otherwise, overwrite it.
    """
    mode = "a" if append else "w"  # Open in append mode if specified
    with open(output_file, mode) as f:
        f.write(f"{len(atom_types)}\n")  # Number of atoms
        f.write(f"{comment}\n")  # Comment line
        for label, pos in zip(atom_labels, positions):
            f.write(f"{label} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")  # Atom label and coordinates

def pkl_to_xyz(pkl_file, nu, mask, all_transform=True):
    """
    Convert a pickle file containing molecular data to XYZ format files, skipping invalid structures.

    Args:
        pkl_file: Path to the input pickle file.
        nu: Molecule number.
        mask: A list of booleans indicating whether to include each structure (True to include, False to skip).
        all_transform: If True, save each structure to a separate file;
                       if False, save all structures in a single file.
    """
    output_file_all = pkl_file.parent / 'all_structures.xyz'  # Output file for combined structures
    
    failed_output_file_all = pkl_file.parent / 'failed_all_structures.xyz'

    strus = {}
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
        if all_transform:
            for i, (da, is_valid) in enumerate(zip(data, mask)):
                if not is_valid:
                    comment = f'The {i}th failed structure of molecule {nu}.'
                    _,_ = torch_geometric_to_xyz(
                    da,
                    option='output',
                    comment=comment,
                    output_file=failed_output_file_all
                    )
                    continue
                # Generate a separate file for each valid structure
                output_file = pkl_file.parent / f'generated_{i}.xyz'
                comment = f'The {i}th generated structure of molecule {nu}.'
                atom_labels, positions = torch_geometric_to_xyz(da, option='output', comment=comment, output_file=output_file)
                strus[i] = {'atom_type':atom_labels, 'positions':positions, 'smiles': da.smiles}
        else:
            # Clear the existing combined file, if it exists
            if output_file_all.exists():  # Check if the file exists
                output_file_all.unlink()
                
            if failed_output_file_all.exists():
                failed_output_file_all.unlink()
                
            for i, (da, is_valid) in enumerate(zip(data, mask)):
                if not is_valid:
                    comment = f'The {i}th failed structure of molecule {nu}.'
                    _,_ = torch_geometric_to_xyz(
                    da,
                    option='output',
                    comment=comment,
                    output_file=failed_output_file_all
                    )
                else:
                    # Append all valid structures to a single file
                    comment = f'The {i}th generated structure of molecule {nu}.'
                    atom_labels, positions = torch_geometric_to_xyz(
                        da,
                        option='output',
                        comment=comment,
                        output_file=output_file_all
                    )
                    strus[i]={'atom_type':atom_labels, 'positions':positions, 'smiles': da.smiles }
    return strus

if __name__ == '__main__': 
    # File path for the pickle file
    file_path = "samples_all.pkl"

    # Open the pickle file and load the data
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print(data[0].pos)
        print(data[0].pos_gen)
        print(data[0].smiles)
        data = data[0]

    #torch_geometric_to_xyz_input(data, output_file="input.xyz")
    torch_geometric_to_xyz(data, output_file="output.xyz")


'''
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

print(data)
G = to_networkx(data, to_undirected=True)

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=10)
plt.title("Graph Visualization")
plt.show()
'''
