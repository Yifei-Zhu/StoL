import sys

def extract_nth_xyz(file_path, n, output_file=None):
    """
    Extract the nth XYZ structure from a multi-structure XYZ file.

    Parameters:
    - file_path: Path to the multi-structure XYZ file.
    - n: Index of the structure to extract (1-based index).
    - output_file: Optional path to save the extracted XYZ structure.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
        
        structures = []
        i = 0
        while i < len(lines):
            # Read the number of atoms
            try:
                num_atoms = int(lines[i].strip())
            except ValueError:
                raise ValueError(f"Invalid number of atoms at line {i+1}.")
            
            # Extract the corresponding structure
            structure = lines[i:i + num_atoms + 2]  # Includes atom lines + comment line
            structures.append(structure)
            i += num_atoms + 2  # Move to the next structure
        
        if n < 1 or n > len(structures):
            raise IndexError(f"Structure number {n} is out of range. File contains {len(structures)} structures.")
        
        # Get the nth structure (1-based index)
        nth_structure = structures[n - 1]
        
        # Print the structure
        print("".join(nth_structure))
        
        # Save to output file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.writelines(nth_structure)
            print(f"Structure {n} saved to {output_file}.")
        
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python extract_xyz.py <file_path> <n> [<output_file>]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    n = int(sys.argv[2])  # Structure index (1-based)
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    print(output_file)
    # Extract the nth XYZ structure
    extract_nth_xyz(file_path, n, output_file)
