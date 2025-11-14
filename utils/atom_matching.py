import torch

def sinkhorn_algorithm(cost_matrix, epsilon=1e-3, max_iter=100, tol=1e-6):
    """
    Perform the Sinkhorn-Knopp algorithm to compute the optimal transport matrix.

    Args:
        cost_matrix (torch.Tensor): The cost matrix of size (N, N), representing pairwise costs.
        epsilon (float): Regularization parameter for entropy. Default is 1e-3.
        max_iter (int): Maximum number of iterations. Default is 100.
        tol (float): Tolerance for convergence. Default is 1e-6.

    Returns:
        torch.Tensor: A doubly stochastic matrix (N, N) approximating the optimal transport matrix.
    """

    # Normalize epsilon and compute the exponent input
    exp_input = -cost_matrix / (epsilon + 1e-7)
    exp_input = torch.clamp(exp_input, min=-50, max=50)  # Clamp values to avoid exponential overflow
    K = torch.exp(exp_input)  # Exponential scaling (N, N)

    # Initialize the row and column marginals
    r = torch.ones(K.size(0), device=K.device) / K.size(0)
    c = torch.ones(K.size(1), device=K.device) / K.size(1)
    
    # distribution parameters, here the distribution is uniform distribution
    a=1.0
    b=1.0

    for _ in range(max_iter):
        r_prev = r.clone()  # Save previous row marginals
        c_prev = c.clone()  # Save previous column marginals

        # Update row and column marginals
        r = a / (K @ c + 1e-7)  # Avoid division by zero
        c = b / (K.t() @ r + 1e-7)  # Avoid division by zero

        # Check for convergence by measuring the relative change in row and column marginals
        r_diff = torch.norm(r - r_prev, p=1)
        c_diff = torch.norm(c - c_prev, p=1)
        if r_diff < tol and c_diff < tol:
            break

    # Return the doubly stochastic matrix
    return torch.diag(r) @ K @ torch.diag(c)

from scipy.optimize import linear_sum_assignment
import numpy as np

def hungarian_alignment(coords1, coords2):
    """
    Align two sets of coordinates using Hungarian algorithm to reorder atoms.
    """
    num_atoms = len(coords1)
    cost_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            cost_matrix[i, j] = np.linalg.norm(coords1[i] - coords2[j])
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return coords1, coords2[col_indices]


def hungarian_with_constraints(coords1, coords2, atom_types1, atom_types2):
    """
    Perform atom alignment using the Hungarian algorithm with atom type constraints.
    
    Args:
        coords1: Array of shape (num_atoms, 3) for the first structure's coordinates.
        coords2: Array of shape (num_atoms, 3) for the second structure's coordinates.
        atom_types1: List of atom types for the first structure.
        atom_types2: List of atom types for the second structure.
        
    Returns:
        reordered_coords2: Reordered coordinates of the second structure.
    """
    num_atoms = len(coords1)
    cost_matrix = np.zeros((num_atoms, num_atoms))
    large_penalty = 1e6  # Penalty for mismatched atom types

    # Build the cost matrix with constraints
    for i in range(num_atoms):
        for j in range(num_atoms):
            if atom_types1[i] != atom_types2[j]:
                cost_matrix[i, j] = large_penalty  # Penalize mismatched types
            else:
                cost_matrix[i, j] = np.linalg.norm(coords1[i] - coords2[j])  # Euclidean distance

    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Reorder coords2 based on the optimal assignment
    reordered_coords2 = coords2[col_indices]
    return reordered_coords2