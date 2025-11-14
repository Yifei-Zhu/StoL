import torch
import numpy as np
def pca_align_single(coords):
    """
    计算 PCA 对齐的特征向量。
    
    参数:
        coords (torch.Tensor): 输入坐标，Shape为 (N, 3)，其中 N 是点的数量。
        
    返回:
        torch.Tensor: 排序后的特征向量，Shape为 (3, 3)。
    """
    # 计算协方差矩阵
    # 注意：PyTorch 没有内置的协方差函数，因此需要手动计算
    # 这里假设 coords 已经是中心化的
    N = coords.shape[0]
    cov = (coords.T @ coords) / (N - 1)  # Shape: (3, 3)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # eigenvectors: (3, 3)
    
    # 按特征值降序排序
    order = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, order]
    
    return eigenvectors

def compute_centroid(coords):
    return torch.mean(coords, dim=0)

def align_centroid(coords, centroid):
    return coords - centroid

def pca_align(mol1, mol2):
    
    centroid1 = compute_centroid(mol1)
    centroid2 = compute_centroid(mol2)

    molecule1_centered = align_centroid(mol1, centroid1)
    molecule2_centered = align_centroid(mol2, centroid2)

    pca1 = pca_align_single(molecule1_centered)  # Shape: (3, 3)
    pca2 = pca_align_single(molecule2_centered)  # Shape: (3, 3)

    rotation = pca2 @ pca1.T  # Shape: (3, 3)

    molecule2_aligned = molecule2_centered @ rotation  # Shape: (N, 3)
    
    return molecule2_aligned


def kabsch_algorithm_classical(P, Q):
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix (R) and translation vector (t)
    that aligns point set P to Q in a least-squares sense.

    Args:
        P (np.ndarray): Predicted positions (N, 3).
        Q (np.ndarray): Target positions (N, 3).

    Returns:
        R (np.ndarray): Optimal rotation matrix (3, 3).
        t (np.ndarray): Translation vector (3,).
    """
    assert P.shape == Q.shape, "Point sets must have the same shape."
    assert P.shape[1] == 3, "Point sets must have 3D coordinates."

    # Compute centroids
    centroid_P = P.mean(axis=0, keepdims=True)  # (1, 3)
    centroid_Q = Q.mean(axis=0, keepdims=True)  # (1, 3)

    # Center the points
    P_centered = P - centroid_P  # (N, 3)
    Q_centered = Q - centroid_Q  # (N, 3)

    # Compute covariance matrix
    H = np.dot(P_centered.T, Q_centered)  # (3, 3)

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)  # U, Vt: (3, 3)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
            
    # Compute translation vector
    t = centroid_Q.squeeze() - np.dot(centroid_P.squeeze(), R)

    
    P_aligned = np.dot(P, R) + t

    return P_aligned


def kabsch_algorithm(P, Q):
    centroid_P = P.mean(dim=0)
    centroid_Q = Q.mean(dim=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = torch.matmul(P_centered.T, Q_centered)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.matmul(Vt.T, U.T)

    # 如果行列式为负，用非 inplace 操作修复
    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()  # 避免修改原始张量
        Vt[-1, :] *= -1
        R = torch.matmul(Vt.T, U.T)

    t = centroid_Q - torch.matmul(centroid_P, R)
    return R, t

def weighted_kabsch_algorithm(P, Q, weights):
    """
    Perform weighted Kabsch algorithm to find the optimal rotation matrix (R) and translation vector (t)
    that aligns point set P to Q, taking into account weights for each point.

    Args:
        P (torch.Tensor): Predicted positions (N, 3).
        Q (torch.Tensor): Target positions (N, 3).
        weights (torch.Tensor): Weights for each point (N, 1).

    Returns:
        R (torch.Tensor): Optimal rotation matrix (3, 3).
        t (torch.Tensor): Translation vector (3,).
    """
    # Normalize weights to sum to 1
    weights_normalized = weights / (weights.sum() + 1e-7)
    
    # Compute weighted centroids
    centroid_P = torch.sum(P * weights_normalized, dim=0)
    centroid_Q = torch.sum(Q * weights_normalized, dim=0)
    
    # Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute weighted covariance matrix
    H = torch.matmul(P_centered.T * weights_normalized.T, Q_centered)
    
    # Singular Value Decomposition (SVD)
    U, S, Vt = torch.linalg.svd(H)
    
    # Compute rotation matrix
    R = torch.matmul(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()  # Avoid modifying original tensor
        Vt[-1, :] *= -1
        R = torch.matmul(Vt.T, U.T)
    
    # Compute translation vector
    t = centroid_Q - torch.matmul(centroid_P, R)
    
    return R, t


