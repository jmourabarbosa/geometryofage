"""
Compute Procrustes distance matrices and Generalized Procrustes Analysis.
"""

import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes


def generalized_procrustes(Xs, max_iter=100, tol=1e-8):
    """Generalized Procrustes Analysis (iterative).

    Aligns a set of matrices to their Fréchet mean via iterative
    rotation/reflection alignment.

    Parameters
    ----------
    Xs : list of ndarray, each shape (n_points, n_dims)
        Matrices to align.  All must have the same shape.
    max_iter : int
    tol : float
        Convergence threshold on mean change.

    Returns
    -------
    aligned : ndarray, shape (n_matrices, n_points, n_dims)
        Aligned matrices.
    mean : ndarray, shape (n_points, n_dims)
        Fréchet mean of the aligned matrices.
    """
    Xs = [X.copy() for X in Xs]
    n = len(Xs)

    # Center each matrix
    for i in range(n):
        Xs[i] -= Xs[i].mean(axis=0)

    # Initialize mean as first matrix
    mean = Xs[0].copy()

    for _ in range(max_iter):
        # Align each matrix to current mean
        for i in range(n):
            R, _ = orthogonal_procrustes(Xs[i], mean)
            Xs[i] = Xs[i] @ R

        # Recompute mean
        new_mean = np.mean(Xs, axis=0)
        new_mean -= new_mean.mean(axis=0)

        # Check convergence
        if np.linalg.norm(new_mean - mean) < tol:
            mean = new_mean
            break
        mean = new_mean

    aligned = np.array(Xs)
    return aligned, mean


def procrustes_distance_matrix(entries):
    """
    Compute pairwise Procrustes distances between all entries.

    Each entry has a 'matrix' of shape (n_pcs, n_features).
    Procrustes compares the transposed matrices (n_features, n_pcs),
    treating features (conditions) as the matched points.

    Parameters
    ----------
    entries : list of dict
        Each must have key 'matrix' with shape (n_pcs, n_features).

    Returns
    -------
    dist : ndarray, shape (n, n)
        Symmetric distance matrix with zeros on the diagonal.
    """
    n = len(entries)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            A = entries[i]["matrix"].T  # (n_features, n_pcs)
            B = entries[j]["matrix"].T
            _, _, d = procrustes(A, B)
            dist[i, j] = d
            dist[j, i] = d

    return dist


def procrustes_disparity(A, B):
    """Compute Procrustes disparity between two matrices.

    Parameters
    ----------
    A, B : ndarray, shape (n_points, n_dims)

    Returns
    -------
    d : float
        Procrustes disparity.
    """
    _, _, d = procrustes(A, B)
    return d


def _upper_tri(mat):
    """Extract upper triangle (excluding diagonal) as 1-D array."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]
