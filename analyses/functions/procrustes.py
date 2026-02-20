"""
Compute Procrustes distance matrices.
"""

import numpy as np
from scipy.spatial import procrustes


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
