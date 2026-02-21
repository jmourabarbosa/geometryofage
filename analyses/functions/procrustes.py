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


def two_stage_alignment(reduced, n_dims=3):
    """Two-stage Procrustes alignment of tuning-curve representations.

    Stage 1: Within each monkey, align age groups via GPA.
    Stage 2: Align monkey means via GPA, propagate rotations to individuals.

    Parameters
    ----------
    reduced : dict
        {monkey: {age_group: dict(tc=ndarray (n_pcs, n_conds, n_epochs), ...)}}
    n_dims : int
        Number of PCs to use for alignment (typically 3 for 3-D plots).

    Returns
    -------
    all_aligned : list of ndarray, each shape (n_points, n_dims)
        All aligned individual representations.
    all_labels : list of dict
        Each has keys 'monkey', 'group'.
    grand_mean : ndarray, shape (n_points, n_dims)
        Grand Fréchet mean across all monkeys.
    monkey_means : dict
        {monkey: ndarray (n_points, n_dims)} aligned monkey means.
    """
    # Collect representations as (n_points, n_dims) matrices
    # n_points = n_conds * n_epochs (e.g. 4 * 2 = 8)
    monkey_reps = {}  # {monkey: list of (n_points, n_dims)}
    monkey_labels = {}  # {monkey: list of group labels}

    for mid, groups_dict in reduced.items():
        reps = []
        labels = []
        for g, info in sorted(groups_dict.items()):
            tc = info['tc'][:n_dims]  # (n_dims, n_conds, n_epochs)
            n_conds, n_epochs = tc.shape[1], tc.shape[2]
            # Reshape to (n_points, n_dims) where points = conds * epochs
            mat = tc.reshape(n_dims, -1).T  # (n_points, n_dims)
            reps.append(mat)
            labels.append(g)
        monkey_reps[mid] = reps
        monkey_labels[mid] = labels

    # Stage 1: Within-monkey GPA
    monkey_aligned = {}  # {monkey: ndarray (n_groups, n_points, n_dims)}
    monkey_mean = {}     # {monkey: ndarray (n_points, n_dims)}

    for mid, reps in monkey_reps.items():
        if len(reps) == 1:
            arr = reps[0].copy()
            arr -= arr.mean(axis=0)
            monkey_aligned[mid] = arr[np.newaxis]
            monkey_mean[mid] = arr.copy()
        else:
            aligned, mean = generalized_procrustes(reps)
            monkey_aligned[mid] = aligned
            monkey_mean[mid] = mean

    # Stage 2: Across-monkey GPA on monkey means
    monkey_ids = sorted(monkey_mean.keys())
    means_list = [monkey_mean[mid] for mid in monkey_ids]
    aligned_means, grand_mean = generalized_procrustes(means_list)

    # Propagate stage-2 rotations to individual representations
    all_aligned = []
    all_labels = []
    monkey_means_aligned = {}

    for k, mid in enumerate(monkey_ids):
        # Find rotation from old mean to aligned mean
        old_mean = monkey_mean[mid]
        old_mean_c = old_mean - old_mean.mean(axis=0)
        new_mean_c = aligned_means[k] - aligned_means[k].mean(axis=0)
        R, _ = orthogonal_procrustes(old_mean_c, new_mean_c)

        monkey_means_aligned[mid] = aligned_means[k]

        for i, g in enumerate(monkey_labels[mid]):
            rep = monkey_aligned[mid][i]
            rep_c = rep - rep.mean(axis=0)
            rep_rotated = rep_c @ R
            all_aligned.append(rep_rotated)
            all_labels.append({'monkey': mid, 'group': g})

    return all_aligned, all_labels, grand_mean, monkey_means_aligned


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
