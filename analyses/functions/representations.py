"""
Build population representations: z-score, PCA, group by monkey × age.
"""

import numpy as np
from sklearn.decomposition import PCA


def zscore_neurons(X):
    """
    Z-score each neuron (row) across conditions (columns).
    Removes neurons with NaN or zero variance.

    Parameters
    ----------
    X : ndarray, shape (n_neurons, n_features)

    Returns
    -------
    X_clean : ndarray, shape (n_good, n_features)
    """
    good = np.all(np.isfinite(X), axis=1) & (np.std(X, axis=1) > 0)
    X = X[good]
    X = (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]
    return X


def pca_reduce(X, n_pcs):
    """
    Reduce neuron dimension via PCA.

    Input X has shape (n_neurons, n_features) where features = conditions.
    PCA is done on X.T so that neurons are the features being compressed.

    Parameters
    ----------
    X : ndarray, shape (n_neurons, n_features)
    n_pcs : int

    Returns
    -------
    X_pca : ndarray, shape (n_pcs, n_features)
    variance_explained : float
        Fraction of total variance explained.
    """
    n_neurons, n_features = X.shape
    n_pcs_actual = min(n_pcs, n_neurons, n_features)

    pca = PCA(n_components=n_pcs_actual)
    projected = pca.fit_transform(X.T).T  # (n_pcs_actual, n_features)

    # Pad with zeros if fewer PCs than requested
    if n_pcs_actual < n_pcs:
        projected = np.vstack([
            projected,
            np.zeros((n_pcs - n_pcs_actual, n_features))
        ])

    return projected, pca.explained_variance_ratio_.sum()


def build_representations(tuning, ids, groups, n_pcs, min_neurons=10, zscore=True):
    """
    Build PCA representations for each (monkey, group) combination.

    Parameters
    ----------
    tuning : ndarray, shape (n_neurons, n_features)
        Flattened tuning curves (e.g., 8 cues × 2 epochs = 16).
    ids : ndarray of str, shape (n_neurons,)
        Monkey ID per neuron.
    groups : ndarray of int, shape (n_neurons,)
        Group label per neuron (age tercile or window index).
    n_pcs : int
    min_neurons : int
    zscore : bool
        Whether to z-score neurons before PCA.

    Returns
    -------
    entries : list of dict
        Each dict has keys: 'monkey', 'group', 'matrix', 'n_neurons', 'var_explained'.
    """
    entries = []
    monkey_names = sorted(set(ids))

    for g in sorted(set(groups)):
        for mid in monkey_names:
            mask = (ids == mid) & (groups == g)
            if mask.sum() < min_neurons:
                continue

            X = tuning[mask]
            if zscore:
                X = zscore_neurons(X)
            else:
                good = np.all(np.isfinite(X), axis=1) & (np.std(X, axis=1) > 0)
                X = X[good]

            if X.shape[0] < min_neurons:
                continue

            matrix, var_exp = pca_reduce(X, n_pcs)

            entries.append({
                "monkey": mid,
                "group": g,
                "matrix": matrix,
                "n_neurons": X.shape[0],
                "var_explained": var_exp,
            })

    return entries


def build_window_entries(tuning, ids, neuron_age, n_pcs,
                         n_windows=20, min_neurons=10, zscore=True):
    """
    Build PCA representations for each (monkey, age window) combination.
    Each monkey's neurons are sorted by age and split into n_windows
    equal-count groups. Monkeys with too few neurons get fewer windows.

    Parameters
    ----------
    tuning : ndarray, shape (n_neurons, n_features)
    ids : ndarray of str, shape (n_neurons,)
    neuron_age : ndarray, shape (n_neurons,)
        Age in days for each neuron.
    n_pcs : int
    n_windows : int
    min_neurons : int
    zscore : bool

    Returns
    -------
    entries : list of dict
        Each dict has keys: 'monkey', 'group', 'center_days', 'matrix',
        'n_neurons', 'var_explained'.
    """
    entries = []
    monkey_names = sorted(set(ids))

    for mid in monkey_names:
        mk_mask = ids == mid
        mk_ages = neuron_age[mk_mask]
        mk_tuning = tuning[mk_mask]

        # Cap windows so each has at least min_neurons
        n_win = min(n_windows, len(mk_ages) // min_neurons)
        if n_win < 1:
            continue

        # Sort by age, split into equal-count groups
        order = np.argsort(mk_ages)
        groups = np.array_split(order, n_win)

        for wi, idx in enumerate(groups):
            X = mk_tuning[idx]
            center = mk_ages[idx].mean()
            if zscore:
                X = zscore_neurons(X)
            else:
                good = np.all(np.isfinite(X), axis=1) & (np.std(X, axis=1) > 0)
                X = X[good]
            if X.shape[0] < min_neurons:
                continue
            matrix, var_exp = pca_reduce(X, n_pcs)
            entries.append({
                'monkey': mid,
                'group': wi,
                'center_days': center,
                'matrix': matrix,
                'n_neurons': X.shape[0],
                'var_explained': var_exp,
            })

    return entries
