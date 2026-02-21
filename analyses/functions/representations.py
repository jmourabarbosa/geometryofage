"""
Build population representations: z-score, PCA, group by monkey × age.
"""

import numpy as np
from sklearn.decomposition import PCA


def _clean_neurons(X):
    """Remove neurons with NaN/Inf or zero variance."""
    good = np.all(np.isfinite(X), axis=1) & (np.std(X, axis=1) > 0)
    return X[good]


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
    X = _clean_neurons(X)
    X = (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]
    return X


def _build_entry(X, n_pcs, min_neurons, zscore, **extra):
    """Clean → check count → PCA → dict. Returns None if too few neurons."""
    X = zscore_neurons(X) if zscore else _clean_neurons(X)
    if X.shape[0] < min_neurons:
        return None
    matrix, var_exp = pca_reduce(X, n_pcs)
    entry = {'matrix': matrix, 'n_neurons': X.shape[0], 'var_explained': var_exp}
    entry.update(extra)
    return entry


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
    
    n_pcs_actual = n_pcs #min(n_pcs, n_neurons, n_features)

    pca = PCA(n_components=n_pcs_actual)
    projected = pca.fit_transform(X.T).T  # (n_pcs_actual, n_features)

    # Pad with zeros if fewer PCs than requested
    if n_pcs_actual < n_pcs:
        projected = np.vstack([
            projected,
            np.zeros((n_pcs - n_pcs_actual, n_features))
        ])

    return projected, pca.explained_variance_ratio_.sum()


def pca_reduce_tuning(grouped, n_pcs, min_neurons=10):
    """PCA-reduce tuning curves per monkey x age group.

    For each group:
      1. Flatten (n_neurons, n_conds, n_epochs) -> (n_neurons, n_conds*n_epochs)
      2. Z-score neurons across the flattened features
      3. Fit PCA on the flattened data
      4. Project each (condition, epoch) separately -> (n_pcs, n_conds, n_epochs)

    Parameters
    ----------
    grouped : dict
        {monkey: {age_group: ndarray (n_neurons, n_conditions, n_epochs)}}
    n_pcs : int
    min_neurons : int
        Groups with fewer clean neurons are skipped.

    Returns
    -------
    reduced : dict
        {monkey: {age_group: dict(tc, n_neurons, var_explained)}}
        tc has shape (n_pcs, n_conditions, n_epochs).
    """
    reduced = {}
    for mid, groups_dict in grouped.items():
        reduced[mid] = {}
        for g, tc in groups_dict.items():
            n_neurons, n_conds, n_epochs = tc.shape

            # Flatten, z-score, check count
            flat = tc.reshape(n_neurons, -1)
            flat = zscore_neurons(flat)
            if flat.shape[0] < min_neurons:
                continue

            # Fit PCA on flattened
            pca = PCA(n_components=n_pcs)
            pca.fit(flat.T)  # (n_features, n_neurons) -> learns components (n_pcs, n_neurons)

            # Project unflattened: for each (cond, epoch), project (n_neurons,) -> (n_pcs,)
            projected = np.zeros((n_pcs, n_conds, n_epochs))
            for c in range(n_conds):
                for e in range(n_epochs):
                    projected[:, c, e] = pca.components_ @ flat[:, c * n_epochs + e]

            reduced[mid][g] = dict(
                tc=projected,
                n_neurons=flat.shape[0],
                var_explained=pca.explained_variance_ratio_.sum(),
            )

    return reduced


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
            entry = _build_entry(tuning[mask], n_pcs, min_neurons, zscore,
                                 monkey=mid, group=g)
            if entry is not None:
                entries.append(entry)

    return entries


