"""
Load behavioral data and compute behavioral distance matrices.
"""

import numpy as np
import pandas as pd
import scipy.stats as sts

from .analysis import assign_age_groups


# Map neural task names to behavioral task names
TASK_MAP_SAC = {
    'ODR 1.5s': 'ODR',
    'ODR 3.0s': 'ODR3',
    'ODRd': 'ODRd',
}

TASK_MAP_PERCORR = {
    'ODR 1.5s': 'ODR',
    'ODR 3.0s': 'ODR3',
    'ODRd': ['ODRdistractor_cardinal', 'ODRdistractor_diagonal'],
}

DI_COLS_8 = [f'DI_{i}' for i in range(1, 9)]
RT_COLS_8 = [f'RT_{i}' for i in range(1, 9)]


def load_behavioral_data(sac_path, sac_odrd_path=None):
    """Load saccade behavioral data (DI and RT).

    Parameters
    ----------
    sac_path : str
        Path to sac_data.csv (ODR / ODR3).
    sac_odrd_path : str, optional
        Path to sac_odrd.csv (ODRd). If given, ODRd sessions are appended
        with Monkey, Task='ODRd', age_month, DI (mean), RT (mean).
    """
    df = pd.read_csv(sac_path)
    df['age_month'] = df['age_month']  # already present

    if sac_odrd_path is not None:
        odrd = pd.read_csv(sac_odrd_path)
        odrd = odrd.rename(columns={'ID': 'Monkey'})
        odrd['age_month'] = odrd['age']  # already in months
        odrd['Task'] = 'ODRd'
        # Use session-level mean DI and RT (columns 'DI' and 'RT')
        odrd = odrd[['Monkey', 'Task', 'age_month', 'DI', 'RT']]
        df = pd.concat([df, odrd], ignore_index=True)

    return df


def load_percorr_data(beh_path, odrd_path):
    """Load and concatenate percent-correct data from both files.

    Adds ``age_month`` to the ODRd dataframe (derived from ``age`` in days).

    Returns
    -------
    DataFrame with columns: Monkey, Task, age_month, percorr
    """
    beh = pd.read_csv(beh_path)[['Monkey', 'Task', 'age_month', 'percorr']]
    odrd = pd.read_csv(odrd_path)
    odrd['age_month'] = odrd['age'] / 365.0 * 12.0
    odrd = odrd[['Monkey', 'Task', 'age_month', 'percorr']]
    return pd.concat([beh, odrd], ignore_index=True)


def behavioral_distance_matrices(beh_df, entries, age_edges, task_name):
    """Compute DI and RT distance matrices matching neural Procrustes entries.

    For each entry (monkey x age group), average DI and RT across sessions
    that match that monkey and age group, then compute pairwise absolute
    differences.

    Parameters
    ----------
    beh_df : DataFrame
        Behavioral data from sac_data.csv.
    entries : list of dict
        Neural entries with 'monkey' and 'group' keys.
    age_edges : tuple
        Same edges used for neural age groups.
    task_name : str
        Neural task name (e.g. 'ODR 1.5s').

    Returns
    -------
    di_dist : ndarray (n, n)
    rt_dist : ndarray (n, n)
    di_vals : ndarray (n,)
    rt_vals : ndarray (n,)
    """
    beh_task = TASK_MAP_SAC.get(task_name)
    if beh_task is None:
        raise ValueError(f'No DI/RT data for task {task_name!r}')

    sub = beh_df[beh_df['Task'] == beh_task].copy()
    sub['age_group'] = assign_age_groups(sub['age_month'].values, age_edges)

    # ODR/ODR3 have per-direction columns; ODRd has session-level mean columns
    use_mean_col = (beh_task == 'ODRd')

    n = len(entries)
    di_vals = np.full(n, np.nan)
    rt_vals = np.full(n, np.nan)

    for idx, e in enumerate(entries):
        mask = (sub['Monkey'] == e['monkey']) & (sub['age_group'] == e['group'])
        rows = sub[mask]
        if len(rows) == 0:
            continue
        if use_mean_col:
            di_vals[idx] = np.nanmean(rows['DI'].values)
            rt_vals[idx] = np.nanmean(rows['RT'].values)
        else:
            di_vals[idx] = np.nanmean(rows[DI_COLS_8].values)
            rt_vals[idx] = np.nanmean(rows[RT_COLS_8].values)

    di_dist = np.abs(di_vals[:, None] - di_vals[None, :])
    rt_dist = np.abs(rt_vals[:, None] - rt_vals[None, :])

    return di_dist, rt_dist, di_vals, rt_vals


def percorr_distance_matrix(percorr_df, entries, age_edges, task_name):
    """Compute percent-correct distance matrix matching neural entries.

    Parameters
    ----------
    percorr_df : DataFrame
        Concatenated percent-correct data (from load_percorr_data).
    entries : list of dict
    age_edges : tuple
    task_name : str

    Returns
    -------
    pc_dist : ndarray (n, n)
    pc_vals : ndarray (n,)
    """
    tasks = TASK_MAP_PERCORR.get(task_name)
    if tasks is None:
        raise ValueError(f'No percorr data for task {task_name!r}')
    if isinstance(tasks, str):
        tasks = [tasks]

    sub = percorr_df[percorr_df['Task'].isin(tasks)].copy()
    sub['age_group'] = assign_age_groups(sub['age_month'].values, age_edges)

    n = len(entries)
    pc_vals = np.full(n, np.nan)

    for idx, e in enumerate(entries):
        mask = (sub['Monkey'] == e['monkey']) & (sub['age_group'] == e['group'])
        rows = sub[mask]
        if len(rows) == 0:
            continue
        pc_vals[idx] = np.nanmean(rows['percorr'].values)

    pc_dist = np.abs(pc_vals[:, None] - pc_vals[None, :])
    return pc_dist, pc_vals


def _upper_tri(mat):
    """Extract upper triangle (excluding diagonal) as 1-D array."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def _spearman_pair(neural_vec, beh_vec):
    """Spearman correlation between two vectors, skipping NaNs."""
    valid = np.isfinite(neural_vec) & np.isfinite(beh_vec)
    if valid.sum() < 3:
        return dict(r=np.nan, p=np.nan, n_pairs=int(valid.sum()))
    r, p = sts.spearmanr(neural_vec[valid], beh_vec[valid])
    return dict(r=r, p=p, n_pairs=int(valid.sum()))


def correlate_behavior_neural(neural_dist, beh_dists):
    """Correlate behavioral and neural distance matrices (upper triangle).

    Uses Spearman rank correlation on upper-triangle pairs.

    Parameters
    ----------
    neural_dist : ndarray (n, n)
    beh_dists : dict
        {label: ndarray (n, n)} — behavioral distance matrices to correlate.

    Returns
    -------
    dict  {label: {r, p, n_pairs}}
    """
    neural_vec = _upper_tri(neural_dist)
    out = {}
    for label, beh_mat in beh_dists.items():
        out[label] = _spearman_pair(neural_vec, _upper_tri(beh_mat))
    return out


def bootstrap_correlation(neural_vec, beh_vec, n_boot=1000, seed=42):
    """Bootstrap Spearman rho by resampling pairs.

    Parameters
    ----------
    neural_vec, beh_vec : 1-D arrays (same length)
    n_boot : int
    seed : int

    Returns
    -------
    boots : ndarray (n_boot,)  — bootstrap distribution of rho values
    """
    valid = np.isfinite(neural_vec) & np.isfinite(beh_vec)
    nv = neural_vec[valid]
    bv = beh_vec[valid]
    n = len(nv)
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.std(nv[idx]) == 0 or np.std(bv[idx]) == 0:
            continue
        boots[b], _ = sts.spearmanr(nv[idx], bv[idx])
    return boots
