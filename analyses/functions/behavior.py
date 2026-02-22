"""
Load behavioral data and compute behavioral distance matrices.
"""

import numpy as np
import pandas as pd

from .analysis import assign_age_groups


# Map neural task names to behavioral task names
TASK_MAP_SAC = {
    'ODR 1.5s': 'ODR',
    'ODR 3.0s': 'ODR3',
    'ODRd': 'ODRd',
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
    if sac_odrd_path is not None:
        odrd = pd.read_csv(sac_odrd_path)
        odrd = odrd.rename(columns={'ID': 'Monkey'})
        odrd['age_month'] = odrd['age']  # already in months
        odrd['Task'] = 'ODRd'
        # Use session-level mean DI and RT (columns 'DI' and 'RT')
        odrd = odrd[['Monkey', 'Task', 'age_month', 'DI', 'RT']]
        df = pd.concat([df, odrd], ignore_index=True)

    return df


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


def get_behavioral_values(beh_df, entries, task_name, monkey_edges):
    """Get per-entry mean DI and RT using per-monkey age bin edges.

    Unlike ``behavioral_distance_matrices`` (which uses global age edges),
    this function uses monkey-specific bin edges for age grouping.

    Parameters
    ----------
    beh_df : DataFrame
        Behavioral data from ``load_behavioral_data``.
    entries : list of dict
        Each dict has 'monkey' and 'group' keys.
    task_name : str
        Neural task name (e.g. 'ODR 1.5s').
    monkey_edges : dict
        {(task_name, monkey_id): tuple of edges} — per-monkey age bin edges.

    Returns
    -------
    di_vals : ndarray (n,)
    rt_vals : ndarray (n,)
    """
    beh_task = TASK_MAP_SAC[task_name]
    sub = beh_df[beh_df['Task'] == beh_task].copy()
    use_mean = (beh_task == 'ODRd')

    n = len(entries)
    di_vals = np.full(n, np.nan)
    rt_vals = np.full(n, np.nan)

    for idx, e in enumerate(entries):
        mid, grp = e['monkey'], e['group']
        edges = monkey_edges[(task_name, mid)]
        rows = sub[sub['Monkey'] == mid].copy()
        if len(rows) == 0:
            continue
        rows['ag'] = np.digitize(rows['age_month'].values, edges)
        rows = rows[rows['ag'] == grp]
        if len(rows) == 0:
            continue
        if use_mean:
            di_vals[idx] = np.nanmean(rows['DI'].values)
            rt_vals[idx] = np.nanmean(rows['RT'].values)
        else:
            di_vals[idx] = np.nanmean(rows[DI_COLS_8].values)
            rt_vals[idx] = np.nanmean(rows[RT_COLS_8].values)
    return di_vals, rt_vals
