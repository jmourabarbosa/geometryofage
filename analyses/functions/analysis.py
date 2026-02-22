"""
Statistical analyses for Procrustes distance comparisons.
"""

import numpy as np
import scipy.stats as sts


def extract_entry_arrays(entries):
    """Extract monkey and group arrays from entries list.

    Parameters
    ----------
    entries : list of dict
        Each dict must have 'monkey' and 'group' keys.

    Returns
    -------
    monkeys : ndarray of str
    groups : ndarray of int
    """
    monkeys = np.array([e['monkey'] for e in entries])
    groups = np.array([e['group'] for e in entries])
    return monkeys, groups


def _mean_within_monkey(me, dist):
    """Compute mean pairwise distance within each monkey.

    Parameters
    ----------
    me : ndarray of str
        Monkey ID per entry.
    dist : ndarray, shape (n, n)

    Returns
    -------
    mean_within : dict
        {monkey_id: mean_distance} for each monkey.
    """
    mean_within = {}
    for mid in sorted(set(me)):
        idx = np.where(me == mid)[0]
        if len(idx) < 2:
            mean_within[mid] = np.nan
            continue
        mean_within[mid] = np.nanmean([dist[i, j] for i in idx for j in idx if j > i])
    return mean_within


def assign_per_monkey_age_groups(ids, abs_age, n_bins):
    """Assign neurons to age groups using per-monkey quantile binning.

    Parameters
    ----------
    ids : ndarray of str
        Monkey ID per neuron.
    abs_age : ndarray
        Absolute age in months per neuron.
    n_bins : int
        Number of quantile bins per monkey.

    Returns
    -------
    groups : ndarray of int
        Age group per neuron (0 to n_bins-1).
    monkey_edges : dict
        {monkey_id: tuple of edges} — inner bin edges per monkey.
    """
    groups = np.full(len(ids), -1, dtype=int)
    monkey_edges = {}
    for mid in sorted(set(ids)):
        mask = ids == mid
        ages_m = abs_age[mask]
        pcts = np.linspace(0, 100, n_bins + 1)[1:-1]
        edges = tuple(np.unique(np.percentile(ages_m, pcts)))
        monkey_edges[mid] = edges
        groups[mask] = np.digitize(ages_m, edges)
    return groups, monkey_edges


def assign_age_groups(age, edges):
    """Assign neurons to age groups using fixed bin edges.

    Parameters
    ----------
    age : ndarray
        Age per neuron (e.g. absolute age in months).
    edges : sequence of float
        Inner bin edges. For 3 groups [before, during, after] pass
        two values, e.g. ``(48, 60)``: group 0 = age < 48,
        group 1 = 48 <= age < 60, group 2 = age >= 60.

    Returns
    -------
    age_group : ndarray of int
    """
    return np.digitize(age, edges)


def cross_monkey_analysis(entries, dist):
    """Adjusted cross-monkey distances and one-sample t-test.

    For each cross-monkey pair, subtract the average within-monkey baseline.
    """
    me, _ = extract_entry_arrays(entries)
    monkey_names = sorted(set(me))
    n = len(entries)

    mean_within = _mean_within_monkey(me, dist)

    cross_raw, cross_adj = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if me[i] == me[j]:
                continue
            d = dist[i, j]
            baseline = (mean_within[me[i]] + mean_within[me[j]]) / 2
            cross_raw.append(d)
            cross_adj.append(d - baseline)

    cross_raw = np.array(cross_raw)
    cross_adj = np.array(cross_adj)
    within_all = np.array(
        [dist[i, j] for i in range(n) for j in range(i + 1, n) if me[i] == me[j]]
    )
    t_stat, p_val = sts.ttest_1samp(cross_adj, 0, nan_policy='omit')

    return dict(
        cross_raw=cross_raw, cross_adj=cross_adj,
        mean_within=mean_within, within_all_pairs=within_all,
        t_stat=t_stat, p_val=p_val, monkey_names=monkey_names,
    )


def cross_monkey_by_group(task_data, psth_data, age_groups, n_pcs, min_neurons,
                          group_labels):
    """Cross-monkey Procrustes distance per age group with linear regression.

    Only uses monkeys with >= min_neurons in every age group.

    Parameters
    ----------
    task_data : dict
        {task_name: dict(ids=..., abs_age=...)}
    psth_data : dict
        {task_name: dict(flat=...)}
    age_groups : dict
        {task_name: ndarray of int}
    n_pcs, min_neurons : int
    group_labels : list of str

    Returns
    -------
    results : dict
        {task_name: dict(group_dists, slope, intercept, r, p, se, common)}
        group_dists is {group_idx: ndarray of distances}.
    pooled : dict
        Pooled regression across all tasks: {slope, intercept, r, p, se}.
    """
    from .representations import build_representations
    from .procrustes import procrustes_distance_matrix

    n_groups = len(group_labels)
    results = {}
    pooled_groups, pooled_dists = [], []

    for task_name in task_data:
        ids = task_data[task_name]['ids']
        tuning = psth_data[task_name]['flat']
        ag = age_groups[task_name]

        monkeys = sorted(set(ids))
        common = [mid for mid in monkeys
                  if all(np.sum((ids == mid) & (ag == g)) >= min_neurons
                         for g in range(n_groups))]

        mask = np.isin(ids, common)
        entries = build_representations(tuning[mask], ids[mask], ag[mask],
                                        n_pcs=n_pcs, min_neurons=min_neurons,
                                        zscore=True)
        dist = procrustes_distance_matrix(entries)
        me, ge = extract_entry_arrays(entries)
        n = len(entries)

        group_dists = {}
        all_g, all_d = [], []
        for g in range(n_groups):
            dists = np.array([dist[i, j] for i in range(n) for j in range(i + 1, n)
                              if me[i] != me[j] and ge[i] == g and ge[j] == g])
            group_dists[g] = dists
            all_g.extend([g] * len(dists))
            all_d.extend(dists)

        all_g = np.array(all_g)
        all_d = np.array(all_d)
        slope, intercept, r, p, se = sts.linregress(all_g, all_d)

        pooled_groups.append(all_g)
        pooled_dists.append(all_d)

        results[task_name] = dict(
            group_dists=group_dists, slope=slope, intercept=intercept,
            r=r, p=p, se=se, common=common,
            n_monkeys=len(monkeys), n_neurons=mask.sum(),
            n_total=len(ids),
        )

    pg = np.concatenate(pooled_groups)
    pd = np.concatenate(pooled_dists)
    s_all, i_all, r_all, p_all, se_all = sts.linregress(pg, pd)
    pooled = dict(slope=s_all, intercept=i_all, r=r_all, p=p_all, se=se_all)

    return results, pooled


def cross_age_analysis(entries, dist):
    """Within-age cross-monkey vs across-age within-monkey analysis."""
    me, ge = extract_entry_arrays(entries)
    monkey_names = sorted(set(me))
    n = len(entries)

    mean_across_age = _mean_within_monkey(me, dist)

    same_age_raw, same_age_adj, diff_age_raw = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if me[i] == me[j]:
                continue
            if ge[i] == ge[j]:
                d = dist[i, j]
                baseline = (mean_across_age[me[i]] + mean_across_age[me[j]]) / 2
                same_age_raw.append(d)
                same_age_adj.append(d - baseline)
            else:
                diff_age_raw.append(dist[i, j])

    same_age_raw = np.array(same_age_raw)
    same_age_adj = np.array(same_age_adj)
    diff_age_raw = np.array(diff_age_raw)
    t_stat, p_val = sts.ttest_1samp(same_age_adj, 0, nan_policy='omit')

    return dict(
        same_age_raw=same_age_raw, same_age_adj=same_age_adj,
        diff_age_raw=diff_age_raw, mean_across_age=mean_across_age,
        t_stat=t_stat, p_val=p_val, monkey_names=monkey_names,
    )


