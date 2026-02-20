"""
Statistical analyses for Procrustes distance comparisons.
"""

import numpy as np
import scipy.stats as sts


def assign_age_groups(ids, abs_age, n_age_groups):
    """Assign neurons to age groups per monkey based on quantiles."""
    monkey_names = sorted(set(ids))
    age_group = np.zeros(len(ids), dtype=int)
    for mid in monkey_names:
        mask = ids == mid
        edges = np.quantile(abs_age[mask], np.linspace(0, 1, n_age_groups + 1))
        age_group[mask] = np.clip(
            np.digitize(abs_age[mask], edges[1:-1]), 0, n_age_groups - 1
        )
    return age_group


def cross_monkey_analysis(entries, dist):
    """Adjusted cross-monkey distances and one-sample t-test.

    For each cross-monkey pair, subtract the average within-monkey baseline.
    """
    me = np.array([e['monkey'] for e in entries])
    monkey_names = sorted(set(me))
    n = len(entries)

    mean_within = {}
    for mid in monkey_names:
        idx = np.where(me == mid)[0]
        if len(idx) < 2:
            mean_within[mid] = np.nan
            continue
        mean_within[mid] = np.mean([dist[i, j] for i in idx for j in idx if j > i])

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
    t_stat, p_val = sts.ttest_1samp(cross_adj, 0)

    return dict(
        cross_raw=cross_raw, cross_adj=cross_adj,
        mean_within=mean_within, within_all_pairs=within_all,
        t_stat=t_stat, p_val=p_val, monkey_names=monkey_names,
    )


def cross_age_analysis(entries, dist):
    """Within-age cross-monkey vs across-age within-monkey analysis."""
    me = np.array([e['monkey'] for e in entries])
    ge = np.array([e['group'] for e in entries])
    monkey_names = sorted(set(me))
    n = len(entries)

    mean_across_age = {}
    for mid in monkey_names:
        idx = np.where(me == mid)[0]
        if len(idx) < 2:
            mean_across_age[mid] = np.nan
            continue
        mean_across_age[mid] = np.mean([dist[i, j] for i in idx for j in idx if j > i])

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
    t_stat, p_val = sts.ttest_1samp(same_age_adj, 0)

    return dict(
        same_age_raw=same_age_raw, same_age_adj=same_age_adj,
        diff_age_raw=diff_age_raw, mean_across_age=mean_across_age,
        t_stat=t_stat, p_val=p_val, monkey_names=monkey_names,
    )
