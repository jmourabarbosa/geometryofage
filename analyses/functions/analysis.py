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
        mean_within[mid] = np.mean([dist[i, j] for i in idx for j in idx if j > i])
    return mean_within


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
    t_stat, p_val = sts.ttest_1samp(cross_adj, 0)

    return dict(
        cross_raw=cross_raw, cross_adj=cross_adj,
        mean_within=mean_within, within_all_pairs=within_all,
        t_stat=t_stat, p_val=p_val, monkey_names=monkey_names,
    )


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
    t_stat, p_val = sts.ttest_1samp(same_age_adj, 0)

    return dict(
        same_age_raw=same_age_raw, same_age_adj=same_age_adj,
        diff_age_raw=diff_age_raw, mean_across_age=mean_across_age,
        t_stat=t_stat, p_val=p_val, monkey_names=monkey_names,
    )


# ── Behavioral performance helpers ──────────────────────────────────────────

def load_behavioral_perf(beh_odr_path, beh_odrd_path, task_metadata, n_age_groups):
    """Load behavioral performance CSVs and bin into age groups matching neural data.

    Parameters
    ----------
    beh_odr_path : str
        Path to ODR behavioral CSV.
    beh_odrd_path : str
        Path to ODRd behavioral CSV.
    task_metadata : dict
        {task_name: (ids, abs_age)} — neural IDs and absolute ages per task.
    n_age_groups : int

    Returns
    -------
    perf_by_entry : dict
        {(task_name, monkey, group): mean_percorr}
    """
    import pandas as pd

    beh_odr = pd.read_csv(beh_odr_path)
    beh_odrd = pd.read_csv(beh_odrd_path)
    beh_odr['abs_age_months'] = beh_odr['age'] / 365.0 * 12.0
    beh_odrd['abs_age_months'] = beh_odrd['age'] / 365.0 * 12.0

    beh_by_task = {
        'ODR 1.5s': beh_odr[beh_odr['Task'] == 'ODR'],
        'ODR 3.0s': beh_odr[beh_odr['Task'] == 'ODR3'],
        'ODRd':     beh_odrd,
    }

    perf_by_entry = {}
    for task_name, (neural_ids, neural_abs_age) in task_metadata.items():
        beh_df = beh_by_task.get(task_name)
        if beh_df is None:
            continue
        for mid in sorted(set(neural_ids)):
            neural_ages_mk = neural_abs_age[neural_ids == mid]
            edges = np.quantile(neural_ages_mk, np.linspace(0, 1, n_age_groups + 1))
            beh_mk = beh_df[beh_df['Monkey'] == mid].copy()
            if len(beh_mk) == 0:
                continue
            beh_mk['group'] = np.clip(
                np.digitize(beh_mk['abs_age_months'].values, edges[1:-1]),
                0, n_age_groups - 1,
            )
            for g in range(n_age_groups):
                sessions = beh_mk[beh_mk['group'] == g]
                if len(sessions) > 0:
                    perf_by_entry[(task_name, mid, g)] = sessions['percorr'].mean()

    return perf_by_entry


def perf_vs_within_monkey_pairs(results, perf_by_entry):
    """Within-monkey cross-age pairs: Procrustes distance vs |Δ performance|.

    Parameters
    ----------
    results : dict
        {task_name: {entries, dist, ...}}
    perf_by_entry : dict
        {(task_name, monkey, group): mean_percorr}

    Returns
    -------
    scatter : dict
        {task_name: {x, y, labels}} where x=|Δ perf|, y=Procrustes distance.
    """
    scatter = {}
    for task_name, R in results.items():
        me, ge = extract_entry_arrays(R['entries'])
        dist = R['dist']
        n = len(me)

        x, y, labels = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                if me[i] != me[j]:
                    continue
                pi = perf_by_entry.get((task_name, me[i], ge[i]))
                pj = perf_by_entry.get((task_name, me[j], ge[j]))
                if pi is None or pj is None:
                    continue
                x.append(abs(pi - pj))
                y.append(dist[i, j])
                labels.append(f'{me[i]} G{ge[i]}-G{ge[j]}')

        scatter[task_name] = dict(x=np.array(x), y=np.array(y), labels=labels)
    return scatter


def perf_vs_cross_monkey(results, perf_by_entry, n_age_groups):
    """Cross-monkey distance (mean over age) vs |Δ mean performance|.

    Parameters
    ----------
    results : dict
    perf_by_entry : dict
    n_age_groups : int

    Returns
    -------
    scatter : dict
        {task_name: {x, y, labels}}
    """
    scatter = {}
    for task_name, R in results.items():
        me, _ = extract_entry_arrays(R['entries'])
        dist = R['dist']
        monkey_names = sorted(set(me))

        mean_perf = {}
        for mid in monkey_names:
            perfs = [perf_by_entry[(task_name, mid, g)]
                     for g in range(n_age_groups)
                     if (task_name, mid, g) in perf_by_entry]
            if perfs:
                mean_perf[mid] = np.mean(perfs)

        x, y, labels = [], [], []
        for ia, ma in enumerate(monkey_names):
            for mb in monkey_names[ia + 1:]:
                if ma not in mean_perf or mb not in mean_perf:
                    continue
                idx_a = np.where(me == ma)[0]
                idx_b = np.where(me == mb)[0]
                pair_dists = [dist[i, j] for i in idx_a for j in idx_b]
                if len(pair_dists) == 0:
                    continue
                x.append(abs(mean_perf[ma] - mean_perf[mb]))
                y.append(np.mean(pair_dists))
                labels.append(f'{ma}-{mb}')

        scatter[task_name] = dict(x=np.array(x), y=np.array(y), labels=labels)
    return scatter


def perf_vs_within_monkey_distance(results, perf_by_entry):
    """Per-entry performance vs mean within-monkey Procrustes distance.

    Parameters
    ----------
    results : dict
    perf_by_entry : dict

    Returns
    -------
    scatter : dict
        {task_name: {x, y, labels, colors, monkey_names}}
    """
    scatter = {}
    for task_name, R in results.items():
        me, ge = extract_entry_arrays(R['entries'])
        dist = R['dist']
        monkey_names = sorted(set(me))
        n = len(me)
        cmap = {m: f'C{i}' for i, m in enumerate(monkey_names)}

        x, y, labels, colors = [], [], [], []
        for i in range(n):
            mid, gi = me[i], ge[i]
            same_mk = [j for j in range(n) if me[j] == mid and j != i]
            if len(same_mk) == 0:
                continue
            mean_d = np.mean([dist[i, j] for j in same_mk])
            perf = perf_by_entry.get((task_name, mid, gi))
            if perf is None:
                continue
            x.append(perf)
            y.append(mean_d)
            labels.append(f'{mid} G{gi}')
            colors.append(cmap[mid])

        scatter[task_name] = dict(
            x=np.array(x), y=np.array(y), labels=labels,
            colors=colors, monkey_names=monkey_names,
        )
    return scatter
