"""
Statistical analyses for Procrustes distance comparisons.
"""

import numpy as np
import scipy.stats as sts

from .representations import extract_entry_arrays


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


def neuron_count_matrix(task_data, bins=25):
    """Neuron counts per monkey and age bin, for each task.

    Uses the union of all monkeys across tasks. Monkeys absent from a
    task get zero counts.

    Parameters
    ----------
    task_data : dict
        {task_name: dict(ids=ndarray, abs_age=ndarray, ...)}
    bins : int
        Number of age bins (shared across all tasks and monkeys).

    Returns
    -------
    results : dict
        {task_name: dict(counts=ndarray, monkeys=list, bin_edges=ndarray)}
        ``counts`` has shape (n_monkeys, n_bins).  Rows follow ``monkeys``.
    """
    all_monkeys = sorted({m for td in task_data.values() for m in set(td['ids'])})
    all_ages = np.concatenate([td['abs_age'] for td in task_data.values()])
    bin_edges = np.linspace(all_ages.min() - 1, all_ages.max() + 1, bins + 1)

    results = {}
    for name in task_data:
        ids = task_data[name]['ids']
        abs_age = task_data[name]['abs_age']
        counts = np.zeros((len(all_monkeys), bins), dtype=int)
        for i, mid in enumerate(all_monkeys):
            ages = abs_age[ids == mid]
            if len(ages) > 0:
                counts[i], _ = np.histogram(ages, bins=bin_edges)
        results[name] = dict(counts=counts, monkeys=all_monkeys, bin_edges=bin_edges)
    return results


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
                          group_labels, method='pearson'):
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
        slope, intercept, r_lr, p_lr, se = sts.linregress(all_g, all_d)
        if method == 'spearman':
            r, p = sts.spearmanr(all_g, all_d)
        else:
            r, p = r_lr, p_lr

        pooled_groups.append(all_g)
        pooled_dists.append(all_d)

        results[task_name] = dict(
            group_dists=group_dists, slope=slope, intercept=intercept,
            r=r, p=p, se=se, common=common,
            n_monkeys=len(monkeys), n_neurons=mask.sum(),
            n_total=len(ids), method=method,
        )

    pg = np.concatenate(pooled_groups)
    pooled_d = np.concatenate(pooled_dists)
    s_all, i_all, r_lr_all, p_lr_all, se_all = sts.linregress(pg, pooled_d)
    if method == 'spearman':
        r_all, p_all = sts.spearmanr(pg, pooled_d)
    else:
        r_all, p_all = r_lr_all, p_lr_all
    pooled = dict(slope=s_all, intercept=i_all, r=r_all, p=p_all, se=se_all,
                  method=method)

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


def build_epoch_representations(task_data, task_epochs, n_pcs, min_neurons,
                                 bin_ms=25, n_age_bins=3):
    """Compute per-epoch PCA representations for each task.

    Parameters
    ----------
    task_data : dict
        {task_name: dict(data, ids, abs_age)}
    task_epochs : dict
        {task_name: dict(t_range, epochs)} — time range and epoch windows per task.
    n_pcs, min_neurons : int
    bin_ms : int
    n_age_bins : int
        Number of per-monkey quantile bins.

    Returns
    -------
    epoch_reps : dict
        {task: {epoch_name: {(monkey, group): entry}}}
    age_groups : dict
        {task: ndarray of int}
    monkey_edges : dict
        {(task, monkey): tuple of edges}
    """
    from .psth import compute_single_trial_rates, compute_tuning_curves
    from .representations import build_representations

    epoch_reps = {}
    age_groups = {}
    monkey_edges = {}

    for task_name, cfg in task_epochs.items():
        if task_name not in task_data:
            continue
        data = task_data[task_name]
        ids = data['ids']
        abs_age = data['abs_age']

        ag, mk_edges = assign_per_monkey_age_groups(ids, abs_age, n_age_bins)
        age_groups[task_name] = ag
        for mid, edges in mk_edges.items():
            monkey_edges[(task_name, mid)] = edges

        bins = np.arange(cfg['t_range'][0], cfg['t_range'][1] + bin_ms, bin_ms)
        bc = (bins[:-1] + bins[1:]) / 2.0
        rates = compute_single_trial_rates(data['data'], bins)
        tuning, enames = compute_tuning_curves(rates, bc, cfg['epochs'])

        epoch_reps[task_name] = {}
        for ei, ename in enumerate(enames):
            entries = build_representations(tuning[:, :, ei], ids, ag,
                                            n_pcs=n_pcs, min_neurons=min_neurons)
            epoch_reps[task_name][ename] = {(e['monkey'], e['group']): e
                                            for e in entries}

    return epoch_reps, age_groups, monkey_edges


def cross_epoch_distances(epoch_reps, comparisons):
    """Compute Procrustes distances between epoch representations.

    Parameters
    ----------
    epoch_reps : dict
        {task: {epoch_name: {(monkey, group): entry}}}
    comparisons : list of tuple
        Pairs of epoch names, e.g. [('cue', 'delay'), ('delay', 'response')].

    Returns
    -------
    cross_epoch : dict
        {task: {label: [dict(monkey, group, distance), ...]}}
    """
    from .procrustes import procrustes_disparity

    cross_epoch = {}
    for task_name in epoch_reps:
        cross_epoch[task_name] = {}
        for ea, eb in comparisons:
            if ea not in epoch_reps[task_name] or eb not in epoch_reps[task_name]:
                continue
            reps_a = epoch_reps[task_name][ea]
            reps_b = epoch_reps[task_name][eb]
            common = sorted(set(reps_a) & set(reps_b))

            rows = []
            for key in common:
                A = reps_a[key]['matrix'].T
                B = reps_b[key]['matrix'].T
                d = procrustes_disparity(A, B)
                rows.append(dict(monkey=key[0], group=key[1], distance=d))

            label = f'{ea}\u2192{eb}'
            cross_epoch[task_name][label] = rows

    return cross_epoch


# ── Cross-task geometry comparison ────────────────────────────────────────────

CAT_NAMES = [
    'Same monkey\nwithin-task',
    'Same monkey\ncross-task',
    'Diff monkey\nwithin-task',
    'Diff monkey\ncross-task',
]


def _assign_splits(ids, rng):
    """Random balanced 50/50 split within each monkey."""
    splits = np.zeros(len(ids), dtype=int)
    for mid in sorted(set(ids)):
        idx = np.where(ids == mid)[0]
        perm = rng.permutation(len(idx))
        splits[idx[perm[len(idx) // 2:]]] = 1
    return splits


def cross_task_cv(tuning_flat, task_ids, n_pcs, min_neurons,
                  n_iter=100, seed=42, verbose=True, method='pearson'):
    """Split-half cross-validation loop for cross-task geometry comparison.

    Parameters
    ----------
    tuning_flat : dict
        {task_name: ndarray (n_neurons, n_features)}
    task_ids : dict
        {task_name: ndarray of monkey IDs}
    n_pcs : int
    min_neurons : int
    n_iter : int
    seed : int

    Returns
    -------
    dict with keys: cat_means, mantel_r, last_dist, last_labels, cat_names, n_iter
    """
    from itertools import combinations
    from .representations import build_representations
    from .procrustes import procrustes_distance_matrix

    task_names = list(task_ids.keys())
    task_pairs = list(combinations(range(len(task_names)), 2))
    rng = np.random.default_rng(seed)

    iter_cat_means = {c: [] for c in CAT_NAMES}
    iter_mantel_r = {(task_names[i], task_names[j]): []
                     for i, j in task_pairs}
    last_dist = None
    last_labels = None

    for it in range(n_iter):
        all_entries = []
        entry_meta = []  # (monkey, task, split)

        for tname, ids in task_ids.items():
            sp = _assign_splits(ids, rng)
            for s in [0, 1]:
                smask = sp == s
                groups = np.zeros(smask.sum(), dtype=int)
                entries = build_representations(
                    tuning_flat[tname][smask], ids[smask], groups,
                    n_pcs=n_pcs, min_neurons=min_neurons, zscore=True)
                for e in entries:
                    all_entries.append(e)
                    entry_meta.append((e['monkey'], tname, s))

        dist = procrustes_distance_matrix(all_entries)
        n = len(all_entries)

        # Categorise pairs
        cats = {c: [] for c in CAT_NAMES}
        for i in range(n):
            mi, ti, si = entry_meta[i]
            for j in range(i + 1, n):
                mj, tj, sj = entry_meta[j]
                d = dist[i, j]
                same_mk = mi == mj
                same_task = ti == tj
                if same_mk and same_task:
                    cats[CAT_NAMES[0]].append(d)
                elif same_mk and not same_task:
                    cats[CAT_NAMES[1]].append(d)
                elif not same_mk and same_task:
                    cats[CAT_NAMES[2]].append(d)
                else:
                    cats[CAT_NAMES[3]].append(d)

        for c in CAT_NAMES:
            iter_cat_means[c].append(np.nanmean(cats[c]))

        # Mantel: correlate within-task distances (split 0) across task pairs
        split0 = {}
        for tname in task_names:
            idx = [k for k, m in enumerate(entry_meta)
                   if m[1] == tname and m[2] == 0]
            monkeys = [entry_meta[k][0] for k in idx]
            sub = dist[np.ix_(idx, idx)]
            split0[tname] = (monkeys, sub)

        for ti, tj in task_pairs:
            ta, tb = task_names[ti], task_names[tj]
            mk_a, d_a = split0[ta]
            mk_b, d_b = split0[tb]
            common = sorted(set(mk_a) & set(mk_b))
            if len(common) >= 3:
                ia = [mk_a.index(m) for m in common]
                ib = [mk_b.index(m) for m in common]
                va = [d_a[ia[x], ia[y]]
                      for x in range(len(common))
                      for y in range(x + 1, len(common))]
                vb = [d_b[ib[x], ib[y]]
                      for x in range(len(common))
                      for y in range(x + 1, len(common))]
                if method == 'spearman':
                    r, _ = sts.spearmanr(va, vb)
                else:
                    r, _ = sts.pearsonr(va, vb)
                iter_mantel_r[(ta, tb)].append(r)

        last_dist = dist
        last_labels = [f"{m[0]}_{m[1].split()[-1]}_{chr(65 + m[2])}"
                       for m in entry_meta]

        if verbose and (it + 1) % 25 == 0:
            print(f'  iteration {it + 1}/{n_iter}')

    if verbose:
        print('Done.')
    return dict(
        cat_means={c: np.array(v) for c, v in iter_cat_means.items()},
        mantel_r={k: np.array(v) for k, v in iter_mantel_r.items()},
        last_dist=last_dist,
        last_labels=last_labels,
        cat_names=CAT_NAMES,
        n_iter=n_iter,
    )
