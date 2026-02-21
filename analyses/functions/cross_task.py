"""
Cross-task geometry comparison via split-half cross-validation.
"""

import numpy as np
from scipy import stats

from .representations import build_representations
from .procrustes import procrustes_distance_matrix

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
                  n_iter=100, seed=42):
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
    task_names = list(task_ids.keys())
    rng = np.random.default_rng(seed)

    iter_cat_means = {c: [] for c in CAT_NAMES}
    iter_mantel_r = []
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
            iter_cat_means[c].append(np.mean(cats[c]))

        # Mantel: correlate within-task distances (split 0) across tasks
        split0 = {}
        for tname in task_names:
            idx = [k for k, m in enumerate(entry_meta)
                   if m[1] == tname and m[2] == 0]
            monkeys = [entry_meta[k][0] for k in idx]
            sub = dist[np.ix_(idx, idx)]
            split0[tname] = (monkeys, sub)

        mk_a, d_a = split0[task_names[0]]
        mk_b, d_b = split0[task_names[1]]
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
            r, _ = stats.pearsonr(va, vb)
            iter_mantel_r.append(r)

        last_dist = dist
        last_labels = [f"{m[0]}_{m[1].split()[1]}_{chr(65 + m[2])}"
                       for m in entry_meta]

        if (it + 1) % 25 == 0:
            print(f'  iteration {it + 1}/{n_iter}')

    print('Done.')
    return dict(
        cat_means={c: np.array(v) for c, v in iter_cat_means.items()},
        mantel_r=np.array(iter_mantel_r),
        last_dist=last_dist,
        last_labels=last_labels,
        cat_names=CAT_NAMES,
        n_iter=n_iter,
    )
