"""
Temporal sliding-window Procrustes analysis with bootstrap.
"""

import numpy as np
import warnings

from .representations import build_representations
from .procrustes import procrustes_distance_matrix
from .representations import extract_entry_arrays


def _sliding_procrustes(psth, bc, ids, age_group, pair_fn,
                        n_pcs, min_neurons, window_ms, step_ms, n_boot,
                        verbose=True):
    """Core sliding window loop: build representations, compute Procrustes,
    bootstrap selected pairs at each time step.

    pair_fn(dist, me, ge, n) -> dict of {key: np.array of distances}
    """
    rng = np.random.default_rng(42)
    window_starts = np.arange(bc[0], bc[-1] - window_ms + step_ms + 1, step_ms)
    window_centers = window_starts + window_ms / 2
    n_t = len(window_centers)

    boots_dict = {}

    for ti, t0 in enumerate(window_starts):
        bmask = (bc >= t0) & (bc < t0 + window_ms)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tuning_t = np.nanmean(psth[:, :, bmask], axis=2)

        entries_t = build_representations(
            tuning_t, ids, age_group,
            n_pcs=n_pcs, min_neurons=min_neurons, zscore=True
        )
        if len(entries_t) < 3:
            if verbose:
                print(f'Time {window_centers[ti]:.1f} ms: not enough entries ({len(entries_t)})')
            continue

        dist_t = procrustes_distance_matrix(entries_t)
        me, ge = extract_entry_arrays(entries_t)
        n = len(entries_t)

        pairs = pair_fn(dist_t, me, ge, n)
        for key, dists in pairs.items():
            if key not in boots_dict:
                boots_dict[key] = np.full((n_boot, n_t), np.nan)
            if len(dists) == 0:
                continue
            for b in range(n_boot):
                boots_dict[key][b, ti] = np.nanmean(rng.choice(dists, len(dists), replace=True))

    return window_centers, boots_dict


def temporal_cross_monkey(psth, bc, ids, common_ids=None,
                          n_pcs=8, min_neurons=9,
                          window_ms=500, step_ms=50, n_boot=1000):
    """Cross-monkey Procrustes distance in sliding windows with bootstrap.

    All neurons per monkey are pooled (no age grouping).

    Parameters
    ----------
    common_ids : list of str, optional
        If given, restrict to these monkeys only.
    """
    if common_ids is not None:
        mask = np.isin(ids, common_ids)
        psth, ids = psth[mask], ids[mask]

    single_group = np.zeros(len(ids), dtype=int)

    def pair_fn(dist, me, ge, n):
        cross = np.array([dist[i, j]
                          for i in range(n) for j in range(i + 1, n)
                          if me[i] != me[j]])
        return {'_': cross}

    t, boots = _sliding_procrustes(
        psth, bc, ids, single_group, pair_fn,
        n_pcs, min_neurons, window_ms, step_ms, n_boot
    )
    return t, boots.get('_', np.full((n_boot, len(t)), np.nan))


def temporal_cross_age(psth, bc, ids, age_group,
                       n_pcs=8, min_neurons=9,
                       window_ms=500, step_ms=50, n_boot=1000):
    """Within-monkey cross-age Procrustes in sliding windows with bootstrap."""
    def pair_fn(dist, me, ge, n):
        ca = np.array([dist[i, j]
                       for i in range(n) for j in range(i + 1, n)
                       if me[i] == me[j] and ge[i] != ge[j]])
        return {'_': ca}

    t, boots = _sliding_procrustes(
        psth, bc, ids, age_group, pair_fn,
        n_pcs, min_neurons, window_ms, step_ms, n_boot
    )
    return t, boots.get('_', np.full((n_boot, len(t)), np.nan))


