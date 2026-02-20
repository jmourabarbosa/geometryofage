"""
KNN decoding and regression from distance matrices.
"""

import numpy as np
from collections import Counter
from scipy.stats import pearsonr


def knn_decode_monkey(dist, entries, k=3):
    """
    KNN decoding of monkey identity, leave-one-window-out CV.

    Parameters
    ----------
    dist : ndarray, shape (n, n)
    entries : list of dict, each with 'monkey' and 'group' keys.
    k : int

    Returns
    -------
    accuracy : float
    y_true : ndarray of str
    y_pred : ndarray of str
    """
    monkeys = np.array([e["monkey"] for e in entries])
    groups = np.array([e["group"] for e in entries])
    n = len(entries)

    y_true, y_pred = [], []

    for test_group in sorted(set(groups)):
        test_idx = np.where(groups == test_group)[0]
        train_idx = np.where(groups != test_group)[0]

        for ti in test_idx:
            d = dist[ti, train_idx]
            knn = train_idx[np.argsort(d)[:k]]
            pred = Counter(monkeys[knn]).most_common(1)[0][0]
            y_true.append(monkeys[ti])
            y_pred.append(pred)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)
    return accuracy, y_true, y_pred


def knn_decode_age(dist, entries, k=3):
    """
    KNN decoding of age group, leave-one-monkey-out CV.

    Uses KNN regression: predicts the mean group index of neighbors,
    then rounds to the nearest integer.

    Parameters
    ----------
    dist : ndarray, shape (n, n)
    entries : list of dict, each with 'monkey' and 'group' keys.
    k : int

    Returns
    -------
    results : dict with keys:
        'y_true', 'y_pred' (continuous), 'y_pred_round',
        'exact_acc', 'pm1_acc', 'pm2_acc'
    """
    monkeys = np.array([e["monkey"] for e in entries])
    groups = np.array([e["group"] for e in entries])
    monkey_names = sorted(set(monkeys))

    y_true, y_pred = [], []

    for test_mk in monkey_names:
        test_idx = np.where(monkeys == test_mk)[0]
        train_idx = np.where(monkeys != test_mk)[0]
        if len(test_idx) == 0:
            continue

        for ti in test_idx:
            d = dist[ti, train_idx]
            knn = train_idx[np.argsort(d)[:k]]
            pred = np.mean(groups[knn])
            y_true.append(groups[ti])
            y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_round = np.round(y_pred).astype(int)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_round": y_pred_round,
        "exact_acc": np.mean(y_pred_round == y_true),
        "pm1_acc": np.mean(np.abs(y_pred_round - y_true) <= 1),
        "pm2_acc": np.mean(np.abs(y_pred_round - y_true) <= 2),
    }


def regress_age(dist, entries, age_values, k=3):
    """
    KNN regression of relative age from Procrustes distances.
    Leave-one-monkey-out CV. Predicts age as mean of K nearest neighbors.

    Age values are normalized within each monkey (0 = youngest, 1 = oldest)
    before regression to remove between-monkey age differences.

    Parameters
    ----------
    dist : ndarray, shape (n, n)
    entries : list of dict, each with 'monkey' key.
    age_values : ndarray, shape (n,)
        Continuous age per entry (e.g., window center in days).
    k : int

    Returns
    -------
    results : dict with keys:
        'y_true', 'y_pred', 'monkey_ids',
        'r', 'p', 'mae'
    """
    monkeys = np.array([e["monkey"] for e in entries])
    monkey_names = sorted(set(monkeys))

    # Normalize age within each monkey to [0, 1]
    age_norm = np.zeros_like(age_values, dtype=float)
    for mid in monkey_names:
        mask = monkeys == mid
        lo, hi = age_values[mask].min(), age_values[mask].max()
        if hi > lo:
            age_norm[mask] = (age_values[mask] - lo) / (hi - lo)
        else:
            age_norm[mask] = 0.5

    y_true, y_pred, mk_ids = [], [], []

    for test_mk in monkey_names:
        test_idx = np.where(monkeys == test_mk)[0]
        train_idx = np.where(monkeys != test_mk)[0]
        if len(test_idx) == 0:
            continue

        for ti in test_idx:
            d = dist[ti, train_idx]
            knn = train_idx[np.argsort(d)[:k]]
            pred = np.mean(age_norm[knn])
            y_true.append(age_norm[ti])
            y_pred.append(pred)
            mk_ids.append(test_mk)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r, p = pearsonr(y_true, y_pred)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "monkey_ids": np.array(mk_ids),
        "r": r,
        "p": p,
        "mae": np.mean(np.abs(y_true - y_pred)),
    }
