"""
Tests for analyses/functions/ using synthetic data.
No real .mat files required.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.representations import zscore_neurons, pca_reduce, build_representations, build_window_entries
from functions.procrustes import procrustes_distance_matrix
from functions.analysis import (assign_age_groups, cross_monkey_analysis, cross_age_analysis,
                                extract_entry_arrays, _mean_within_monkey)
from functions.decoding import knn_decode_monkey, knn_decode_age, regress_age, _knn_loo_predict
from functions.temporal import rates_to_psth, temporal_cross_monkey, temporal_cross_age, temporal_cross_age_by_pair
from functions.plotting import (_baseline_normalize, plot_cross_monkey, plot_distance_matrices,
                                plot_cross_age, plot_temporal, plot_temporal_by_pair,
                                plot_age_decoding, plot_correlation_panels)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_entries(n_monkeys=3, n_groups=2, n_pcs=4, n_features=8, seed=0):
    """Create synthetic entries for Procrustes tests."""
    rng = np.random.default_rng(seed)
    entries = []
    monkeys = [f"M{i}" for i in range(n_monkeys)]
    for g in range(n_groups):
        for mid in monkeys:
            entries.append({
                "monkey": mid,
                "group": g,
                "matrix": rng.standard_normal((n_pcs, n_features)),
                "n_neurons": 50,
                "var_explained": 0.6,
            })
    return entries


def _make_tuning(n_neurons=200, n_features=16, seed=0):
    """Synthetic tuning matrix with a few bad rows."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_neurons, n_features))
    # Add some zero-variance and NaN rows
    X[0] = 5.0
    X[1] = np.nan
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# representations.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestZscoreNeurons:
    def test_removes_nan_rows(self):
        X = np.array([[1, 2, 3], [np.nan, 1, 2], [4, 5, 6]], dtype=float)
        out = zscore_neurons(X)
        assert out.shape[0] == 2

    def test_removes_zero_variance(self):
        X = np.array([[1, 2, 3], [5, 5, 5], [4, 5, 6]], dtype=float)
        out = zscore_neurons(X)
        assert out.shape[0] == 2

    def test_output_is_zscored(self):
        X = np.array([[10, 20, 30], [100, 200, 300]], dtype=float)
        out = zscore_neurons(X)
        np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=1), 1, atol=1e-10)

    def test_empty_after_cleaning(self):
        X = np.array([[5, 5, 5], [np.nan, 1, 2]], dtype=float)
        out = zscore_neurons(X)
        assert out.shape[0] == 0


class TestPcaReduce:
    def test_output_shape(self):
        X = np.random.default_rng(0).standard_normal((50, 8))
        proj, var = pca_reduce(X, 4)
        assert proj.shape == (4, 8)
        assert 0 < var <= 1

    def test_fewer_neurons_than_pcs(self):
        X = np.random.default_rng(0).standard_normal((3, 8))
        proj, var = pca_reduce(X, 8)
        assert proj.shape == (8, 8)
        # Padded rows should be zeros
        np.testing.assert_array_equal(proj[3:], 0)

    def test_fewer_features_than_pcs(self):
        X = np.random.default_rng(0).standard_normal((50, 3))
        proj, var = pca_reduce(X, 8)
        assert proj.shape == (8, 3)

    def test_variance_explained_range(self):
        X = np.random.default_rng(0).standard_normal((100, 10))
        _, var = pca_reduce(X, 10)
        np.testing.assert_allclose(var, 1.0, atol=1e-10)


class TestBuildRepresentations:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 120
        tuning = rng.standard_normal((n, 16))
        ids = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
        groups = np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20)
        entries = build_representations(tuning, ids, groups, n_pcs=4, min_neurons=10)
        assert len(entries) == 6  # 3 monkeys x 2 groups
        for e in entries:
            assert e["matrix"].shape == (4, 16)
            assert e["n_neurons"] >= 10

    def test_skips_small_groups(self):
        rng = np.random.default_rng(42)
        tuning = rng.standard_normal((25, 8))
        ids = np.array(["A"] * 20 + ["B"] * 5)
        groups = np.array([0] * 10 + [1] * 10 + [0] * 5)
        entries = build_representations(tuning, ids, groups, n_pcs=4, min_neurons=10)
        # B group 0 has only 5 neurons -> skipped
        monkeys_in = [e["monkey"] for e in entries]
        assert "B" not in monkeys_in

    def test_no_zscore(self):
        rng = np.random.default_rng(42)
        tuning = rng.standard_normal((40, 8))
        ids = np.array(["A"] * 40)
        groups = np.array([0] * 40)
        entries = build_representations(tuning, ids, groups, n_pcs=4, zscore=False)
        assert len(entries) == 1


class TestBuildWindowEntries:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        tuning = rng.standard_normal((n, 16))
        ids = np.array(["A"] * 100 + ["B"] * 100)
        ages = np.concatenate([np.linspace(100, 500, 100), np.linspace(100, 500, 100)])
        entries = build_window_entries(tuning, ids, ages, n_pcs=4, n_windows=5, min_neurons=10)
        assert len(entries) > 0
        assert all("center_days" in e for e in entries)
        # Each monkey should contribute windows
        mk_set = set(e["monkey"] for e in entries)
        assert mk_set == {"A", "B"}

    def test_too_few_neurons(self):
        rng = np.random.default_rng(42)
        tuning = rng.standard_normal((5, 8))
        ids = np.array(["A"] * 5)
        ages = np.linspace(100, 200, 5)
        entries = build_window_entries(tuning, ids, ages, n_pcs=4, n_windows=3, min_neurons=10)
        assert len(entries) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# procrustes.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcrustesDistanceMatrix:
    def test_symmetric(self):
        entries = _make_entries(n_monkeys=3, n_groups=1)
        dist = procrustes_distance_matrix(entries)
        np.testing.assert_array_equal(dist, dist.T)

    def test_diagonal_zero(self):
        entries = _make_entries(n_monkeys=3, n_groups=1)
        dist = procrustes_distance_matrix(entries)
        np.testing.assert_array_equal(np.diag(dist), 0)

    def test_nonnegative(self):
        entries = _make_entries(n_monkeys=4, n_groups=2)
        dist = procrustes_distance_matrix(entries)
        assert np.all(dist >= 0)

    def test_self_distance_zero(self):
        entry = _make_entries(n_monkeys=1, n_groups=1)
        # Duplicate the same entry
        entries = [entry[0], {"monkey": "M0", "group": 0,
                              "matrix": entry[0]["matrix"].copy(),
                              "n_neurons": 50, "var_explained": 0.6}]
        dist = procrustes_distance_matrix(entries)
        np.testing.assert_allclose(dist[0, 1], 0, atol=1e-10)

    def test_shape(self):
        entries = _make_entries(n_monkeys=3, n_groups=2)
        dist = procrustes_distance_matrix(entries)
        assert dist.shape == (6, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# analysis.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssignAgeGroups:
    def test_correct_n_groups(self):
        ids = np.array(["A"] * 30 + ["B"] * 30)
        ages = np.concatenate([np.linspace(100, 300, 30), np.linspace(100, 300, 30)])
        groups = assign_age_groups(ids, ages, 3)
        assert set(groups) == {0, 1, 2}

    def test_per_monkey_independent(self):
        ids = np.array(["A"] * 20 + ["B"] * 20)
        ages_A = np.linspace(100, 200, 20)
        ages_B = np.linspace(500, 600, 20)  # Different age range
        ages = np.concatenate([ages_A, ages_B])
        groups = assign_age_groups(ids, ages, 2)
        # Both monkeys should have both groups even though age ranges differ
        assert set(groups[:20]) == {0, 1}
        assert set(groups[20:]) == {0, 1}

    def test_single_group(self):
        ids = np.array(["A"] * 10)
        ages = np.linspace(100, 200, 10)
        groups = assign_age_groups(ids, ages, 1)
        assert np.all(groups == 0)


class TestCrossMonkeyAnalysis:
    def test_output_keys(self):
        entries = _make_entries(n_monkeys=3, n_groups=2)
        dist = procrustes_distance_matrix(entries)
        result = cross_monkey_analysis(entries, dist)
        expected_keys = {"cross_raw", "cross_adj", "mean_within",
                         "within_all_pairs", "t_stat", "p_val", "monkey_names"}
        assert set(result.keys()) == expected_keys

    def test_cross_distances_are_positive(self):
        entries = _make_entries(n_monkeys=3, n_groups=2)
        dist = procrustes_distance_matrix(entries)
        result = cross_monkey_analysis(entries, dist)
        assert len(result["cross_raw"]) > 0
        assert np.all(result["cross_raw"] > 0)

    def test_within_distances_exist(self):
        entries = _make_entries(n_monkeys=3, n_groups=2)
        dist = procrustes_distance_matrix(entries)
        result = cross_monkey_analysis(entries, dist)
        assert len(result["within_all_pairs"]) > 0


class TestCrossAgeAnalysis:
    def test_output_keys(self):
        entries = _make_entries(n_monkeys=3, n_groups=2)
        dist = procrustes_distance_matrix(entries)
        result = cross_age_analysis(entries, dist)
        expected_keys = {"same_age_raw", "same_age_adj", "diff_age_raw",
                         "mean_across_age", "t_stat", "p_val", "monkey_names"}
        assert set(result.keys()) == expected_keys

    def test_same_age_pairs_exist(self):
        entries = _make_entries(n_monkeys=3, n_groups=2)
        dist = procrustes_distance_matrix(entries)
        result = cross_age_analysis(entries, dist)
        assert len(result["same_age_raw"]) > 0
        assert len(result["diff_age_raw"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# decoding.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnnDecodeMonkey:
    def test_output_types(self):
        entries = _make_entries(n_monkeys=3, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        acc, y_true, y_pred = knn_decode_monkey(dist, entries, k=1)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        assert len(y_true) == len(y_pred)

    def test_perfect_decoding_with_zero_within(self):
        """If within-monkey distances are 0 and cross are large, accuracy should be high."""
        rng = np.random.default_rng(99)
        entries = []
        for g in range(3):
            for mi, mid in enumerate(["A", "B"]):
                # Same base matrix per monkey, different across monkeys
                base = rng.standard_normal((4, 8)) if g == 0 else entries[mi]["matrix"]
                entries.append({"monkey": mid, "group": g,
                                "matrix": base.copy(), "n_neurons": 50, "var_explained": 0.5})
        dist = procrustes_distance_matrix(entries)
        acc, _, _ = knn_decode_monkey(dist, entries, k=1)
        assert acc == 1.0


class TestKnnDecodeAge:
    def test_output_keys(self):
        entries = _make_entries(n_monkeys=3, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        result = knn_decode_age(dist, entries, k=1)
        expected_keys = {"y_true", "y_pred", "y_pred_round",
                         "exact_acc", "pm1_acc", "pm2_acc"}
        assert set(result.keys()) == expected_keys

    def test_accuracy_bounds(self):
        entries = _make_entries(n_monkeys=3, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        result = knn_decode_age(dist, entries, k=1)
        assert 0 <= result["exact_acc"] <= 1
        assert result["pm1_acc"] >= result["exact_acc"]
        assert result["pm2_acc"] >= result["pm1_acc"]


class TestRegressAge:
    def test_output_keys(self):
        entries = _make_entries(n_monkeys=3, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        ages = np.array([e["group"] * 100 + 50 for e in entries], dtype=float)
        result = regress_age(dist, entries, ages, k=2)
        expected_keys = {"y_true", "y_pred", "monkey_ids", "r", "p", "mae"}
        assert set(result.keys()) == expected_keys

    def test_age_normalization(self):
        entries = _make_entries(n_monkeys=2, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        ages = np.array([e["group"] * 100 + 50 for e in entries], dtype=float)
        result = regress_age(dist, entries, ages, k=2)
        # y_true should be in [0, 1] since normalized within each monkey
        assert np.all(result["y_true"] >= 0)
        assert np.all(result["y_true"] <= 1)


# ═══════════════════════════════════════════════════════════════════════════════
# psth.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeTuningCurves:
    def test_basic(self):
        from functions.psth import compute_tuning_curves
        n_neurons, n_conds, n_bins = 5, 4, 20
        rng = np.random.default_rng(0)
        rates = []
        for i in range(n_neurons):
            neuron_rates = []
            for c in range(n_conds):
                neuron_rates.append(rng.standard_normal((10, n_bins)))
            rates.append(neuron_rates)
        bc = np.linspace(-500, 2000, n_bins)
        epochs = {"cue": (0, 500), "delay": (500, 2000)}
        tuning, epoch_names = compute_tuning_curves(rates, bc, epochs)
        assert tuning.shape == (n_neurons, n_conds, 2)
        assert epoch_names == ["cue", "delay"]

    def test_empty_trials(self):
        from functions.psth import compute_tuning_curves
        rates = [[np.empty((0, 10)), np.random.default_rng(0).standard_normal((5, 10))]]
        bc = np.linspace(-500, 2000, 10)
        epochs = {"cue": (0, 500)}
        tuning, _ = compute_tuning_curves(rates, bc, epochs)
        assert np.isnan(tuning[0, 0, 0])
        assert np.isfinite(tuning[0, 1, 0])


# ═══════════════════════════════════════════════════════════════════════════════
# temporal.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestRatesToPsth:
    def test_shape(self):
        rng = np.random.default_rng(0)
        rates = []
        for i in range(10):
            neuron_rates = []
            for c in range(4):
                neuron_rates.append(rng.standard_normal((8, 20)))
            rates.append(neuron_rates)
        psth = rates_to_psth(rates)
        assert psth.shape == (10, 4, 20)

    def test_nan_for_empty_trials(self):
        rates = [[np.empty((0, 5)), np.random.default_rng(0).standard_normal((3, 5))]]
        psth = rates_to_psth(rates)
        assert np.all(np.isnan(psth[0, 0]))
        assert np.all(np.isfinite(psth[0, 1]))


class TestTemporalCrossMonkey:
    def test_basic(self):
        """Smoke test with synthetic PSTH data."""
        rng = np.random.default_rng(42)
        n_neurons = 100
        n_conds = 8
        n_bins = 40
        psth = rng.standard_normal((n_neurons, n_conds, n_bins))
        bc = np.linspace(-500, 2500, n_bins)
        ids = np.array(["A"] * 35 + ["B"] * 35 + ["C"] * 30)
        age_group = np.array([0] * 35 + [0] * 35 + [0] * 30)
        t, boots = temporal_cross_monkey(psth, bc, ids, age_group,
                                          n_pcs=4, min_neurons=10,
                                          window_ms=500, step_ms=200, n_boot=10)
        assert len(t) > 0
        assert boots.shape[0] == 10
        assert boots.shape[1] == len(t)


class TestTemporalCrossAge:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n_neurons = 120
        n_conds = 8
        n_bins = 40
        psth = rng.standard_normal((n_neurons, n_conds, n_bins))
        bc = np.linspace(-500, 2500, n_bins)
        ids = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
        age_group = np.array([0] * 20 + [1] * 20 +
                             [0] * 20 + [1] * 20 +
                             [0] * 20 + [1] * 20)
        t, boots = temporal_cross_age(psth, bc, ids, age_group,
                                       n_pcs=4, min_neurons=10,
                                       window_ms=500, step_ms=200, n_boot=10)
        assert len(t) > 0
        assert boots.shape[0] == 10


class TestTemporalCrossAgeByPair:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n_neurons = 120
        n_conds = 8
        n_bins = 40
        psth = rng.standard_normal((n_neurons, n_conds, n_bins))
        bc = np.linspace(-500, 2500, n_bins)
        ids = np.array(["A"] * 60 + ["B"] * 60)
        age_group = np.array([0] * 20 + [1] * 20 + [2] * 20 +
                             [0] * 20 + [1] * 20 + [2] * 20)
        age_pairs = [(0, 1), (1, 2), (0, 2)]
        t, boots_dict = temporal_cross_age_by_pair(
            psth, bc, ids, age_group, age_pairs,
            n_pcs=4, min_neurons=10,
            window_ms=500, step_ms=200, n_boot=10)
        assert len(t) > 0
        assert isinstance(boots_dict, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# plotting.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaselineNormalize:
    def test_subtracts_pre_cue_mean(self):
        boots = np.array([[10, 10, 20, 30],
                          [5, 5, 15, 25]], dtype=float)
        t = np.array([-200, -100, 100, 200])
        result = _baseline_normalize(boots, t)
        # Pre-cue mean for row 0 = 10, row 1 = 5
        np.testing.assert_allclose(result[0], [0, 0, 10, 20])
        np.testing.assert_allclose(result[1], [0, 0, 10, 20])

    def test_no_pre_cue(self):
        boots = np.array([[10, 20, 30]], dtype=float)
        t = np.array([100, 200, 300])
        result = _baseline_normalize(boots, t)
        # No t < 0, so mean of empty = NaN -> result is NaN
        assert np.all(np.isnan(result))


def _two_task_entries():
    """Build entries and dist for two tasks (needed for 2D axes in plots)."""
    e1 = _make_entries(n_monkeys=3, n_groups=2, seed=0)
    e2 = _make_entries(n_monkeys=3, n_groups=2, seed=1)
    d1 = procrustes_distance_matrix(e1)
    d2 = procrustes_distance_matrix(e2)
    return e1, d1, e2, d2


class TestPlotCrossMonkey:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        e1, d1, e2, d2 = _two_task_entries()
        cm1 = cross_monkey_analysis(e1, d1)
        cm2 = cross_monkey_analysis(e2, d2)
        results = {"ODR 1.5s": {"cross_monkey": cm1},
                   "ODR 3.0s": {"cross_monkey": cm2}}
        plot_cross_monkey(results)
        plt.close("all")


class TestPlotDistanceMatrices:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        e1, d1, e2, d2 = _two_task_entries()
        l1 = [f"{e['monkey']}_G{e['group']}" for e in e1]
        l2 = [f"{e['monkey']}_G{e['group']}" for e in e2]
        results = {"ODR 1.5s": {"dist": d1, "labels": l1},
                   "ODR 3.0s": {"dist": d2, "labels": l2}}
        plot_distance_matrices(results)
        plt.close("all")


class TestPlotCrossAge:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        e1, d1, e2, d2 = _two_task_entries()
        ca1 = cross_age_analysis(e1, d1)
        ca2 = cross_age_analysis(e2, d2)
        results = {"ODR 1.5s": {"cross_age": ca1},
                   "ODR 3.0s": {"cross_age": ca2}}
        plot_cross_age(results)
        plt.close("all")


class TestPlotTemporal:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        t = np.linspace(-500, 2500, 30)
        boots = rng.standard_normal((10, 30))
        temporal_results = {"ODR 1.5s": {"t": t, "boots": boots}}
        plot_temporal(temporal_results)
        plt.close("all")


class TestPlotTemporalByPair:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        t = np.linspace(-500, 2500, 30)
        boots_by_pair = {
            (0, 1): rng.standard_normal((10, 30)),
            (1, 2): rng.standard_normal((10, 30)),
            (0, 2): rng.standard_normal((10, 30)),
        }
        temporal_pair_results = {
            "ODR 1.5s": {"t": t, "boots_by_pair": boots_by_pair}
        }
        plot_temporal_by_pair(temporal_pair_results)
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════════
# New helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractEntryArrays:
    def test_basic(self):
        entries = _make_entries(n_monkeys=3, n_groups=2)
        monkeys, groups = extract_entry_arrays(entries)
        assert monkeys.shape == (6,)
        assert groups.shape == (6,)
        assert set(monkeys) == {"M0", "M1", "M2"}
        assert set(groups) == {0, 1}

    def test_order_preserved(self):
        entries = [{"monkey": "A", "group": 0}, {"monkey": "B", "group": 1}]
        monkeys, groups = extract_entry_arrays(entries)
        assert monkeys[0] == "A"
        assert groups[1] == 1


class TestMeanWithinMonkey:
    def test_basic(self):
        entries = _make_entries(n_monkeys=2, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        me, _ = extract_entry_arrays(entries)
        mean_w = _mean_within_monkey(me, dist)
        assert set(mean_w.keys()) == {"M0", "M1"}
        # Each monkey has 3 entries => 3 pairwise distances
        assert all(v > 0 for v in mean_w.values())

    def test_single_entry_monkey(self):
        entries = _make_entries(n_monkeys=2, n_groups=1)
        dist = procrustes_distance_matrix(entries)
        me = np.array(["M0", "M1"])
        mean_w = _mean_within_monkey(me, dist)
        assert np.isnan(mean_w["M0"])
        assert np.isnan(mean_w["M1"])


class TestKnnLooPredict:
    def test_classification(self):
        from collections import Counter
        entries = _make_entries(n_monkeys=3, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        monkeys, groups = extract_entry_arrays(entries)
        agg = lambda x: Counter(x).most_common(1)[0][0]
        y_true, y_pred, split_ids = _knn_loo_predict(dist, monkeys, groups, k=1, agg_fn=agg)
        assert len(y_true) == len(y_pred) == len(split_ids)
        assert len(y_true) == len(entries)

    def test_regression(self):
        entries = _make_entries(n_monkeys=3, n_groups=3)
        dist = procrustes_distance_matrix(entries)
        monkeys, groups = extract_entry_arrays(entries)
        y_true, y_pred, split_ids = _knn_loo_predict(
            dist, groups.astype(float), monkeys, k=1, agg_fn=np.mean)
        assert len(y_true) == len(entries)
        assert all(isinstance(v, (float, np.floating)) for v in y_pred)


class TestPlotAgeDecoding:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        e1, d1, e2, d2 = _two_task_entries()
        results = {
            "ODR 1.5s": {"entries": e1, "dist": d1},
            "ODR 3.0s": {"entries": e2, "dist": d2},
        }
        plot_age_decoding(results, k=1)
        plt.close("all")


class TestPlotCorrelationPanels:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        scatter = {
            "ODR 1.5s": dict(x=np.array([1, 2, 3, 4.0]),
                             y=np.array([0.1, 0.2, 0.3, 0.4]),
                             labels=["a", "b", "c", "d"]),
            "ODR 3.0s": dict(x=np.array([5, 6, 7, 8.0]),
                             y=np.array([0.5, 0.6, 0.7, 0.8]),
                             labels=["e", "f", "g", "h"]),
        }
        plot_correlation_panels(scatter, "x label", "y label", suptitle="test")
        plt.close("all")
