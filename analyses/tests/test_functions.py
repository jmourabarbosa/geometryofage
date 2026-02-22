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

from functions.representations import (zscore_neurons, pca_reduce, build_representations,
                                       _clean_neurons, _build_entry)
from functions.procrustes import procrustes_distance_matrix
from functions.analysis import (assign_age_groups, assign_per_monkey_age_groups,
                                cross_monkey_analysis, cross_age_analysis,
                                _mean_within_monkey)
from functions.representations import extract_entry_arrays
from functions.psth import rates_to_psth
from functions.temporal import temporal_cross_monkey, temporal_cross_age
from functions.plotting import (_baseline_normalize, plot_cross_monkey, plot_distance_matrices,
                                plot_cross_age, plot_temporal,
                                plot_cross_task, print_cross_task_summary,
                                plot_cross_monkey_by_group, print_cross_monkey_by_group_summary,
                                plot_3d_representation, wall_projections,
                                plot_3d_grid, plot_within_monkey_alignment,
                                plot_global_alignment, plot_cross_epoch_correlations,
                                TASK_COLORS, STIM_COLORS, STIM_LABELS,
                                AGE_COLORS, AGE_GROUP_LABELS)
from functions.load_data import (CARDINAL_COLS, TASK_EPOCHS,
                                 filter_common_monkeys, load_cardinal_task_data)
from functions.analysis import (build_epoch_representations, cross_epoch_distances)
from functions.representations import tuning_to_matrix


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
    def test_three_groups(self):
        ages = np.array([-10, -5, 0, 5, 10], dtype=float)
        groups = assign_age_groups(ages, (-3, 3))
        # 0 is in 'during' (bin 1), 5 is >= 3 so 'after' (bin 2)
        np.testing.assert_array_equal(groups, [0, 0, 1, 2, 2])

    def test_two_groups(self):
        ages = np.array([-10, -1, 0, 5], dtype=float)
        groups = assign_age_groups(ages, (0,))
        np.testing.assert_array_equal(groups, [0, 0, 1, 1])

    def test_missing_group_ok(self):
        # All values above both edges -> all in group 2
        ages = np.array([10, 20, 30], dtype=float)
        groups = assign_age_groups(ages, (-6, 6))
        np.testing.assert_array_equal(groups, [2, 2, 2])


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
        t, boots = temporal_cross_monkey(psth, bc, ids,
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



class TestComputeFlatTuning:
    def test_output_shape(self):
        from unittest.mock import patch
        from functions.psth import compute_flat_tuning

        rng = np.random.default_rng(0)
        n_neurons, n_conds, n_epochs = 5, 4, 2
        mock_rates = [[rng.standard_normal((10, 50)) for _ in range(n_conds)]
                       for _ in range(n_neurons)]
        mock_tuning = rng.standard_normal((n_neurons, n_conds, n_epochs))
        dummy_data = np.empty((n_neurons, n_conds), dtype=object)

        with patch('functions.psth.compute_single_trial_rates', return_value=mock_rates), \
             patch('functions.psth.compute_tuning_curves', return_value=(mock_tuning, ['cue', 'delay'])):
            flat, rates, bc = compute_flat_tuning(
                dummy_data, (-500, 2000), {'cue': (0, 500), 'delay': (500, 2000)}, bin_ms=50)

        assert flat.shape == (n_neurons, n_conds * n_epochs)
        assert rates is mock_rates
        assert len(bc) == 50  # (2500 / 50) bins


class TestCrossTaskCv:
    def test_output_keys(self):
        from functions.analysis import cross_task_cv

        rng = np.random.default_rng(0)
        n = 60
        tuning_flat = {
            'Task A': rng.standard_normal((n, 16)),
            'Task B': rng.standard_normal((n, 16)),
        }
        task_ids = {
            'Task A': np.array(['M0'] * 20 + ['M1'] * 20 + ['M2'] * 20),
            'Task B': np.array(['M0'] * 20 + ['M1'] * 20 + ['M2'] * 20),
        }
        results = cross_task_cv(tuning_flat, task_ids,
                                n_pcs=4, min_neurons=5, n_iter=3, seed=0)

        assert 'cat_means' in results
        assert 'mantel_r' in results
        assert 'last_dist' in results
        assert 'last_labels' in results
        assert 'cat_names' in results
        assert 'n_iter' in results

        # mantel_r is a dict keyed by task-pair tuples
        assert isinstance(results['mantel_r'], dict)
        assert ('Task A', 'Task B') in results['mantel_r']
        assert len(results['mantel_r'][('Task A', 'Task B')]) == 3

        # Check iteration count
        for c in results['cat_names']:
            assert len(results['cat_means'][c]) == 3

        # Check category names
        assert len(results['cat_names']) == 4

    def test_three_tasks(self):
        from functions.analysis import cross_task_cv

        rng = np.random.default_rng(0)
        n = 60
        tuning_flat = {
            'Task A': rng.standard_normal((n, 8)),
            'Task B': rng.standard_normal((n, 8)),
            'Task C': rng.standard_normal((n, 8)),
        }
        ids = np.array(['M0'] * 20 + ['M1'] * 20 + ['M2'] * 20)
        task_ids = {t: ids.copy() for t in tuning_flat}
        results = cross_task_cv(tuning_flat, task_ids,
                                n_pcs=4, min_neurons=5, n_iter=3, seed=0)

        # Should have 3 pairs
        assert len(results['mantel_r']) == 3
        expected_pairs = {('Task A', 'Task B'), ('Task A', 'Task C'),
                          ('Task B', 'Task C')}
        assert set(results['mantel_r'].keys()) == expected_pairs


class TestPlotCrossTask:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        from functions.analysis import CAT_NAMES

        rng = np.random.default_rng(0)
        n_iter = 5
        results = {
            'cat_means': {c: rng.uniform(0.1, 0.5, n_iter) for c in CAT_NAMES},
            'mantel_r': {('Task A', 'Task B'): rng.uniform(0.5, 0.9, n_iter)},
            'last_dist': rng.uniform(0, 1, (6, 6)),
            'last_labels': [f'M{i}' for i in range(6)],
            'cat_names': CAT_NAMES,
            'n_iter': n_iter,
        }
        plot_cross_task(results)
        plt.close("all")

    def test_smoke_three_pairs(self):
        import matplotlib.pyplot as plt
        from functions.analysis import CAT_NAMES

        rng = np.random.default_rng(0)
        n_iter = 5
        results = {
            'cat_means': {c: rng.uniform(0.1, 0.5, n_iter) for c in CAT_NAMES},
            'mantel_r': {
                ('Task A', 'Task B'): rng.uniform(0.5, 0.9, n_iter),
                ('Task A', 'Task C'): rng.uniform(0.3, 0.7, n_iter),
                ('Task B', 'Task C'): rng.uniform(0.4, 0.8, n_iter),
            },
            'last_dist': rng.uniform(0, 1, (9, 9)),
            'last_labels': [f'M{i}' for i in range(9)],
            'cat_names': CAT_NAMES,
            'n_iter': n_iter,
        }
        plot_cross_task(results)
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════════
# representations.py — _clean_neurons, _build_entry
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanNeurons:
    def test_removes_nan(self):
        X = np.array([[1, 2], [np.nan, 1], [3, 4]], dtype=float)
        out = _clean_neurons(X)
        assert out.shape[0] == 2
        np.testing.assert_array_equal(out, [[1, 2], [3, 4]])

    def test_removes_zero_variance(self):
        X = np.array([[1, 2], [5, 5], [3, 4]], dtype=float)
        out = _clean_neurons(X)
        assert out.shape[0] == 2

    def test_preserves_good_rows(self):
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        out = _clean_neurons(X)
        np.testing.assert_array_equal(out, X)


class TestBuildEntry:
    def test_success(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 8))
        entry = _build_entry(X, n_pcs=4, min_neurons=10, zscore=True)
        assert entry is not None
        assert entry['matrix'].shape == (4, 8)
        assert entry['n_neurons'] >= 10
        assert 0 < entry['var_explained'] <= 1

    def test_too_few_returns_none(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 8))
        entry = _build_entry(X, n_pcs=4, min_neurons=10, zscore=True)
        assert entry is None

    def test_extra_keys_propagated(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 8))
        entry = _build_entry(X, n_pcs=4, min_neurons=10, zscore=True,
                             monkey='A', group=2)
        assert entry['monkey'] == 'A'
        assert entry['group'] == 2


class TestAssignPerMonkeyAgeGroups:
    def test_basic(self):
        ids = np.array(['A'] * 30 + ['B'] * 30)
        ages = np.concatenate([np.linspace(20, 80, 30), np.linspace(30, 90, 30)])
        groups, edges = assign_per_monkey_age_groups(ids, ages, 3)
        assert groups.shape == (60,)
        assert set(groups) == {0, 1, 2}
        assert 'A' in edges and 'B' in edges
        assert len(edges['A']) == 2  # 3 bins -> 2 inner edges

    def test_single_monkey(self):
        ids = np.array(['X'] * 20)
        ages = np.linspace(10, 50, 20)
        groups, edges = assign_per_monkey_age_groups(ids, ages, 2)
        assert set(groups) == {0, 1}
        assert len(edges['X']) == 1


class TestTuningToMatrix:
    def test_shape(self):
        info = {'tc': np.random.default_rng(0).standard_normal((5, 4, 2))}
        pts = tuning_to_matrix(info, n_dims=3)
        assert pts.shape == (8, 3)  # 4*2 points, 3 dims

    def test_fewer_dims(self):
        info = {'tc': np.random.default_rng(0).standard_normal((5, 4, 2))}
        pts = tuning_to_matrix(info, n_dims=2)
        assert pts.shape == (8, 2)


class TestPlot3dRepresentation:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pts = np.random.default_rng(0).standard_normal((4, 3))
        colors = ['r', 'g', 'b', 'y']
        plot_3d_representation(ax, pts, colors)
        plt.close('all')


class TestWallProjections:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pts = np.random.default_rng(0).standard_normal((4, 3))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
        wall_projections(ax, pts)
        plt.close('all')


# ═══════════════════════════════════════════════════════════════════════════════
# New functions: pca_reduce fix, constants, filter, cross-epoch, plotting
# ═══════════════════════════════════════════════════════════════════════════════

class TestPcaReduceFix:
    """Verify pca_reduce handles n_pcs > min(n_neurons, n_features) via padding."""

    def test_pads_when_fewer_neurons(self):
        X = np.random.default_rng(0).standard_normal((3, 8))
        proj, var = pca_reduce(X, 8)
        assert proj.shape == (8, 8)
        np.testing.assert_array_equal(proj[3:], 0)

    def test_pads_when_fewer_features(self):
        X = np.random.default_rng(0).standard_normal((50, 3))
        proj, var = pca_reduce(X, 8)
        assert proj.shape == (8, 3)
        np.testing.assert_array_equal(proj[3:], 0)

    def test_no_padding_when_enough(self):
        X = np.random.default_rng(0).standard_normal((50, 8))
        proj, var = pca_reduce(X, 4)
        assert proj.shape == (4, 8)
        assert not np.all(proj == 0)


class TestConstants:
    def test_task_colors_hex(self):
        for name, color in TASK_COLORS.items():
            assert color.startswith('#'), f'{name} color not hex: {color}'

    def test_cardinal_cols(self):
        assert CARDINAL_COLS == [0, 2, 4, 6]

    def test_task_epochs_keys(self):
        assert 'ODR 1.5s' in TASK_EPOCHS
        assert 'ODR 3.0s' in TASK_EPOCHS
        assert 'ODRd' in TASK_EPOCHS
        for v in TASK_EPOCHS.values():
            assert 't_range' in v
            assert 'epochs' in v

    def test_stim_colors_labels_match(self):
        assert len(STIM_COLORS) == len(STIM_LABELS) == 4

    def test_age_colors_labels_match(self):
        assert len(AGE_COLORS) == len(AGE_GROUP_LABELS) == 3


class TestFilterCommonMonkeys:
    def test_basic(self):
        task_data = {
            'T1': {'ids': np.array(['A', 'B', 'C', 'A']),
                    'data': np.zeros((4, 8))},
            'T2': {'ids': np.array(['B', 'C', 'D', 'B']),
                    'data': np.zeros((4, 8))},
        }
        filtered, common = filter_common_monkeys(task_data)
        assert common == ['B', 'C']
        assert len(filtered['T1']['ids']) == 2  # B and C
        assert len(filtered['T2']['ids']) == 3  # B, C, B

    def test_subset_tasks(self):
        task_data = {
            'T1': {'ids': np.array(['A', 'B']), 'data': np.zeros((2, 8))},
            'T2': {'ids': np.array(['B', 'C']), 'data': np.zeros((2, 8))},
            'T3': {'ids': np.array(['A', 'B']), 'data': np.zeros((2, 8))},
        }
        filtered, common = filter_common_monkeys(task_data, task_names=['T1', 'T3'])
        assert common == ['A', 'B']
        assert set(filtered.keys()) == {'T1', 'T3'}


class TestCrossEpochDistances:
    def test_basic(self):
        rng = np.random.default_rng(42)
        # Build fake epoch_reps
        epoch_reps = {'TaskA': {}}
        for ename in ['cue', 'delay']:
            epoch_reps['TaskA'][ename] = {}
            for mid in ['M0', 'M1']:
                for g in [0, 1]:
                    epoch_reps['TaskA'][ename][(mid, g)] = {
                        'matrix': rng.standard_normal((4, 8)),
                    }

        comparisons = [('cue', 'delay')]
        result = cross_epoch_distances(epoch_reps, comparisons)
        assert 'TaskA' in result
        assert 'cue\u2192delay' in result['TaskA']
        rows = result['TaskA']['cue\u2192delay']
        assert len(rows) == 4  # 2 monkeys x 2 groups
        assert all('distance' in r for r in rows)
        assert all(r['distance'] > 0 for r in rows)


class TestPrintCrossTaskSummary:
    def test_smoke(self, capsys):
        from functions.analysis import CAT_NAMES
        rng = np.random.default_rng(0)
        n_iter = 5
        results = {
            'cat_means': {c: rng.uniform(0.1, 0.5, n_iter) for c in CAT_NAMES},
            'mantel_r': {('Task A', 'Task B'): rng.uniform(0.5, 0.9, n_iter)},
            'cat_names': CAT_NAMES,
        }
        print_cross_task_summary(results)
        captured = capsys.readouterr()
        assert 'Category means' in captured.out
        assert 'Mantel r' in captured.out


class TestPrintCrossMonkeyByGroupSummary:
    def test_smoke(self, capsys):
        results = {
            'ODR 1.5s': {
                'common': ['M0', 'M1'],
                'n_monkeys': 3,
                'n_neurons': 50,
                'n_total': 100,
                'group_dists': {0: np.array([0.1, 0.2]), 1: np.array([0.3])},
                'slope': 0.05,
                'se': 0.01,
                'r': 0.5,
                'p': 0.03,
            }
        }
        pooled = {'slope': 0.04, 'se': 0.008, 'r': 0.6, 'p': 0.01}
        print_cross_monkey_by_group_summary(results, pooled, ['young', 'old'])
        captured = capsys.readouterr()
        assert 'ODR 1.5s' in captured.out
        assert 'All tasks pooled' in captured.out


class TestPlot3dGrid:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        reduced = {
            'M0': {0: {'tc': rng.standard_normal((4, 4, 2)),
                        'n_neurons': 20, 'var_explained': 0.7}},
            'M1': {0: {'tc': rng.standard_normal((4, 4, 2)),
                        'n_neurons': 20, 'var_explained': 0.7}},
        }
        plot_3d_grid(reduced, epoch_idx=0)
        plt.close('all')


class TestPlotWithinMonkeyAlignment:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        reduced = {
            'M0': {
                0: {'tc': rng.standard_normal((4, 4, 2)),
                    'n_neurons': 20, 'var_explained': 0.7},
                1: {'tc': rng.standard_normal((4, 4, 2)),
                    'n_neurons': 20, 'var_explained': 0.7},
            },
        }
        epoch_idx = np.arange(0, 8, 2)  # cue indices
        plot_within_monkey_alignment(reduced, epoch_idx)
        plt.close('all')


class TestPlotGlobalAlignment:
    def test_smoke(self):
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        reduced = {
            'M0': {
                0: {'tc': rng.standard_normal((4, 4, 2)),
                    'n_neurons': 20, 'var_explained': 0.7},
                1: {'tc': rng.standard_normal((4, 4, 2)),
                    'n_neurons': 20, 'var_explained': 0.7},
            },
            'M1': {
                0: {'tc': rng.standard_normal((4, 4, 2)),
                    'n_neurons': 20, 'var_explained': 0.7},
            },
        }
        epoch_idx = np.arange(0, 8, 2)
        plot_global_alignment(reduced, epoch_idx)
        plt.close('all')
