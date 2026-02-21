"""
Plotting functions for Procrustes analysis pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from .analysis import extract_entry_arrays
from .decoding import knn_decode_age

TASK_EVENTS = {
    'ODR 1.5s': {'Cue': 0, 'Delay': 500, 'Resp': 2000},
    'ODR 3.0s': {'Cue': 0, 'Delay': 500, 'Resp': 3000},
    'ODRd':     {'Cue': 0, 'Delay': 500, 'Dist': 1700, 'Resp': 3000},
}
TASK_COLORS = {'ODR 1.5s': 'C0', 'ODR 3.0s': 'C1', 'ODRd': 'C2'}


def plot_cross_monkey(results):
    """Histograms of within vs cross-monkey distances (raw and adjusted)."""
    n_tasks = len(results)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 4.5 * n_tasks))

    for row, (task_name, R) in enumerate(results.items()):
        cm = R['cross_monkey']

        ax = axes[row, 0]
        ax.hist(cm['within_all_pairs'], bins=12, alpha=0.6, density=True,
                label=f'Within (n={len(cm["within_all_pairs"])})')
        ax.hist(cm['cross_raw'], bins=15, alpha=0.6, density=True,
                label=f'Cross (n={len(cm["cross_raw"])})')
        ax.axvline(cm['within_all_pairs'].mean(), color='C0', ls='--', lw=1.5)
        ax.axvline(cm['cross_raw'].mean(), color='C1', ls='--', lw=1.5)
        ax.set_xlabel('Procrustes distance')
        ax.set_ylabel('Density')
        ax.set_title(f'{task_name} \u2014 raw distributions')
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.hist(cm['cross_adj'], bins=15, alpha=0.7, edgecolor='k')
        ax.axvline(0, color='r', ls='--', lw=1.5, label='0')
        ax.axvline(cm['cross_adj'].mean(), color='C0', ls='--', lw=1.5,
                   label=f'mean={cm["cross_adj"].mean():.4f}')
        ax.set_xlabel('Cross dist \u2212 within baseline')
        ax.set_ylabel('Count')
        ax.set_title(f'{task_name} \u2014 adjusted (t={cm["t_stat"]:.2f}, p={cm["p_val"]:.4f})')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_distance_matrices(results):
    """Procrustes distance matrices as heatmaps."""
    n_tasks = len(results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))

    for ax, (task_name, R) in zip(axes, results.items()):
        n = len(R['labels'])
        im = ax.imshow(R['dist'], cmap='viridis')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(R['labels'], rotation=90, fontsize=7)
        ax.set_yticklabels(R['labels'], fontsize=7)
        ax.set_title(f'{task_name} ({n} entries)')
        plt.colorbar(im, ax=ax, label='Procrustes disparity')

    plt.tight_layout()
    plt.show()


def plot_cross_age(results):
    """Histograms of within-age vs across-age distances (raw and adjusted)."""
    n_tasks = len(results)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 4.5 * n_tasks))

    for row, (task_name, R) in enumerate(results.items()):
        ca = R['cross_age']

        ax = axes[row, 0]
        ax.hist(ca['same_age_raw'], bins=12, alpha=0.6, density=True,
                label=f'Same age (n={len(ca["same_age_raw"])})')
        ax.hist(ca['diff_age_raw'], bins=15, alpha=0.6, density=True,
                label=f'Diff age (n={len(ca["diff_age_raw"])})')
        ax.axvline(ca['same_age_raw'].mean(), color='C0', ls='--', lw=1.5)
        ax.axvline(ca['diff_age_raw'].mean(), color='C1', ls='--', lw=1.5)
        ax.set_xlabel('Procrustes distance')
        ax.set_ylabel('Density')
        ax.set_title(f'{task_name} \u2014 cross-monkey pairs by age match')
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.hist(ca['same_age_adj'], bins=12, alpha=0.7, edgecolor='k')
        ax.axvline(0, color='r', ls='--', lw=1.5, label='0')
        ax.axvline(ca['same_age_adj'].mean(), color='C0', ls='--', lw=1.5,
                   label=f'mean={ca["same_age_adj"].mean():.4f}')
        ax.set_xlabel('Same-age cross-monkey \u2212 within-monkey baseline')
        ax.set_ylabel('Count')
        ax.set_title(f'{task_name} \u2014 adjusted (t={ca["t_stat"]:.2f}, p={ca["p_val"]:.4f})')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def _baseline_normalize(boots, t):
    """Subtract pre-cue (t < 0) mean from each bootstrap trace."""
    return boots - np.nanmean(boots[:, t < 0], axis=1, keepdims=True)


def plot_temporal(temporal_results,
                  ylabel='Mean Procrustes distance', title='Temporal Procrustes'):
    """Temporal bootstrap traces with baseline normalization and ±SE shading."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for task_name, TR in temporal_results.items():
        c = TASK_COLORS[task_name]
        boots = _baseline_normalize(TR['boots'], TR['t'])
        mean = np.nanmean(boots, axis=0)
        sem = np.nanstd(boots, axis=0)

        ax.plot(TR['t'], mean, label=task_name, lw=1.5, color=c)
        ax.fill_between(TR['t'], mean - sem, mean + sem, color=c, alpha=0.2)
        for _, t_ms in TASK_EVENTS[task_name].items():
            ax.axvline(t_ms, color=c, ls=':', alpha=0.4, lw=1)

    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Window center (ms from cue onset)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_temporal_by_pair(temporal_pair_results):
    """Temporal cross-age Procrustes separated by age pair."""
    pair_styles = {(0, 1): '-', (1, 2): '--', (0, 2): ':'}
    pair_labels = {(0, 1): 'G0-G1', (1, 2): 'G1-G2', (0, 2): 'G0-G2'}

    fig, ax = plt.subplots(figsize=(14, 5))

    for task_name, TR in temporal_pair_results.items():
        c = TASK_COLORS[task_name]
        for pair, boots in TR['boots_by_pair'].items():
            boots_norm = _baseline_normalize(boots, TR['t'])
            mean = np.nanmean(boots_norm, axis=0)
            sem = np.nanstd(boots_norm, axis=0)

            ax.plot(TR['t'], mean, ls=pair_styles[pair], lw=1.5, color=c,
                    label=f'{task_name} {pair_labels[pair]}')
            ax.fill_between(TR['t'], mean - sem, mean + sem, color=c, alpha=0.08)

        for _, t_ms in TASK_EVENTS[task_name].items():
            ax.axvline(t_ms, color=c, ls=':', alpha=0.4, lw=1)

    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Window center (ms from cue onset)')
    ax.set_ylabel('Mean cross-age Procrustes distance')
    ax.set_title('Cross-age by pair over time (\u00b1SE)')
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.show()


def plot_age_decoding(results, k=3):
    """KNN age decoding scatter plot per task.

    Parameters
    ----------
    results : dict
        {task_name: {entries, dist, ...}} from the main pipeline.
    k : int
        Number of neighbors for KNN.
    """
    n_tasks = len(results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5))
    if n_tasks == 1:
        axes = [axes]

    for ax, (task_name, R) in zip(axes, results.items()):
        entries = R['entries']
        dist = R['dist']
        monkeys, _ = extract_entry_arrays(entries)
        monkey_names = sorted(set(monkeys))

        dec = knn_decode_age(dist, entries, k=k)
        y_true, y_pred = dec['y_true'], dec['y_pred_round']

        # Color by monkey (LOO-monkey order matches monkey_names iteration)
        mk_ids = []
        for test_mk in monkey_names:
            mk_ids.extend([test_mk] * np.sum(monkeys == test_mk))
        mk_ids = np.array(mk_ids)

        cmap = {m: f'C{i}' for i, m in enumerate(monkey_names)}
        colors = [cmap[m] for m in mk_ids]

        # Jitter for visibility
        rng = np.random.default_rng(0)
        jx = rng.uniform(-0.15, 0.15, len(y_true))
        jy = rng.uniform(-0.15, 0.15, len(y_true))
        ax.scatter(y_true + jx, y_pred + jy, c=colors, s=40,
                   edgecolors='k', linewidths=0.5)

        ticks = sorted(set(y_true))
        ax.plot([ticks[0] - 0.3, ticks[-1] + 0.3],
                [ticks[0] - 0.3, ticks[-1] + 0.3], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('True age group')
        ax.set_ylabel('Predicted age group (KNN, LOO-monkey)')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(f'{task_name}\nacc={dec["exact_acc"]:.2f}, '
                     f'\u00b11 acc={dec["pm1_acc"]:.2f}')

        for m in monkey_names:
            ax.scatter([], [], c=cmap[m], label=m, edgecolors='k', linewidths=0.5)
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.show()


def plot_correlation_panels(scatter_data, xlabel, ylabel, suptitle=None):
    """Generic scatter + Pearson r panel plot for behavioral correlations.

    Parameters
    ----------
    scatter_data : dict
        {task_name: {x, y, labels}}
    xlabel, ylabel : str
    suptitle : str, optional
    """
    n_tasks = len(scatter_data)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5))
    if n_tasks == 1:
        axes = [axes]

    for ax, (task_name, S) in zip(axes, scatter_data.items()):
        x, y = S['x'], S['y']
        if len(x) <= 2:
            ax.set_title(f'{task_name}\nnot enough data')
            continue

        r, p = pearsonr(x, y)
        ax.scatter(x, y, s=40, edgecolors='k', linewidths=0.5)

        for k, lbl in enumerate(S.get('labels', [])):
            ax.annotate(lbl, (x[k], y[k]), fontsize=5, alpha=0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{task_name}\nr={r:.2f}, p={p:.3f}, n={len(x)}')

    if suptitle:
        plt.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cross_task(results):
    """Bar chart with SE error bars, Mantel histogram, and example distance matrix.

    Parameters
    ----------
    results : dict
        Output of cross_task_cv.
    """
    cat_means = results['cat_means']
    cat_names = results['cat_names']
    mantel_r = results['mantel_r']
    n_iter = results['n_iter']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Bar plot: mean distance per category with SE error bars
    ax = axes[0]
    cat_colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    positions = range(len(cat_names))
    means = [np.mean(cat_means[c]) for c in cat_names]
    sems = [np.std(cat_means[c]) for c in cat_names]
    ax.bar(positions, means, yerr=sems, capsize=5, color=cat_colors, alpha=0.7,
           edgecolor='k', linewidth=0.5)
    ymin = min(m - s for m, s in zip(means, sems))
    ax.set_ylim(bottom=ymin * 0.9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(cat_names, fontsize=9)
    ax.set_ylabel('Procrustes distance')
    ax.set_title(f'Mean distance per category ({n_iter} iterations, \u00b1SE)')
    for i, c in enumerate(cat_names):
        ax.text(i, means[i] + sems[i] + 0.003, f'{means[i]:.3f}',
                ha='center', fontsize=9)

    # Mantel correlation histogram
    ax = axes[1]
    ax.hist(mantel_r, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
    ax.axvline(np.mean(mantel_r), color='r', ls='--', lw=1.5,
               label=f'mean r = {np.mean(mantel_r):.3f}')
    ax.set_xlabel('Pearson r (Mantel-like)')
    ax.set_ylabel('Count')
    ax.set_title('Cross-task distance correlation across splits')
    ax.legend()

    # Example distance matrix (last iteration)
    ax = axes[2]
    last_dist = results['last_dist']
    last_labels = results['last_labels']
    n = len(last_labels)
    im = ax.imshow(last_dist, cmap='viridis')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(last_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(last_labels, fontsize=7)
    plt.colorbar(im, ax=ax, label='Procrustes disparity')
    ax.set_title('Distance matrix (last iteration)')

    plt.tight_layout()
    plt.show()

    # Summary stats
    print('Category means [\u00b1SE]:')
    for c in cat_names:
        m = np.mean(cat_means[c])
        sem = np.std(cat_means[c])
        print(f'  {c.replace(chr(10), " "):30s}  {m:.4f} \u00b1 {sem:.4f}')

    print(f'\nMantel r: mean = {np.mean(mantel_r):.3f} \u00b1 '
          f'{np.std(mantel_r):.3f} (SE)')

    # Bootstrap CI on differences
    comparisons = [
        ('Same monkey cross-task vs diff monkey within-task',
         cat_names[1], cat_names[2]),
        ('Same monkey cross-task vs diff monkey cross-task',
         cat_names[1], cat_names[3]),
    ]
    print()
    for label, ca, cb in comparisons:
        diffs = cat_means[ca] - cat_means[cb]
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        print(f'{label}:\n'
              f'  diff median = {np.median(diffs):.4f}, '
              f'95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], '
              f'{"excludes" if ci_lo > 0 or ci_hi < 0 else "includes"} zero')
