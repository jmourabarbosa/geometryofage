"""
Plotting functions for Procrustes analysis pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

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


def plot_temporal(temporal_results, ci=95,
                  ylabel='Mean Procrustes distance', title='Temporal Procrustes'):
    """Temporal bootstrap traces with baseline normalization and CI shading."""
    alpha = (100 - ci) / 2
    fig, ax = plt.subplots(figsize=(14, 5))

    for task_name, TR in temporal_results.items():
        c = TASK_COLORS[task_name]
        boots = _baseline_normalize(TR['boots'], TR['t'])
        mean = np.nanmean(boots, axis=0)
        lo = np.nanpercentile(boots, alpha, axis=0)
        hi = np.nanpercentile(boots, 100 - alpha, axis=0)

        ax.plot(TR['t'], mean, label=task_name, lw=1.5, color=c)
        ax.fill_between(TR['t'], lo, hi, color=c, alpha=0.2)
        for _, t_ms in TASK_EVENTS[task_name].items():
            ax.axvline(t_ms, color=c, ls=':', alpha=0.4, lw=1)

    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Window center (ms from cue onset)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_temporal_by_pair(temporal_pair_results, ci=95):
    """Temporal cross-age Procrustes separated by age pair."""
    pair_styles = {(0, 1): '-', (1, 2): '--', (0, 2): ':'}
    pair_labels = {(0, 1): 'G0-G1', (1, 2): 'G1-G2', (0, 2): 'G0-G2'}
    alpha = (100 - ci) / 2

    fig, ax = plt.subplots(figsize=(14, 5))

    for task_name, TR in temporal_pair_results.items():
        c = TASK_COLORS[task_name]
        for pair, boots in TR['boots_by_pair'].items():
            boots_norm = _baseline_normalize(boots, TR['t'])
            mean = np.nanmean(boots_norm, axis=0)
            lo = np.nanpercentile(boots_norm, alpha, axis=0)
            hi = np.nanpercentile(boots_norm, 100 - alpha, axis=0)

            ax.plot(TR['t'], mean, ls=pair_styles[pair], lw=1.5, color=c,
                    label=f'{task_name} {pair_labels[pair]}')
            ax.fill_between(TR['t'], lo, hi, color=c, alpha=0.08)

        for _, t_ms in TASK_EVENTS[task_name].items():
            ax.axvline(t_ms, color=c, ls=':', alpha=0.4, lw=1)

    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Window center (ms from cue onset)')
    ax.set_ylabel('Mean cross-age Procrustes distance')
    ax.set_title(f'Cross-age by pair over time ({ci}% CI)')
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.show()
