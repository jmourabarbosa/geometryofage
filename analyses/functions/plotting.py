"""
Plotting functions for Procrustes analysis pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts


TASK_EVENTS = {
    'ODR 1.5s': {'Cue': 0, 'Delay': 500, 'Resp': 2000},
    'ODR 3.0s': {'Cue': 0, 'Delay': 500, 'Resp': 3000},
    'ODRd':     {'Cue': 0, 'Delay': 500, 'Dist': 1700, 'Resp': 3000},
}
TASK_COLORS = {'ODR 1.5s': 'C0', 'ODR 3.0s': 'C1', 'ODRd': 'C2'}


def plot_3d_representation(ax, pts, stim_colors, s=40, alpha=1.0,
                           lw=1.5, edge='k', ew=0.5):
    """Plot one 3-D representation: colored dots + circular connection.

    Parameters
    ----------
    ax : Axes3D
    pts : ndarray, shape (n_points, 3)
    stim_colors : list of color specs, one per point
    s, alpha, lw, edge, ew : scatter/line styling
    """
    loop = np.vstack([pts, pts[0:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], '-', color='gray',
            lw=lw, alpha=alpha * 0.5, zorder=1)
    for i in range(len(pts)):
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2],
                   s=s, color=stim_colors[i], alpha=alpha,
                   edgecolors=edge, linewidths=ew, zorder=2, clip_on=False)


def wall_projections(ax, pts):
    """Draw gray shadow projections on the 3 walls of a 3-D axis.

    Parameters
    ----------
    ax : Axes3D
    pts : ndarray, shape (n_points, 3)
    """
    loop = np.vstack([pts, pts[0:1]])
    xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ax.plot(loop[:, 0], loop[:, 1], np.full(len(loop), zl[0]),
            color='gray', lw=2, alpha=0.15)
    ax.plot(loop[:, 0], np.full(len(loop), yl[1]), loop[:, 2],
            color='gray', lw=2, alpha=0.15)
    ax.plot(np.full(len(loop), xl[0]), loop[:, 1], loop[:, 2],
            color='gray', lw=2, alpha=0.15)


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
        ax.axvline(np.nanmean(cm['within_all_pairs']), color='C0', ls='--', lw=1.5)
        ax.axvline(np.nanmean(cm['cross_raw']), color='C1', ls='--', lw=1.5)
        ax.set_xlabel('Procrustes distance')
        ax.set_ylabel('Density')
        ax.set_title(f'{task_name} \u2014 raw distributions')
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.hist(cm['cross_adj'], bins=15, alpha=0.7, edgecolor='k')
        ax.axvline(0, color='r', ls='--', lw=1.5, label='0')
        ax.axvline(np.nanmean(cm['cross_adj']), color='C0', ls='--', lw=1.5,
                   label=f'mean={np.nanmean(cm["cross_adj"]):.4f}')
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

        # ax = axes[row, 0]
        # ax.hist(ca['same_age_raw'], bins=12, alpha=0.6, density=True,
        #         label=f'Same age (n={len(ca["same_age_raw"])})')
        # ax.hist(ca['diff_age_raw'], bins=15, alpha=0.6, density=True,
        #         label=f'Diff age (n={len(ca["diff_age_raw"])})')
        # ax.axvline(np.nanmean(ca['same_age_raw']), color='C0', ls='--', lw=1.5)
        # ax.axvline(np.nanmean(ca['diff_age_raw']), color='C1', ls='--', lw=1.5)
        # ax.set_xlabel('Procrustes distance')
        # ax.set_ylabel('Density')
        # ax.set_title(f'{task_name} \u2014 cross-monkey pairs by age match')
        # ax.legend(fontsize=8)

        ax = axes[row, 0]
        ax.hist(ca['same_age_adj'], bins=12, alpha=0.7, edgecolor='k')
        ax.axvline(0, color='r', ls='--', lw=1.5, label='0')
        ax.axvline(np.nanmean(ca['same_age_adj']), color='C0', ls='--', lw=1.5,
                   label=f'mean={np.nanmean(ca["same_age_adj"]):.4f}')
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
    means = [np.nanmean(cat_means[c]) for c in cat_names]
    sems = [np.nanstd(cat_means[c]) for c in cat_names]
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

    # Mantel correlation histogram (one per task pair)
    ax = axes[1]
    pair_colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod', 'gray']
    for idx, (pair, vals) in enumerate(mantel_r.items()):
        if len(vals) == 0:
            continue
        color = pair_colors[idx % len(pair_colors)]
        label = f'{pair[0]} vs {pair[1]}'
        ax.hist(vals, bins=20, color=color, edgecolor='k', alpha=0.5,
                label=f'{label}: r={np.nanmean(vals):.3f}')
        ax.axvline(np.nanmean(vals), color=color, ls='--', lw=1.5)
    ax.set_xlabel('Pearson r (Mantel-like)')
    ax.set_ylabel('Count')
    ax.set_title('Cross-task distance correlation across splits')
    ax.legend(fontsize=7)

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
        m = np.nanmean(cat_means[c])
        sem = np.nanstd(cat_means[c])
        print(f'  {c.replace(chr(10), " "):30s}  {m:.4f} \u00b1 {sem:.4f}')

    print('\nMantel r:')
    for pair, vals in mantel_r.items():
        if len(vals) == 0:
            continue
        print(f'  {pair[0]} vs {pair[1]}: mean = {np.nanmean(vals):.3f} \u00b1 '
              f'{np.nanstd(vals):.3f} (SE)')

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
        ci_lo, ci_hi = np.nanpercentile(diffs, [2.5, 97.5])
        print(f'{label}:\n'
              f'  diff median = {np.nanmedian(diffs):.4f}, '
              f'95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], '
              f'{"excludes" if ci_lo > 0 or ci_hi < 0 else "includes"} zero')


def plot_cross_monkey_by_group(results, pooled, group_labels):
    """Scatter + regression of cross-monkey distance by age group.

    Parameters
    ----------
    results : dict
        Per-task output from cross_monkey_by_group.
    pooled : dict
        Pooled regression output from cross_monkey_by_group.
    group_labels : list of str
    """
    n_groups = len(group_labels)
    task_names = list(results.keys())
    offsets = np.linspace(-0.15, 0.15, len(task_names))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ti, task_name in enumerate(task_names):
        R = results[task_name]
        c = TASK_COLORS[task_name]

        print(f'{task_name}: {len(R["common"])}/{R["n_monkeys"]} monkeys in all '
              f'{n_groups} groups ({R["common"]}), '
              f'{R["n_neurons"]}/{R["n_total"]} neurons')

        all_g, all_d = [], []
        for g in range(n_groups):
            dists = R['group_dists'][g]
            all_g.extend([g] * len(dists))
            all_d.extend(dists)
            print(f'  {group_labels[g]}: {len(dists)} pairs')
        all_g = np.array(all_g)
        all_d = np.array(all_d)

        print(f'  regression: slope={R["slope"]:.4f} +/- {R["se"]:.4f}, '
              f'r={R["r"]:.3f}, p={R["p"]:.4f}')

        # Left: scatter + mean/SEM + regression line
        rng = np.random.default_rng(ti)
        x_jitter = all_g + offsets[ti] + rng.uniform(-0.05, 0.05, len(all_g))
        axes[0].scatter(x_jitter, all_d, color=c, alpha=0.3, s=15, edgecolors='none')

        group_means = [np.nanmean(R['group_dists'][g]) if len(R['group_dists'][g]) > 0
                       else np.nan for g in range(n_groups)]
        group_sems = [np.nanstd(R['group_dists'][g]) / np.sqrt(len(R['group_dists'][g]))
                      if len(R['group_dists'][g]) > 1 else 0 for g in range(n_groups)]
        x_off = np.arange(n_groups) + offsets[ti]
        axes[0].errorbar(x_off, group_means, yerr=group_sems, fmt='o', color=c,
                         capsize=4, markersize=7, zorder=5)

        xfit = np.array([0, n_groups - 1]) + offsets[ti]
        axes[0].plot(xfit, R['intercept'] + R['slope'] * np.array([0, n_groups - 1]),
                     color=c, ls='--', lw=1.5,
                     label=f'{task_name}: slope={R["slope"]:.4f}, p={R["p"]:.4f}')

        # Right: slope with CI
        axes[1].errorbar(ti, R['slope'], yerr=1.96 * R['se'], fmt='o', color=c,
                         capsize=6, markersize=8,
                         label=f'{task_name} (p={R["p"]:.4f})')

    # Pooled line and point
    print(f'\nAll tasks pooled: slope={pooled["slope"]:.4f} +/- {pooled["se"]:.4f}, '
          f'r={pooled["r"]:.3f}, p={pooled["p"]:.4f}')

    axes[0].plot([0, n_groups - 1],
                 [pooled['intercept'],
                  pooled['intercept'] + pooled['slope'] * (n_groups - 1)],
                 color='k', ls='-', lw=2,
                 label=f'All tasks: slope={pooled["slope"]:.4f}, p={pooled["p"]:.4f}')

    axes[1].errorbar(len(task_names), pooled['slope'], yerr=1.96 * pooled['se'],
                     fmt='s', color='k', capsize=6, markersize=8,
                     label=f'All tasks (p={pooled["p"]:.4f})')

    axes[0].set_xticks(range(n_groups))
    axes[0].set_xticklabels(group_labels)
    axes[0].set_ylabel('Cross-monkey Procrustes distance')
    axes[0].set_title('Cross-monkey distance by age group (common monkeys)')
    axes[0].legend(fontsize=8)

    axes[1].axhline(0, color='k', ls='--', lw=1)
    axes[1].set_xticks(range(len(task_names) + 1))
    axes[1].set_xticklabels(list(task_names) + ['All'])
    axes[1].set_ylabel('Regression slope (distance / age group)')
    axes[1].set_title('Slope \u00b1 95% CI')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_cross_age_bars(results):
    """Bar plot of mean adjusted same-age cross-monkey distance per task + pooled.

    Parameters
    ----------
    results : dict
        {task_name: dict(cross_age=...)} — output of the main pipeline.
    """
    task_names = list(results.keys())
    n = len(task_names)

    means, sems, pvals, labels = [], [], [], []
    all_adj = []

    for name in task_names:
        ca = results[name]['cross_age']
        adj = ca['same_age_adj']
        all_adj.append(adj)
        means.append(np.nanmean(adj))
        sems.append(np.nanstd(adj) / np.sqrt(np.sum(np.isfinite(adj))))
        pvals.append(ca['p_val'])
        labels.append(name)

    # Pooled
    pooled_adj = np.concatenate(all_adj)
    pooled_mean = np.nanmean(pooled_adj)
    pooled_sem = np.nanstd(pooled_adj) / np.sqrt(np.sum(np.isfinite(pooled_adj)))
    pooled_t, pooled_p = sts.ttest_1samp(pooled_adj, 0, nan_policy='omit')
    means.append(pooled_mean)
    sems.append(pooled_sem)
    pvals.append(pooled_p)
    labels.append('All tasks')

    colors = [TASK_COLORS.get(name, 'C0') for name in task_names] + ['k']

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.7,
                  edgecolor='k', linewidth=0.5)

    for i, (m, s, p) in enumerate(zip(means, sems, pvals)):
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(i, m + s + 0.002, f'{m:.4f}\np={p:.4f} {star}',
                ha='center', fontsize=8, va='bottom')

    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Same-age cross-monkey distance \u2212 within-monkey baseline')
    ax.set_title('Adjusted cross-age distance (mean \u00b1 SEM)')
    ax.set_ylim(-1*(max(means) + max(sems) + 0.01), max(means) + max(sems) + 0.01)
    # remove top and right spines and bottom border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_behavior_neural_bars(results, beh_dist):
    """Scatter plots of neural vs behavioral (DI, RT) distances with regression lines.

    Two panels: one for DI, one for RT.  Each panel shows per-task dots
    coloured by task with per-task regression lines, plus a black pooled
    regression line across all tasks.

    Parameters
    ----------
    results : dict
        {task_name: dict(dist=...)} — neural Procrustes results.
    beh_dist : dict
        {task_name: dict(di_dist=..., rt_dist=...)} — behavioral distance matrices.
    """
    from .behavior import _upper_tri

    measure_keys = {'DI': 'di_dist', 'RT': 'rt_dist'}
    task_names = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (measure, dist_key) in zip(axes, measure_keys.items()):
        pooled_n, pooled_b = [], []

        for task_name in task_names:
            nv = _upper_tri(results[task_name]['dist'])
            bv = _upper_tri(beh_dist[task_name][dist_key])
            valid = np.isfinite(nv) & np.isfinite(bv)
            nv_v, bv_v = nv[valid], bv[valid]

            c = TASK_COLORS[task_name]
            ax.scatter(bv_v-np.mean(bv_v), nv_v-np.mean(nv_v), color=c, alpha=0.35, s=18, edgecolors='none')

            # Per-task regression
            slope, intercept, r, p, se = sts.linregress(bv_v-np.mean(bv_v), nv_v-np.mean(nv_v))
            x_fit = np.array([bv_v.min() - np.mean(bv_v), bv_v.max() - np.mean(bv_v)])
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.plot(x_fit, intercept + slope * x_fit, color=c, lw=1.5,
                    label=f'{task_name}: r={r:.3f}, p={p:.3f} {star}')

            pooled_n.append(nv_v - np.mean(nv_v))
            pooled_b.append(bv_v - np.mean(bv_v))

        # Pooled regression
        pn = np.concatenate(pooled_n)
        pb = np.concatenate(pooled_b)
        slope, intercept, r, p, se = sts.linregress(pb, pn)
        x_fit = np.array([pb.min(), pb.max()])
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.plot(x_fit, intercept + slope * x_fit, color='k', lw=2,
                label=f'All tasks: r={r:.3f}, p={p:.3f} {star}')

        ax.set_xlabel(f'{measure} distance')
        ax.set_ylabel('Neural Procrustes distance')
        ax.set_title(f'Neural vs {measure}')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_age_distributions(task_data, age_edges=None):
    """Histogram of neuron ages per monkey, one row per task.

    Parameters
    ----------
    task_data : dict
        {task_name: dict(ids=..., abs_age=...)}
    age_edges : tuple of float, optional
        Bin edges for age groups; drawn as vertical lines.
    """
    for name in task_data:
        ids = task_data[name]['ids']
        abs_age = task_data[name]['abs_age']
        monkeys = sorted(set(ids))
        n = len(monkeys)

        fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.5), sharey=True, sharex=True)
        if n == 1:
            axes = [axes]

        bins = np.linspace(abs_age.min() - 1, abs_age.max() + 1, 25)

        for ax, mid in zip(axes, monkeys):
            ages = abs_age[ids == mid]
            ax.hist(ages, bins=bins, color='steelblue', edgecolor='white', linewidth=0.4)
            ax.set_title(f'{mid} (n={len(ages)})', fontsize=9)
            if age_edges is not None:
                for edge in age_edges:
                    ax.axvline(edge, color='gray', ls=':', lw=1)

        axes[0].set_ylabel('Neurons')
        fig.supxlabel('Age (months)', fontsize=10)
        fig.suptitle(name, fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.show()
