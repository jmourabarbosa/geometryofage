"""
Plotting functions for Procrustes analysis pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts


# See also TASK_EPOCHS in load_data.py for analysis epoch windows.
TASK_EVENTS = {
    'ODR 1.5s': {'Cue': 0, 'Delay': 500, 'Resp': 2000},
    'ODR 3.0s': {'Cue': 0, 'Delay': 500, 'Resp': 3000},
    'ODRd':     {'Cue': 0, 'Delay': 500, 'Dist': 1700, 'Resp': 3000},
}
TASK_COLORS = {'ODR 1.5s': '#1b9e77', 'ODR 3.0s': '#d95f02', 'ODRd': '#7570b3'}

STIM_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
STIM_LABELS = ['0°', '90°', '180°', '270°']
AGE_COLORS = ['#1b9e77', '#d95f02', '#7570b3']
AGE_GROUP_LABELS = ['young', 'middle', 'old']


def p_to_stars(p, ns_label='ns'):
    """Convert p-value to significance stars."""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ns_label


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


def wall_projections(ax, pts, color='gray', alpha=0.15):
    """Draw shadow projections on the 3 walls of a 3-D axis.

    Parameters
    ----------
    ax : Axes3D
    pts : ndarray, shape (n_points, 3)
    color : str or tuple
    alpha : float
    """
    loop = np.vstack([pts, pts[0:1]])
    xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ax.plot(loop[:, 0], loop[:, 1], np.full(len(loop), zl[0]),
            color=color, lw=2, alpha=alpha)
    ax.plot(loop[:, 0], np.full(len(loop), yl[1]), loop[:, 2],
            color=color, lw=2, alpha=alpha)
    ax.plot(np.full(len(loop), xl[0]), loop[:, 1], loop[:, 2],
            color=color, lw=2, alpha=alpha)


def plot_surface_patch(ax, pts, color='lightsteelblue', alpha=0.25, n=12):
    """Bilinear surface through 4 points arranged as a quad.

    Parameters
    ----------
    ax : Axes3D
    pts : ndarray, shape (4, 3)
    color : str or tuple
    alpha : float
    n : int
        Grid resolution.
    """
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, v)
    X = (1-U)*(1-V)*pts[0,0] + U*(1-V)*pts[1,0] + U*V*pts[2,0] + (1-U)*V*pts[3,0]
    Y = (1-U)*(1-V)*pts[0,1] + U*(1-V)*pts[1,1] + U*V*pts[2,1] + (1-U)*V*pts[3,1]
    Z = (1-U)*(1-V)*pts[0,2] + U*(1-V)*pts[1,2] + U*V*pts[2,2] + (1-U)*V*pts[3,2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False, zorder=0)


def wall_surface_projections(ax, pts, color='gray', alpha=0.10, n=12):
    """Project bilinear surface patch onto the 3 walls of a 3-D axis.

    Parameters
    ----------
    ax : Axes3D
    pts : ndarray, shape (4, 3)
    color : str or tuple
    alpha : float
    n : int
        Grid resolution.
    """
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, v)
    X = (1-U)*(1-V)*pts[0,0] + U*(1-V)*pts[1,0] + U*V*pts[2,0] + (1-U)*V*pts[3,0]
    Y = (1-U)*(1-V)*pts[0,1] + U*(1-V)*pts[1,1] + U*V*pts[2,1] + (1-U)*V*pts[3,1]
    Z = (1-U)*(1-V)*pts[0,2] + U*(1-V)*pts[1,2] + U*V*pts[2,2] + (1-U)*V*pts[3,2]
    xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ax.plot_surface(X, Y, np.full_like(Z, zl[0]), color=color, alpha=alpha, shade=False, zorder=0)
    ax.plot_surface(X, np.full_like(Y, yl[1]), Z, color=color, alpha=alpha, shade=False, zorder=0)
    ax.plot_surface(np.full_like(X, xl[0]), Y, Z, color=color, alpha=alpha, shade=False, zorder=0)


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
    """Histograms of adjusted same-age cross-monkey distances."""
    n_tasks = len(results)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(6, 4.5 * n_tasks))
    if n_tasks == 1:
        axes = [axes]

    for row, (task_name, R) in enumerate(results.items()):
        ca = R['cross_age']

        ax = axes[row]
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


def print_cross_task_summary(results):
    """Print summary statistics for cross-task CV results.

    Parameters
    ----------
    results : dict
        Output of cross_task_cv.
    """
    cat_means = results['cat_means']
    cat_names = results['cat_names']
    mantel_r = results['mantel_r']

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

        all_g, all_d = [], []
        for g in range(n_groups):
            dists = R['group_dists'][g]
            all_g.extend([g] * len(dists))
            all_d.extend(dists)
        all_g = np.array(all_g)
        all_d = np.array(all_d)

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


def print_cross_monkey_by_group_summary(results, pooled, group_labels):
    """Print summary statistics for cross-monkey by group analysis.

    Parameters
    ----------
    results : dict
        Per-task output from cross_monkey_by_group.
    pooled : dict
        Pooled regression output from cross_monkey_by_group.
    group_labels : list of str
    """
    n_groups = len(group_labels)
    for task_name, R in results.items():
        print(f'{task_name}: {len(R["common"])}/{R["n_monkeys"]} monkeys in all '
              f'{n_groups} groups ({R["common"]}), '
              f'{R["n_neurons"]}/{R["n_total"]} neurons')
        for g in range(n_groups):
            dists = R['group_dists'][g]
            print(f'  {group_labels[g]}: {len(dists)} pairs')
        print(f'  regression: slope={R["slope"]:.4f} +/- {R["se"]:.4f}, '
              f'r={R["r"]:.3f}, p={R["p"]:.4f}')
    print(f'\nAll tasks pooled: slope={pooled["slope"]:.4f} +/- {pooled["se"]:.4f}, '
          f'r={pooled["r"]:.3f}, p={pooled["p"]:.4f}')


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
        star = p_to_stars(p)
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
    from .procrustes import _upper_tri

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
            star = p_to_stars(p)
            ax.plot(x_fit, intercept + slope * x_fit, color=c, lw=1.5,
                    label=f'{task_name}: r={r:.3f}, p={p:.3f} {star}')

            pooled_n.append(nv_v - np.mean(nv_v))
            pooled_b.append(bv_v - np.mean(bv_v))

        # Pooled regression
        pn = np.concatenate(pooled_n)
        pb = np.concatenate(pooled_b)
        slope, intercept, r, p, se = sts.linregress(pb, pn)
        x_fit = np.array([pb.min(), pb.max()])
        star = p_to_stars(p)
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


def plot_cross_epoch_correlations(cross_epoch, beh_df, monkey_edges, task_epochs,
                                   comparisons):
    """Scatter + regression plots of cross-epoch distance vs behavioral measures.

    Parameters
    ----------
    cross_epoch : dict
        {task: {label: [dict(monkey, group, distance), ...]}}
    beh_df : DataFrame
        Behavioral data from load_behavioral_data.
    monkey_edges : dict
        {(task, monkey): tuple of edges}
    task_epochs : dict
        Task names (keys used for iteration order).
    comparisons : list of tuple
        Epoch pairs, e.g. [('cue', 'delay'), ('delay', 'response')].
    """
    from .behavior import get_behavioral_values

    fig, axes = plt.subplots(2, len(comparisons), figsize=(14, 8))

    for ci, (ea, eb) in enumerate(comparisons):
        label = f'{ea}\u2192{eb}'

        for ri, beh_name in enumerate(['DI', 'RT']):
            ax = axes[ri, ci]
            all_d, all_beh = [], []

            for task_name in task_epochs:
                if task_name not in cross_epoch:
                    continue
                rows = cross_epoch[task_name].get(label, [])
                if not rows:
                    continue

                entries = [{'monkey': r['monkey'], 'group': r['group']}
                           for r in rows]
                di_vals, rt_vals = get_behavioral_values(
                    beh_df, entries, task_name, monkey_edges)
                beh_vals = di_vals if beh_name == 'DI' else rt_vals

                dists = np.array([r['distance'] for r in rows])
                valid = np.isfinite(beh_vals)

                ax.scatter(dists[valid] - np.mean(dists[valid]),
                           beh_vals[valid] - np.mean(beh_vals[valid]),
                           c=TASK_COLORS[task_name], label=task_name,
                           s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

                all_d.extend(dists[valid] - np.mean(dists[valid]))
                all_beh.extend(beh_vals[valid] - np.mean(beh_vals[valid]))

            all_d = np.array(all_d)
            all_beh = np.array(all_beh)
            if len(all_d) >= 3:
                rho, p = sts.spearmanr(all_d, all_beh)
                m, b = np.polyfit(all_d, all_beh, 1)
                x_line = np.linspace(all_d.min(), all_d.max(), 50)
                ax.plot(x_line, m * x_line + b, 'k-', lw=1.5, alpha=0.8)
                ax.set_title(f'{label}: {beh_name} (\u03c1={rho:.3f}, p={p:.3f})',
                             fontsize=9)
            else:
                ax.set_title(f'{label}: {beh_name}', fontsize=9)

            ax.set_xlabel(f'Procrustes distance ({label})', fontsize=8)
            ax.set_ylabel(beh_name, fontsize=8)
            ax.tick_params(labelsize=7)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7)

    fig.suptitle('Cross-epoch Procrustes distance vs behavioral measures',
                 fontsize=11, y=1.02)
    plt.tight_layout()


def plot_3d_grid(reduced, epoch_idx, stim_colors=None, stim_labels=None,
                 age_labels=None, title=''):
    """Grid of individual 3D representations.

    Parameters
    ----------
    reduced : dict
        {monkey: {age_group: dict(tc=ndarray (n_pcs, n_conds, n_epochs), ...)}}
    epoch_idx : int
        Which epoch to plot (index into last dim of tc).
    stim_colors : list
    stim_labels : list
    age_labels : list
    title : str
    """
    if stim_colors is None:
        stim_colors = STIM_COLORS
    if stim_labels is None:
        stim_labels = STIM_LABELS
    if age_labels is None:
        age_labels = AGE_GROUP_LABELS

    n_plots = sum(len(g) for g in reduced.values())
    ncols = min(n_plots, 5)
    nrows = int(np.ceil(n_plots / ncols))
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))

    i = 1
    for mid, groups in reduced.items():
        for g, info in groups.items():
            ax = fig.add_subplot(nrows, ncols, i, projection='3d')
            tc = info['tc']
            pts = tc[:3, :, epoch_idx].T  # (n_conds, 3)
            plot_3d_representation(ax, pts, stim_colors)

            g_label = age_labels[g] if g < len(age_labels) else str(g)
            ax.set_title(f'{mid} / {g_label}', fontsize=8)
            ax.set_xlabel('PC1', fontsize=7)
            ax.set_ylabel('PC2', fontsize=7)
            ax.set_zlabel('PC3', fontsize=7)
            ax.tick_params(labelsize=6)
            if i == 1:
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=c, markersize=8, label=l)
                           for c, l in zip(stim_colors, stim_labels)]
                ax.legend(handles=handles, fontsize=6, loc='best')
            i += 1

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()


def plot_within_monkey_alignment(reduced, epoch_idx, stim_colors=None,
                                  stim_labels=None, title='', n_dims=3):
    """Within-monkey Procrustes alignment of age groups.

    Parameters
    ----------
    reduced : dict
        {monkey: {age_group: dict(tc=...)}}
    epoch_idx : ndarray
        Indices into the flattened (n_conds * n_epochs) point matrix for the epoch.
    stim_colors, stim_labels : list
    title : str
    n_dims : int
    """
    from .procrustes import generalized_procrustes
    from .representations import tuning_to_matrix

    if stim_colors is None:
        stim_colors = STIM_COLORS
    if stim_labels is None:
        stim_labels = STIM_LABELS

    monkeys_multi = [m for m in reduced if len(reduced[m]) > 1]
    ncols = min(len(monkeys_multi), 4)
    nrows = int(np.ceil(len(monkeys_multi) / ncols))
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for idx, mid in enumerate(monkeys_multi):
        groups_dict = reduced[mid]
        group_ids = sorted(groups_dict.keys())
        mats = [tuning_to_matrix(groups_dict[g], n_dims) for g in group_ids]
        aligned, mean = generalized_procrustes(mats)

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

        for k, g in enumerate(group_ids):
            pts = aligned[k][epoch_idx]
            plot_3d_representation(ax, pts, stim_colors, s=25, alpha=0.35,
                                   lw=1, edge='none')

        mean_pts = mean[epoch_idx]
        plot_3d_representation(ax, mean_pts, stim_colors, s=100, alpha=1.0,
                               lw=2, edge='k', ew=0.8)
        wall_projections(ax, mean_pts)

        ax.set_title(mid, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_zlabel('PC3', fontsize=8)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markersize=8, label=l) for c, l in zip(stim_colors, stim_labels)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()


def plot_global_alignment(reduced, epoch_idx, stim_colors=None,
                           stim_labels=None, title='', n_dims=3):
    """Global Procrustes alignment of all monkey x age groups.

    Parameters
    ----------
    reduced : dict
        {monkey: {age_group: dict(tc=...)}}
    epoch_idx : ndarray
        Indices into the flattened point matrix for the epoch.
    stim_colors, stim_labels : list
    title : str
    n_dims : int
    """
    from .procrustes import generalized_procrustes
    from .representations import tuning_to_matrix

    if stim_colors is None:
        stim_colors = STIM_COLORS
    if stim_labels is None:
        stim_labels = STIM_LABELS

    all_mats = []
    for mid, groups_dict in reduced.items():
        for g in sorted(groups_dict.keys()):
            all_mats.append(tuning_to_matrix(groups_dict[g], n_dims))

    aligned_all, grand_mean = generalized_procrustes(all_mats)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(len(aligned_all)):
        pts = aligned_all[k][epoch_idx]
        for i in range(len(pts)):
            ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2],
                       s=15, color=stim_colors[i], alpha=0.2,
                       edgecolors='none', zorder=1, clip_on=False)

    mean_pts = grand_mean[epoch_idx]
    plot_3d_representation(ax, mean_pts, stim_colors, s=120, alpha=1.0,
                           lw=2.5, edge='k', ew=0.8)
    wall_projections(ax, mean_pts)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.set_zlabel('PC3', fontsize=9)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markersize=8, label=l) for c, l in zip(stim_colors, stim_labels)]
    ax.legend(handles=handles, fontsize=8, loc='upper left')
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
# draw_* helpers — take a provided ax (for composite figures)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_3d_alignment(ax, task_result, plot_epochs, epoch_colors, stim_colors, n_conds):
    """Draw per-task Procrustes-aligned 3-D grand-mean with surface patches."""
    enames = task_result['enames']
    lim = task_result['lim']
    n_ep = len(plot_epochs)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    for ename in plot_epochs:
        ei = enames.index(ename)
        epoch_idx = np.arange(ei, n_conds * n_ep, n_ep)
        ec = epoch_colors[ename]
        mean_pts = task_result['grand_mean'][epoch_idx]

        plot_surface_patch(ax, mean_pts, color=ec, alpha=0.2)

        loop = np.vstack([mean_pts, mean_pts[0:1]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], '-', color=ec,
                lw=2.5, alpha=0.6, zorder=1)
        for i in range(len(mean_pts)):
            ax.scatter(mean_pts[i, 0], mean_pts[i, 1], mean_pts[i, 2],
                       s=60, color='k', alpha=1.0,
                       edgecolors=stim_colors[i], linewidths=1.5,
                       zorder=2, clip_on=False)

        wall_projections(ax, mean_pts, color=ec, alpha=0.15)
        wall_surface_projections(ax, mean_pts, color=ec, alpha=0.10)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))


def draw_cross_task_bars(ax, ct_results, cat_colors=None):
    """Bar chart of mean Procrustes distance per cross-task category."""
    cat_means = ct_results['cat_means']
    cat_names = ct_results['cat_names']
    if cat_colors is None:
        cat_colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    means = [np.nanmean(cat_means[c]) for c in cat_names]
    sems = [np.nanstd(cat_means[c]) for c in cat_names]
    ax.bar(range(len(cat_names)), means, yerr=sems, capsize=5, color=cat_colors,
           alpha=0.7, edgecolor='k', linewidth=0.5)
    ymin = min(m - s for m, s in zip(means, sems))
    ax.set_ylim(bottom=ymin * 0.9)
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_names, fontsize=7)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def draw_cross_age_bars(ax, indiv_results, task_colors):
    """Bar plot of adjusted same-age cross-monkey distance with pooled bar."""
    task_names = list(indiv_results.keys())
    means, sems, pvals, labels = [], [], [], []
    all_adj = []
    for name in task_names:
        ca = indiv_results[name]['cross_age']
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
    _, pooled_p = sts.ttest_1samp(pooled_adj, 0, nan_policy='omit')
    means.append(pooled_mean); sems.append(pooled_sem); pvals.append(pooled_p)
    labels.append('All tasks')
    colors = [task_colors.get(n, 'C0') for n in task_names] + ['k']

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.7,
           edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ylim = max(abs(m) + s for m, s in zip(means, sems)) + 0.02
    ax.set_ylim(-ylim, ylim)

    for i, (m, s, p) in enumerate(zip(means, sems, pvals)):
        star = p_to_stars(p)
        y_pos = m + s + 0.003 if m >= 0 else m - s - 0.003
        va = 'bottom' if m >= 0 else 'top'
        ax.text(i, y_pos, star, ha='center', va=va, fontsize=7, fontweight='bold')

    arrow_x = -0.55
    ax.annotate('', xy=(arrow_x, ylim * 0.85), xytext=(arrow_x, 0),
                arrowprops=dict(arrowstyle='->', color='0.4', lw=1.5))
    ax.annotate('', xy=(arrow_x, -ylim * 0.85), xytext=(arrow_x, 0),
                arrowprops=dict(arrowstyle='->', color='0.4', lw=1.5))
    ax.text(arrow_x + 0.15, ylim * 0.5, 'different individuals\n(same age)', fontsize=8,
            color='0.4', va='center', ha='center')
    ax.text(arrow_x + 0.15, -ylim * 0.5, 'different ages\n(same individual)', fontsize=8,
            color='0.4', va='center', ha='center')
    ax.set_xlim(left=-1.0)


def draw_cross_monkey_scatter(ax, results_by_group, pooled, age_group_labels, task_colors):
    """Jittered scatter + regression of cross-monkey distance by age group."""
    n_groups = len(age_group_labels)
    task_names = list(results_by_group.keys())
    offsets = np.linspace(-0.15, 0.15, len(task_names))

    for ti, task_name in enumerate(task_names):
        R = results_by_group[task_name]
        c = task_colors[task_name]

        all_g, all_d = [], []
        for g in range(n_groups):
            dists = R['group_dists'][g]
            all_g.extend([g] * len(dists))
            all_d.extend(dists)
        all_g = np.array(all_g)
        all_d = np.array(all_d)

        rng = np.random.default_rng(ti)
        x_jitter = all_g + offsets[ti] + rng.uniform(-0.05, 0.05, len(all_g))
        ax.scatter(x_jitter, all_d, color=c, alpha=0.3, s=15, edgecolors='none')

        group_means = [np.nanmean(R['group_dists'][g]) if len(R['group_dists'][g]) > 0
                       else np.nan for g in range(n_groups)]
        group_sems = [np.nanstd(R['group_dists'][g]) / np.sqrt(len(R['group_dists'][g]))
                      if len(R['group_dists'][g]) > 1 else 0 for g in range(n_groups)]
        x_off = np.arange(n_groups) + offsets[ti]
        ax.errorbar(x_off, group_means, yerr=group_sems, fmt='o', color=c,
                    capsize=4, markersize=5, zorder=5)

        xfit = np.array([0, n_groups - 1]) + offsets[ti]
        ax.plot(xfit, R['intercept'] + R['slope'] * np.array([0, n_groups - 1]),
                color=c, ls='--', lw=1.5,
                label=f'{task_name}: p={R["p"]:.4f}')

    ax.plot([0, n_groups - 1],
            [pooled['intercept'], pooled['intercept'] + pooled['slope'] * (n_groups - 1)],
            color='k', ls='-', lw=2,
            label=f'All: p={pooled["p"]:.4f}')

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(age_group_labels, fontsize=8)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.legend(fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def draw_neural_vs_behavior(ax, indiv_results, beh_dist, measure, task_colors,
                            xlabel=None, show_ylabel=True, show_left_spine=True):
    """Scatter + regression of neural vs behavioral (DI or RT) distances."""
    from .procrustes import _upper_tri

    task_names = list(indiv_results.keys())
    pooled_n, pooled_b = [], []

    for task_name in task_names:
        nv = _upper_tri(indiv_results[task_name]['dist'])
        bv = _upper_tri(beh_dist[task_name][measure])
        valid = np.isfinite(nv) & np.isfinite(bv)
        nv_v = nv[valid] - np.mean(nv[valid])
        bv_v = bv[valid] - np.mean(bv[valid])

        c = task_colors[task_name]
        ax.scatter(bv_v, nv_v, color=c, alpha=0.35, s=18, edgecolors='none')

        slope, intercept, r, p, se = sts.linregress(bv_v, nv_v)
        x_fit = np.array([bv_v.min(), bv_v.max()])
        star = p_to_stars(p)
        ax.plot(x_fit, intercept + slope * x_fit, color=c, lw=1.5,
                label=f'{task_name}: r={r:.3f} {star}')

        pooled_n.append(nv_v)
        pooled_b.append(bv_v)

    pn = np.concatenate(pooled_n)
    pb = np.concatenate(pooled_b)
    slope, intercept, r, p, se = sts.linregress(pb, pn)
    x_fit = np.array([pb.min(), pb.max()])
    star = p_to_stars(p)
    ax.plot(x_fit, intercept + slope * x_fit, color='k', lw=2,
            label=f'All: r={r:.3f} {star}')

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel('Neural distance' if show_ylabel else '', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not show_left_spine:
        ax.spines['left'].set_visible(False)


def draw_cross_epoch_vs_behavior(ax, cross_epoch, cross_epoch_defs, beh_df, monkey_edges,
                                 comparison_label, measure, task_colors,
                                 xlabel=None, ylabel=None, show_left_spine=True):
    """Scatter + Spearman regression of cross-epoch distance vs DI or RT."""
    from .behavior import get_behavioral_values

    all_d, all_beh = [], []
    for task_name in cross_epoch_defs:
        rows = cross_epoch[task_name].get(comparison_label, [])
        if not rows:
            continue
        entries = [{'monkey': r['monkey'], 'group': r['group']} for r in rows]
        di_vals, rt_vals = get_behavioral_values(beh_df, entries, task_name, monkey_edges)
        beh_vals = di_vals if measure == 'DI' else rt_vals
        dists = np.array([r['distance'] for r in rows])
        valid = np.isfinite(beh_vals)
        d_sub = dists[valid] - np.mean(dists[valid])
        b_sub = beh_vals[valid] - np.mean(beh_vals[valid])
        c = task_colors[task_name]
        ax.scatter(d_sub, b_sub, c=c, s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
        all_d.extend(d_sub)
        all_beh.extend(b_sub)

    all_d = np.array(all_d)
    all_beh = np.array(all_beh)
    if len(all_d) >= 3:
        rho, p = sts.spearmanr(all_d, all_beh)
        m, b = np.polyfit(all_d, all_beh, 1)
        x_line = np.linspace(all_d.min(), all_d.max(), 50)
        star = p_to_stars(p)
        ax.plot(x_line, m * x_line + b, 'k-', lw=1.5, alpha=0.8)
        ax.set_title(f'{measure} (\u03c1={rho:.3f} {star})', fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not show_left_spine:
        ax.spines['left'].set_visible(False)


def draw_correlation_matrices(fig, gs_slot, cross_epoch, cross_epoch_defs, beh_df,
                              monkey_edges, pairs, pos_map):
    """Summary Spearman rho matrices (DI, RT) with shared horizontal colorbar."""
    from .behavior import get_behavioral_values
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    gs_inner = gs_slot.subgridspec(2, 2, height_ratios=[0.12, 1], hspace=0.15, wspace=0.35)
    ax_cb = fig.add_subplot(gs_inner[0, :])
    ax_di_mat = fig.add_subplot(gs_inner[1, 0])
    ax_rt_mat = fig.add_subplot(gs_inner[1, 1])

    rho_vals = {}
    p_vals_mat = {}
    for beh_name in ['DI', 'RT']:
        rhos, ps = [], []
        for ea, eb in pairs:
            label = f'{ea}\u2192{eb}'
            all_d, all_beh = [], []
            for task_name in cross_epoch_defs:
                rows = cross_epoch[task_name].get(label, [])
                if not rows:
                    continue
                entries = [{'monkey': r['monkey'], 'group': r['group']} for r in rows]
                di_vals, rt_vals = get_behavioral_values(beh_df, entries, task_name, monkey_edges)
                beh_vals = di_vals if beh_name == 'DI' else rt_vals
                dists = np.array([r['distance'] for r in rows])
                valid = np.isfinite(beh_vals)
                all_d.extend(dists[valid] - np.mean(dists[valid]))
                all_beh.extend(beh_vals[valid] - np.mean(beh_vals[valid]))
            if len(all_d) >= 3:
                rho, p = sts.spearmanr(all_d, all_beh)
                rhos.append(rho)
                ps.append(p)
            else:
                rhos.append(np.nan)
                ps.append(np.nan)
        rho_vals[beh_name] = rhos
        p_vals_mat[beh_name] = ps

    cmap = plt.cm.RdBu_r
    norm = Normalize(vmin=-1, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    fig.colorbar(sm, cax=ax_cb, orientation='horizontal', label='Spearman \u03c1')
    ax_cb.xaxis.set_ticks_position('top')
    ax_cb.xaxis.set_label_position('top')
    ax_cb.tick_params(labelsize=5)

    for ax_mat, beh_name in [(ax_di_mat, 'DI'), (ax_rt_mat, 'RT')]:
        for k, (r, c) in pos_map.items():
            rho = rho_vals[beh_name][k]
            p = p_vals_mat[beh_name][k]
            color = cmap(norm(rho)) if np.isfinite(rho) else 'lightgrey'
            ax_mat.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color,
                                            edgecolor='k', linewidth=1.5))
            if np.isfinite(rho):
                stars = p_to_stars(p, ns_label='')
                ax_mat.text(c + 0.5, r + 0.5, f'{rho:.2f}{stars}',
                            ha='center', va='center', fontsize=8, fontweight='bold')

        ax_mat.text(0.5, 1.5, 'delay', ha='center', va='center', fontsize=6)
        ax_mat.set_xlim(0, 2)
        ax_mat.set_ylim(2, 0)
        ax_mat.set_xticks([1.5])
        ax_mat.set_xticklabels(['response'], fontsize=6)
        ax_mat.set_yticks([0.5])
        ax_mat.set_yticklabels(['cue'], fontsize=6)
        ax_mat.tick_params(length=0)
        ax_mat.set_aspect('equal')
        ax_mat.set_title(beh_name, fontsize=8)
        ax_mat.spines[:].set_visible(False)
