#!/usr/bin/env python
"""Interactive 3D per-task global alignment (cue / delay / response).

Run:  python plot_per_task_alignment.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from functions import (
    load_cardinal_task_data,
    pooled_tuning_by_group, pca_reduce_tuning,
    tuning_to_matrix, wall_projections,
    plot_surface_patch, wall_surface_projections,
    generalized_procrustes,
    STIM_COLORS, STIM_LABELS, TASK_EPOCHS,
)

# ── Parameters ────────────────────────────────────────────────────────────
DATA_DIR = '../data_raw'
AGE_EDGES = (48, 60)
BIN_MS = 25
N_PCS = 4
MIN_NEURONS = N_PCS + 1

PLOT_EPOCHS = ['cue', 'delay', 'response']
EPOCH_COLORS = {'cue': '#2196F3', 'delay': '#FF9800', 'response': '#9C27B0'}
TASK_LIST = ['ODR 1.5s', 'ODR 3.0s', 'ODRd']


# ── Load data ─────────────────────────────────────────────────────────────
print('Loading data...')
cardinal_data = load_cardinal_task_data(DATA_DIR)

# ── Compute alignments ───────────────────────────────────────────────────
print('Computing alignments...')
task_results = {}
for task_name in TASK_LIST:
    all_epochs = TASK_EPOCHS[task_name]['epochs']
    epochs = {k: all_epochs[k] for k in PLOT_EPOCHS}

    grouped_t, enames = pooled_tuning_by_group(
        {task_name: cardinal_data[task_name]}, epochs, AGE_EDGES, bin_ms=BIN_MS)
    reduced_t = pca_reduce_tuning(grouped_t, n_pcs=N_PCS, min_neurons=MIN_NEURONS)

    all_mats = []
    for mid in sorted(reduced_t.keys()):
        for g in sorted(reduced_t[mid].keys()):
            all_mats.append(tuning_to_matrix(reduced_t[mid][g], n_dims=3))
    aligned_all, grand_mean = generalized_procrustes(all_mats)

    lim = np.max(np.abs(grand_mean)) * 1.6
    task_results[task_name] = dict(aligned_all=aligned_all, grand_mean=grand_mean,
                                   enames=enames, lim=lim)

# ── Plot ──────────────────────────────────────────────────────────────────
print('Plotting...')
n_conds = 4
n_ep = len(PLOT_EPOCHS)
fig = plt.figure(figsize=(18, 6))

for col, task_name in enumerate(TASK_LIST):
    res = task_results[task_name]
    enames = res['enames']
    lim = res['lim']

    ax = fig.add_subplot(1, 3, col + 1, projection='3d')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    # Collect per-stimulus points across epochs for connecting lines
    stim_pts = {i: [] for i in range(n_conds)}

    for ename in PLOT_EPOCHS:
        ei = enames.index(ename)
        epoch_idx = np.arange(ei, n_conds * n_ep, n_ep)
        ec = EPOCH_COLORS[ename]
        mean_pts = res['grand_mean'][epoch_idx]

        plot_surface_patch(ax, mean_pts, color=ec, alpha=0.2)

        loop = np.vstack([mean_pts, mean_pts[0:1]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], '-', color=ec,
                lw=2.5, alpha=0.6, zorder=1)
        for i in range(len(mean_pts)):
            ax.scatter(mean_pts[i, 0], mean_pts[i, 1], mean_pts[i, 2],
                       s=120, color='k', alpha=1.0,
                       edgecolors=STIM_COLORS[i], linewidths=1.5,
                       zorder=2, clip_on=False)
            stim_pts[i].append(mean_pts[i])

        wall_projections(ax, mean_pts, color=ec, alpha=0.15)
        wall_surface_projections(ax, mean_pts, color=ec, alpha=0.10)

    # Connect same stimulus across epochs
    for i in range(n_conds):
        pts = np.array(stim_pts[i])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-', color=STIM_COLORS[i],
                lw=1.5, alpha=0.7, zorder=1)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_zlabel('PC3', fontsize=8)
    ax.set_title(task_name, fontsize=13)

# Combined legend
dir_handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='k', markeredgecolor=c,
                       markeredgewidth=1.5, markersize=10, label=l)
               for c, l in zip(STIM_COLORS, STIM_LABELS)]
epoch_handles = [Line2D([0], [0], color=EPOCH_COLORS[e], lw=3, label=e)
                 for e in PLOT_EPOCHS]
fig.legend(handles=dir_handles + epoch_handles, loc='lower center',
           ncol=len(STIM_COLORS) + len(PLOT_EPOCHS), fontsize=12,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Per-task global alignment (cue / delay / response)', fontsize=15)
plt.tight_layout()
plt.show()
