"""
Interactive 3D MDS embedding of Procrustes distances between
monkey × age group neural representations.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.manifold import MDS
from itertools import product
import plotly.graph_objects as go

data = np.load('tuning_curves.npz', allow_pickle=True)
tuning_all   = data['tuning_all']
id_col       = data['id_col']
age_group    = data['age_group']
monkey_names = list(data['monkey_names'])
age_edges    = data['age_edges']

# ── Tuning: cue + delay ──────────────────────────────────────────────────────
cue_rate   = tuning_all[:, :, 0]
delay_rate = np.nanmean(tuning_all[:, :, 1:3], axis=2)
tuning_cd  = np.stack([cue_rate, delay_rate], axis=2)

# ── PCA per (monkey, age) → 15 PCs ───────────────────────────────────────────
N_PCS = 15
monkey_ids, age_ids, matrices = [], [], []

for g in range(3):
    for mi, mid in enumerate(monkey_names):
        mask = (id_col == mid) & (age_group == g)
        X = tuning_cd[mask].reshape(-1, 16)
        good = np.all(np.isfinite(X), axis=1) & (np.std(X, axis=1) > 0)
        X = X[good]
        X = (X - X.mean(axis=1)[:, None]) / X.std(axis=1)[:, None]
        n_pcs = min(N_PCS, X.shape[0], 16)
        pca = PCA(n_components=n_pcs)
        X_pca = pca.fit_transform(X.T).T
        if n_pcs < N_PCS:
            X_pca = np.vstack([X_pca, np.zeros((N_PCS - n_pcs, 16))])
        matrices.append(X_pca)
        monkey_ids.append(mid)
        age_ids.append(g)

# ── Procrustes distance matrix ────────────────────────────────────────────────
n = len(matrices)
dist = np.zeros((n, n))
for i, j in product(range(n), repeat=2):
    if i < j:
        _, _, d = procrustes(matrices[i].T, matrices[j].T)
        dist[i, j] = dist[j, i] = d

# ── 3D MDS ────────────────────────────────────────────────────────────────────
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dist)

# ── Plot with Plotly ──────────────────────────────────────────────────────────
colors = {m: c for m, c in zip(monkey_names,
          ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
           '#911eb4', '#42d4f4', '#f032e6', '#bfef45'])}

age_labels = ['Young', 'Middle', 'Old']

fig = go.Figure()

# One trace per monkey so legend groups them
for mid in monkey_names:
    idx = [i for i in range(n) if monkey_ids[i] == mid]
    fig.add_trace(go.Scatter3d(
        x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
        mode='markers+text',
        marker=dict(size=12, color=colors[mid], line=dict(width=1, color='black')),
        text=[str(age_ids[i]) for i in idx],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial Black'),
        hovertext=[f"{mid} {age_labels[age_ids[i]]}" for i in idx],
        hoverinfo='text',
        name=mid,
    ))

    # Lines connecting age groups within each monkey
    fig.add_trace(go.Scatter3d(
        x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
        mode='lines',
        line=dict(color=colors[mid], width=3),
        showlegend=False,
        hoverinfo='skip',
    ))

fig.update_layout(
    title='Procrustes distances: 3D MDS (label = age group 0/1/2)',
    scene=dict(
        xaxis_title='MDS dim 1',
        yaxis_title='MDS dim 2',
        zaxis_title='MDS dim 3',
    ),
    width=900, height=700,
    legend_title='Monkey',
)

fig.write_html('procrustes_3d.html')
fig.show()
print('Saved procrustes_3d.html (open in browser to rotate)')
