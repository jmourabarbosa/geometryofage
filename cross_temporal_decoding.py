"""
Cross-temporal SVM decoding on pseudo-populations (ODR task).

For each pair (t_train, t_test), an SVM is trained on population vectors at
t_train and tested at t_test using stratified K-fold cross-validation.

Pseudo-population: since neurons come from different sessions, single-trial
responses are randomly paired across neurons within each condition.
"""

import scipy.io as sio
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")

# ── Parameters ────────────────────────────────────────────────────────────────
BIN_MS       = 50          # bin width (ms)
T_START      = -1000       # ms relative to cue onset
T_END        = 4000        # ms
MIN_TRIALS   = 5           # minimum trials per condition to include a neuron
N_PSEUDO     = 10          # pseudo-trials per condition (sampled with replacement)
N_CV_FOLDS   = 5           # stratified K-fold
N_PERM       = 10          # number of pseudo-population resamplings
C_REG        = 1.0         # SVM regularisation
RANDOM_SEED  = 42

bin_edges   = np.arange(T_START, T_END + BIN_MS, BIN_MS)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
n_bins      = len(bin_centers)

np.random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. EXTRACT SINGLE-TRIAL FIRING RATES
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading ODR data ...")
mat = sio.loadmat(
    os.path.join(DATA_DIR, "odr_data_both_sig_is_best_20240109.mat"),
    squeeze_me=False,
)
odr = mat["odr_data_new"]
n_neurons_raw, n_cond = odr.shape
print(f"  {n_neurons_raw} neurons × {n_cond} conditions")

print("Extracting single-trial firing rates ...")
# Store as list of lists: single_trial[neuron][condition] = array (n_trials, n_bins)
single_trial = []
trial_counts = np.zeros((n_neurons_raw, n_cond), dtype=int)

for i in range(n_neurons_raw):
    neuron_data = []
    for c in range(n_cond):
        cell = odr[i, c]
        if cell is None or (isinstance(cell, np.ndarray) and cell.size == 0):
            neuron_data.append(np.empty((0, n_bins)))
            continue

        n_trials = max(cell.shape)
        ts_arr  = cell["TS"].flatten()
        cue_arr = cell["Cue_onT"].flatten()

        trial_rates = []
        for t in range(n_trials):
            ts_raw  = ts_arr[t]
            cue_raw = cue_arr[t]
            if hasattr(ts_raw, "flatten"):
                ts_raw = ts_raw.flatten()
            if hasattr(cue_raw, "flatten"):
                cue_raw = cue_raw.flatten()
                if cue_raw.size > 0:
                    cue_raw = cue_raw[0]

            if hasattr(ts_raw, "size") and ts_raw.size > 0:
                aligned_ms = (ts_raw - cue_raw) * 1000.0
                hist, _ = np.histogram(aligned_ms, bins=bin_edges)
            else:
                hist = np.zeros(n_bins)

            trial_rates.append(hist / (BIN_MS / 1000.0))  # Hz

        arr = np.array(trial_rates)  # (n_trials, n_bins)
        neuron_data.append(arr)
        trial_counts[i, c] = arr.shape[0]

    single_trial.append(neuron_data)

    if i % 500 == 0:
        print(f"  neuron {i}/{n_neurons_raw}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SELECT NEURONS WITH ENOUGH TRIALS
# ═══════════════════════════════════════════════════════════════════════════════
min_per_neuron = trial_counts.min(axis=1)  # min across conditions
valid_neurons  = np.where(min_per_neuron >= MIN_TRIALS)[0]
n_valid = len(valid_neurons)

print(f"\n  Valid neurons (>= {MIN_TRIALS} trials/cond): {n_valid} / {n_neurons_raw}")
print(f"  Pseudo-trials per condition: {N_PSEUDO} (sampled with replacement)")
print(f"  Total samples: {N_PSEUDO * n_cond} ({n_cond} cond × {N_PSEUDO} trials)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-TEMPORAL DECODING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Strategy (fast):
#   For each CV fold, train SVM at t_train → store weights.
#   Then project test data at every t_test in one shot.
#   This avoids fitting 100×100 SVMs per fold.
#
# ─────────────────────────────────────────────────────────────────────────────

labels = np.repeat(np.arange(n_cond), N_PSEUDO)  # (n_cond * N_PSEUDO,)
n_samples = len(labels)

acc_all = np.zeros((N_PERM, n_bins, n_bins))  # resamplings × t_train × t_test

print(f"\nRunning cross-temporal decoding "
      f"({N_PERM} resamplings × {N_CV_FOLDS}-fold CV) ...")
t0 = time.time()

for perm in range(N_PERM):
    print(f"\nPermutation {perm + 1}/{N_PERM} ...")
    # ── Build pseudo-population matrix: (n_samples, n_valid, n_bins) ──
    # For each neuron independently, randomly sample N_PSEUDO trials per condition
    # (with replacement when N_PSEUDO > available trials)
    X = np.zeros((n_samples, n_valid, n_bins))
    for j, ni in enumerate(valid_neurons):
        for c in range(n_cond):
            n_avail = single_trial[ni][c].shape[0]
            replace = N_PSEUDO > n_avail
            chosen = np.random.choice(n_avail, size=N_PSEUDO, replace=replace)
            X[c * N_PSEUDO : (c + 1) * N_PSEUDO, j, :] = \
                single_trial[ni][c][chosen, :]

    # ── Cross-validated cross-temporal decoding ──
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED + perm)
    fold_acc = np.zeros((N_CV_FOLDS, n_bins, n_bins))

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X[:, 0, 0], labels)):
        y_train = labels[train_idx]
        y_test  = labels[test_idx]

        for t_train in range(n_bins):
            # Extract & scale features at t_train
            X_tr = X[train_idx][:, :, t_train]   # (n_train, n_neurons)
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)

            # Train SVM
            clf = LinearSVC(C=C_REG, max_iter=10000, dual=False)
            clf.fit(X_tr_s, y_train)

            # Test at every t_test using the same scaler and model
            for t_test in range(n_bins):
                X_te = X[test_idx][:, :, t_test]
                X_te_s = scaler.transform(X_te)
                fold_acc[fold_i, t_train, t_test] = \
                    np.mean(clf.predict(X_te_s) == y_test)

    acc_all[perm] = fold_acc.mean(axis=0)  # average over folds

    elapsed = time.time() - t0
    print(f"  perm {perm + 1}/{N_PERM}  "
          f"({elapsed:.0f}s elapsed, "
          f"diag acc={np.diag(acc_all[perm]).mean():.3f})")

# Average over resamplings
acc_mean = acc_all.mean(axis=0)   # (n_bins, n_bins)
acc_std  = acc_all.std(axis=0)

total_time = time.time() - t0
print(f"\nDone in {total_time:.0f}s")
print(f"  Diagonal accuracy (mean): {np.diag(acc_mean).mean():.3f}")
print(f"  Chance level: {1.0 / n_cond:.3f}")
print(f"  Peak accuracy: {acc_mean.max():.3f} at "
      f"t_train={bin_centers[np.unravel_index(acc_mean.argmax(), acc_mean.shape)[0]]:.0f} ms, "
      f"t_test={bin_centers[np.unravel_index(acc_mean.argmax(), acc_mean.shape)[1]]:.0f} ms")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cross_temporal_decoding.npz")
np.savez_compressed(
    out_path,
    acc_mean=acc_mean,
    acc_std=acc_std,
    acc_all=acc_all,
    bin_centers=bin_centers,
    bin_edges=bin_edges,
    bin_ms=BIN_MS,
    n_neurons=n_valid,
    n_cond=n_cond,
    n_pseudo=N_PSEUDO,
    n_cv_folds=N_CV_FOLDS,
    n_perm=N_PERM,
    chance=1.0 / n_cond,
)
print(f"\nSaved to {out_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PLOT
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Cross-temporal matrix --
    ax = axes[0]
    im = ax.imshow(
        acc_mean,
        origin="lower",
        aspect="equal",
        extent=[T_START, T_END, T_START, T_END],
        vmin=1.0 / n_cond,
        vmax=acc_mean.max(),
        cmap="magma",
    )
    ax.axvline(0, color="w", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(0, color="w", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("Test time (ms)")
    ax.set_ylabel("Train time (ms)")
    ax.set_title("Cross-temporal decoding accuracy")
    plt.colorbar(im, ax=ax, label="Accuracy")

    # -- Diagonal (temporal) --
    ax = axes[1]
    diag = np.diag(acc_mean)
    diag_std = np.diag(acc_std)
    ax.plot(bin_centers, diag, "k-", lw=1.5)
    ax.fill_between(bin_centers, diag - diag_std, diag + diag_std,
                    alpha=0.2, color="k")
    ax.axhline(1.0 / n_cond, color="gray", ls="--", lw=1, label="chance")
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(500, color="blue", ls=":", lw=0.8, alpha=0.5, label="cue off (500ms)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Diagonal decoding accuracy")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(T_START, T_END)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cross_temporal_decoding.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")
except ImportError:
    print("matplotlib not available, skipping plot")
