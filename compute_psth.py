"""
Compute PSTH (peri-stimulus time histogram) per neuron and condition
for both ODR and ODRd tasks.

Spike times are aligned to cue onset and binned.
Output: firing rate in Hz per neuron × condition × time bin.
"""

import scipy.io as sio
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")

# ── Parameters ────────────────────────────────────────────────────────────────
BIN_MS = 50            # bin width in ms
T_START = -1000        # ms before cue onset
T_END = 4000           # ms after cue onset (covers fixation + cue + 3s delay)
bin_edges = np.arange(T_START, T_END + BIN_MS, BIN_MS)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
n_bins = len(bin_centers)


def extract_psth(data_matrix, n_neurons, n_conditions, label=""):
    """
    Compute PSTH from an odr_data_new-style cell array.

    Parameters
    ----------
    data_matrix : ndarray of objects, shape (n_neurons, n_conditions)
        Each cell is a struct array of trials with fields 'TS' and 'Cue_onT'.
    n_neurons, n_conditions : int
    label : str, for progress printing

    Returns
    -------
    psth : ndarray, shape (n_neurons, n_conditions, n_bins)
        Mean firing rate in Hz.
    trial_counts : ndarray, shape (n_neurons, n_conditions)
        Number of trials per neuron/condition.
    """
    psth = np.full((n_neurons, n_conditions, n_bins), np.nan)
    trial_counts = np.zeros((n_neurons, n_conditions), dtype=int)

    for i in range(n_neurons):
        if i % 500 == 0:
            print(f"  [{label}] neuron {i}/{n_neurons}")
        for c in range(n_conditions):
            cell = data_matrix[i, c]
            if cell is None or (isinstance(cell, np.ndarray) and cell.size == 0):
                continue

            n_trials = max(cell.shape)
            ts_arr = cell['TS'].flatten()
            cue_arr = cell['Cue_onT'].flatten()

            # Accumulate spike counts across trials
            counts_sum = np.zeros(n_bins, dtype=float)
            valid_trials = 0

            for t in range(n_trials):
                ts_raw = ts_arr[t]
                cue_raw = cue_arr[t]

                # Flatten arrays
                if hasattr(ts_raw, 'flatten'):
                    ts_raw = ts_raw.flatten()
                if hasattr(cue_raw, 'flatten'):
                    cue_raw = cue_raw.flatten()
                    if cue_raw.size > 0:
                        cue_raw = cue_raw[0]

                # Skip empty trials
                if not hasattr(ts_raw, 'size') or ts_raw.size == 0:
                    valid_trials += 1  # still a valid trial, just no spikes
                    continue

                # Align to cue onset, convert to ms
                aligned_ms = (ts_raw - cue_raw) * 1000.0

                # Bin spikes
                hist, _ = np.histogram(aligned_ms, bins=bin_edges)
                counts_sum += hist
                valid_trials += 1

            trial_counts[i, c] = valid_trials

            if valid_trials > 0:
                # Convert to firing rate: mean spike count per bin / bin width (s)
                psth[i, c, :] = (counts_sum / valid_trials) / (BIN_MS / 1000.0)

    return psth, trial_counts


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ODR TASK
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading ODR data ...")
mat1 = sio.loadmat(
    os.path.join(DATA_DIR, "odr_data_both_sig_is_best_20240109.mat"),
    squeeze_me=False,
)
odr = mat1["odr_data_new"]
n_neurons_odr = odr.shape[0]
n_cond_odr = odr.shape[1]
print(f"  {n_neurons_odr} neurons × {n_cond_odr} conditions")

print("Computing ODR PSTHs ...")
psth_odr, trials_odr = extract_psth(odr, n_neurons_odr, n_cond_odr, label="ODR")

print(f"\n  psth_odr shape: {psth_odr.shape}  "
      f"(neurons × conditions × time bins)")
print(f"  bin_centers: {n_bins} bins, {T_START} to {T_END} ms, {BIN_MS} ms width")
print(f"  trials per neuron/cond: mean={trials_odr.mean():.1f}, "
      f"median={np.median(trials_odr):.0f}, "
      f"range=[{trials_odr.min()}, {trials_odr.max()}]")

# Quick sanity check: population-average firing rate
valid_mask = ~np.isnan(psth_odr)
mean_fr = np.nanmean(psth_odr)
print(f"  Population-average firing rate: {mean_fr:.2f} Hz")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ODRd TASK
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading ODRd data ...")
mat2 = sio.loadmat(
    os.path.join(DATA_DIR, "odrd_data_sig_on_best_20231018.mat"),
    squeeze_me=False,
)
odrd = mat2["odrd_data_new"]
n_neurons_odrd = odrd.shape[0]
n_cond_odrd = odrd.shape[1]
print(f"  {n_neurons_odrd} neurons × {n_cond_odrd} conditions")

print("Computing ODRd PSTHs ...")
psth_odrd, trials_odrd = extract_psth(odrd, n_neurons_odrd, n_cond_odrd, label="ODRd")

print(f"\n  psth_odrd shape: {psth_odrd.shape}")
print(f"  trials per neuron/cond: mean={trials_odrd.mean():.1f}, "
      f"median={np.median(trials_odrd):.0f}, "
      f"range=[{trials_odrd.min()}, {trials_odrd.max()}]")
mean_fr_d = np.nanmean(psth_odrd)
print(f"  Population-average firing rate: {mean_fr_d:.2f} Hz")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SAVE
# ═══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psth_data.npz")
np.savez_compressed(
    out_path,
    # ODR
    psth_odr=psth_odr,
    trials_odr=trials_odr,
    # ODRd
    psth_odrd=psth_odrd,
    trials_odrd=trials_odrd,
    # Shared
    bin_centers=bin_centers,
    bin_edges=bin_edges,
    bin_ms=BIN_MS,
    t_start=T_START,
    t_end=T_END,
)
print(f"\nSaved to {out_path}")
print(f"  File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Time window:  {T_START} to {T_END} ms (cue-aligned)")
print(f"  Bin width:    {BIN_MS} ms")
print(f"  Time bins:    {n_bins}")
print(f"")
print(f"  ODR task:     {n_neurons_odr} neurons × {n_cond_odr} conditions × {n_bins} bins")
print(f"  ODRd task:    {n_neurons_odrd} neurons × {n_cond_odrd} conditions × {n_bins} bins")
print(f"")
print(f"  To load in Python:")
print(f"    data = np.load('psth_data.npz')")
print(f"    psth_odr  = data['psth_odr']   # shape (2131, 8, {n_bins})")
print(f"    psth_odrd = data['psth_odrd']   # shape (1319, 4, {n_bins})")
print(f"    t = data['bin_centers']         # time axis in ms")
