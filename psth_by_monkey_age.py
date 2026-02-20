"""
Compute PSTHs separated by monkey and age group (3 age terciles per monkey).

Extracts neuron metadata (ID, age) directly from the .mat workspace bytes,
then computes trial-averaged PSTHs grouped by monkey × age tercile.
"""

import scipy.io as sio
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")

# ── Parameters ────────────────────────────────────────────────────────────────
BIN_MS  = 50
T_START = -1000
T_END   = 2500            # 1.5s delay: cue 0–500ms, delay 500–2000ms, saccade ~2000ms+
N_AGE_GROUPS = 3          # terciles within each monkey

bin_edges   = np.arange(T_START, T_END + BIN_MS, BIN_MS)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
n_bins      = len(bin_centers)
n_cond      = 8

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & EXTRACT METADATA FROM WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading .mat file ...")
mat = sio.loadmat(
    os.path.join(DATA_DIR, "odr_data_both_sig_is_best_20240109.mat"),
    squeeze_me=False,
)
odr  = mat["odr_data_new"]
ws   = mat["__function_workspace__"].tobytes()
n_neurons_raw = odr.shape[0]  # 2131

# ── Extract monkey IDs (UTF-16-LE encoded in workspace) ──
ids_to_find = ["OLI", "PIC", "QUA", "ROS", "SON", "TRI", "UNI", "VIK"]
all_id_hits = []
for name in ids_to_find:
    encoded = name.encode("utf-16-le")
    start = 0
    while True:
        pos = ws.find(encoded, start)
        if pos == -1:
            break
        all_id_hits.append((pos, name))
        start = pos + len(encoded)
all_id_hits.sort()
id_col_raw = np.array([h[1] for h in all_id_hits])
assert len(id_col_raw) == n_neurons_raw

# ── Extract numeric columns (miDOUBLE, 8-byte header then data) ──
def read_doubles(offset, n=n_neurons_raw):
    return np.frombuffer(ws[offset + 8 : offset + 8 + n * 8], dtype="<f8")

neuron_age_raw = read_doubles(303472)   # days relative to maturation
mature_age_raw = read_doubles(611080)   # maturation reference (days)
delay_dur_raw  = read_doubles(754488)   # delay duration (s)

# ── Filter: keep only 1.5s delay neurons ──────────────────────────────────────
# All 8 monkeys have 1.5s delay data (the standard ODR task).
# 4 monkeys (OLI, PIC, ROS, UNI) additionally have 3s delay (ODR3).
# 29 neurons with delay=0 have no trial data and are excluded.
keep = delay_dur_raw == 1.5
print(f"  Raw: {n_neurons_raw} neurons")
print(f"  Delay durations: 0s={int((delay_dur_raw==0).sum())}, "
      f"1.5s={int(keep.sum())}, 3s={int((delay_dur_raw==3).sum())}")
print(f"  Keeping only 1.5s delay: {int(keep.sum())} neurons")

odr        = odr[keep]
id_col     = id_col_raw[keep]
neuron_age = neuron_age_raw[keep]
mature_age = mature_age_raw[keep]
abs_age_months = (neuron_age + mature_age) / 365.0 * 12.0
n_neurons  = int(keep.sum())

print(f"  {n_neurons} neurons, {len(set(id_col))} monkeys")
for mid in sorted(set(id_col)):
    print(f"    {mid}: {int((id_col == mid).sum())} neurons")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ASSIGN AGE GROUPS (terciles within each monkey)
# ═══════════════════════════════════════════════════════════════════════════════
age_group = np.zeros(n_neurons, dtype=int)
monkeys   = sorted(set(id_col))

print("\nAge group edges (absolute age in months):")
age_group_edges = {}
for mid in monkeys:
    mask = id_col == mid
    ages = abs_age_months[mask]
    edges = np.quantile(ages, np.linspace(0, 1, N_AGE_GROUPS + 1))
    age_group_edges[mid] = edges
    # Assign groups 0, 1, 2
    groups = np.digitize(ages, edges[1:-1])  # 0-indexed terciles
    age_group[mask] = groups
    labels = [f"{edges[g]:.1f}–{edges[g+1]:.1f}" for g in range(N_AGE_GROUPS)]
    counts = [int((groups == g).sum()) for g in range(N_AGE_GROUPS)]
    print(f"  {mid:>4s}: " + "  |  ".join(
        f"G{g}: {labels[g]} mo (n={counts[g]})" for g in range(N_AGE_GROUPS)))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE PSTHs PER NEURON & CONDITION (trial-averaged)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing per-neuron PSTHs ...")
psth_all = np.full((n_neurons, n_cond, n_bins), np.nan)

for i in range(n_neurons):
    if i % 500 == 0:
        print(f"  neuron {i}/{n_neurons}")
    for c in range(n_cond):
        cell = odr[i, c]
        if cell is None or (isinstance(cell, np.ndarray) and cell.size == 0):
            continue
        n_trials = max(cell.shape)
        ts_arr  = cell["TS"].flatten()
        cue_arr = cell["Cue_onT"].flatten()

        counts_sum = np.zeros(n_bins)
        valid = 0
        for t in range(n_trials):
            ts  = ts_arr[t]
            cue = cue_arr[t]
            if hasattr(ts, "flatten"):
                ts = ts.flatten()
            if hasattr(cue, "flatten"):
                cue = cue.flatten()
                if cue.size > 0:
                    cue = cue[0]
            if hasattr(ts, "size") and ts.size > 0:
                aligned = (ts - cue) * 1000.0
                hist, _ = np.histogram(aligned, bins=bin_edges)
                counts_sum += hist
            valid += 1

        if valid > 0:
            psth_all[i, c, :] = (counts_sum / valid) / (BIN_MS / 1000.0)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. AVERAGE PSTHs BY MONKEY × AGE GROUP × CONDITION
# ═══════════════════════════════════════════════════════════════════════════════
print("\nAveraging by monkey × age group × condition ...")
# Result dict: psth_grouped[monkey][age_group] = (n_cond, n_bins)
psth_grouped = {}
neuron_counts = {}

for mid in monkeys:
    psth_grouped[mid] = {}
    neuron_counts[mid] = {}
    for g in range(N_AGE_GROUPS):
        mask = (id_col == mid) & (age_group == g)
        n_sel = mask.sum()
        if n_sel > 0:
            # Average across neurons (ignoring NaN)
            psth_grouped[mid][g] = np.nanmean(psth_all[mask], axis=0)  # (n_cond, n_bins)
        else:
            psth_grouped[mid][g] = np.full((n_cond, n_bins), np.nan)
        neuron_counts[mid][g] = n_sel
        print(f"  {mid} age_group {g}: {n_sel} neurons")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE
# ═══════════════════════════════════════════════════════════════════════════════
# Pack into arrays for easy saving
monkey_list = monkeys
n_monkeys = len(monkey_list)

# Shape: (n_monkeys, N_AGE_GROUPS, n_cond, n_bins)
psth_array = np.full((n_monkeys, N_AGE_GROUPS, n_cond, n_bins), np.nan)
count_array = np.zeros((n_monkeys, N_AGE_GROUPS), dtype=int)
age_edges_array = np.zeros((n_monkeys, N_AGE_GROUPS + 1))

for mi, mid in enumerate(monkey_list):
    for g in range(N_AGE_GROUPS):
        psth_array[mi, g] = psth_grouped[mid][g]
        count_array[mi, g] = neuron_counts[mid][g]
    age_edges_array[mi] = age_group_edges[mid]

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "psth_by_monkey_age.npz")
np.savez_compressed(
    out_path,
    psth=psth_array,             # (8 monkeys, 3 age groups, 8 conditions, 100 bins)
    neuron_counts=count_array,   # (8 monkeys, 3 age groups)
    age_edges=age_edges_array,   # (8 monkeys, 4 edges)
    monkey_names=np.array(monkey_list),
    bin_centers=bin_centers,
    bin_edges=bin_edges,
    bin_ms=BIN_MS,
    # Also save per-neuron data for flexibility
    psth_all=psth_all,           # (2131, 8, 100)
    id_col=id_col,               # (2131,)
    abs_age_months=abs_age_months,
    age_group=age_group,
)
print(f"\nSaved to {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLOT
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    age_labels = ["Young", "Middle", "Old"]
    age_colors = ["#2196F3", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for mi, mid in enumerate(monkey_list):
        ax = axes[mi]
        for g in range(N_AGE_GROUPS):
            # Average across conditions for the population PSTH
            mean_psth = np.nanmean(psth_array[mi, g], axis=0)  # avg over 8 conds
            n = count_array[mi, g]
            e0, e1 = age_edges_array[mi, g], age_edges_array[mi, g + 1]
            ax.plot(bin_centers, mean_psth, color=age_colors[g], lw=1.5,
                    label=f"{age_labels[g]} ({e0:.0f}–{e1:.0f} mo, n={n})")
        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(500, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(mid, fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        if mi >= 4:
            ax.set_xlabel("Time from cue (ms)")
        if mi % 4 == 0:
            ax.set_ylabel("Firing rate (Hz)")

    fig.suptitle("Population PSTH by monkey and age group (averaged across 8 cue conditions)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "psth_by_monkey_age.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    # Also plot per-condition for one monkey (e.g., UNI, the largest)
    fig2, axes2 = plt.subplots(N_AGE_GROUPS, 1, figsize=(10, 9), sharex=True, sharey=True)
    cmap = plt.cm.hsv(np.linspace(0, 0.85, n_cond))
    mi_uni = monkey_list.index("UNI")
    for g in range(N_AGE_GROUPS):
        ax = axes2[g]
        for c in range(n_cond):
            ax.plot(bin_centers, psth_array[mi_uni, g, c], color=cmap[c], lw=1.2,
                    label=f"Cue {c+1}")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axvline(500, color="gray", ls=":", lw=0.8)
        n = count_array[mi_uni, g]
        e0, e1 = age_edges_array[mi_uni, g], age_edges_array[mi_uni, g + 1]
        ax.set_title(f"UNI – {age_labels[g]} ({e0:.0f}–{e1:.0f} mo, n={n} neurons)",
                     fontsize=11)
        ax.set_ylabel("Firing rate (Hz)")
        if g == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper right")
    axes2[-1].set_xlabel("Time from cue (ms)")
    fig2.suptitle("UNI: PSTH per condition × age group", fontsize=13, y=1.01)
    plt.tight_layout()
    fig2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "psth_UNI_conditions.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig2_path}")

except ImportError:
    print("matplotlib not available, skipping plots")

print("\nDone.")
print(f"\nTo load:")
print(f"  data = np.load('psth_by_monkey_age.npz', allow_pickle=True)")
print(f"  psth = data['psth']              # (8 monkeys, 3 ages, 8 conds, {n_bins} bins)")
print(f"  names = data['monkey_names']     # ['OLI', 'PIC', ...]")
print(f"  t = data['bin_centers']           # time in ms")
print(f"  edges = data['age_edges']         # (8, 4) age tercile edges in months")
print(f"  # Also per-neuron: data['psth_all'], data['id_col'], data['abs_age_months']")
