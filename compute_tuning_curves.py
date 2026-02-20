"""
Compute tuning curves: mean firing rate per neuron × condition × epoch.

4 epochs for the 1.5s delay ODR task:
  - Cue:         0 – 500 ms
  - Early delay: 500 – 1250 ms
  - Late delay:  1250 – 2000 ms
  - Response:    2000 – 2500 ms

Uses the already-computed per-neuron PSTHs from psth_by_monkey_age.npz.
"""

import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(base, "psth_by_monkey_age.npz"), allow_pickle=True)

psth_all       = data["psth_all"]          # (1180, 8, 70)
id_col         = data["id_col"]            # (1180,)
abs_age_months = data["abs_age_months"]    # (1180,)
age_group      = data["age_group"]         # (1180,) 0/1/2
bin_centers    = data["bin_centers"]        # (70,)
monkey_names   = data["monkey_names"]      # (8,)
age_edges      = data["age_edges"]         # (8, 4)
neuron_counts  = data["neuron_counts"]     # (8, 3)

n_neurons, n_cond, n_bins = psth_all.shape

# ── Define epochs ─────────────────────────────────────────────────────────────
epoch_names = ["cue", "early_delay", "late_delay", "response"]
epoch_windows = {
    "cue":         (0,    500),
    "early_delay": (500,  1250),
    "late_delay":  (1250, 2000),
    "response":    (2000, 2500),
}
n_epochs = len(epoch_names)

# Find bin indices for each epoch
epoch_bins = {}
for name, (t0, t1) in epoch_windows.items():
    mask = (bin_centers >= t0) & (bin_centers < t1)
    epoch_bins[name] = mask
    print(f"  {name:12s}: {t0:5d}–{t1:5d} ms  ({mask.sum()} bins)")

# ── Per-neuron tuning curves ─────────────────────────────────────────────────
# Shape: (1180 neurons, 8 conditions, 4 epochs)
tuning_all = np.full((n_neurons, n_cond, n_epochs), np.nan)
for ei, name in enumerate(epoch_names):
    tuning_all[:, :, ei] = np.nanmean(psth_all[:, :, epoch_bins[name]], axis=2)

print(f"\nPer-neuron tuning curves: {tuning_all.shape}  "
      f"(neurons × conditions × epochs)")

# ── Grouped: 8 monkeys × 3 age groups × 8 conditions × 4 epochs ─────────────
monkeys = list(monkey_names)
n_monkeys = len(monkeys)
n_age_groups = 3

tuning_grouped = np.full((n_monkeys, n_age_groups, n_cond, n_epochs), np.nan)
for mi, mid in enumerate(monkeys):
    for g in range(n_age_groups):
        mask = (id_col == mid) & (age_group == g)
        if mask.sum() > 0:
            tuning_grouped[mi, g] = np.nanmean(tuning_all[mask], axis=0)

print(f"Grouped tuning curves: {tuning_grouped.shape}  "
      f"(monkeys × age_groups × conditions × epochs)")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(base, "tuning_curves.npz")
np.savez_compressed(
    out_path,
    # Grouped
    tuning=tuning_grouped,           # (8, 3, 8, 4)
    monkey_names=monkey_names,       # (8,)
    age_edges=age_edges,             # (8, 4)
    neuron_counts=neuron_counts,     # (8, 3)
    epoch_names=np.array(epoch_names),
    epoch_windows=np.array([(t0, t1) for t0, t1 in epoch_windows.values()]),
    # Per-neuron
    tuning_all=tuning_all,           # (1180, 8, 4)
    id_col=id_col,                   # (1180,)
    abs_age_months=abs_age_months,   # (1180,)
    age_group=age_group,             # (1180,)
)
print(f"\nSaved to {out_path}")
