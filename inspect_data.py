"""
Inspect the ODR dataset: count neurons, sessions, and trials per subject.
Loads the primary .mat file and combines with exported CSV metadata.

Paper: Zhu et al. (2024) - https://pubmed.ncbi.nlm.nih.gov/39229176/
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")
GAM_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAM", "data")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW .MAT FILE  (odr_data_new: 2131 neurons × 8 cue directions)
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading odr_data_both_sig_is_best_20240109.mat ...")
mat = sio.loadmat(
    os.path.join(DATA_DIR, "odr_data_both_sig_is_best_20240109.mat"),
    squeeze_me=False,
)
odr_data = mat["odr_data_new"]         # shape (2131, 8), dtype=object
n_neurons_raw = odr_data.shape[0]
n_classes = odr_data.shape[1]          # 8 cue directions

print(f"  odr_data_new shape: {odr_data.shape}  "
      f"({n_neurons_raw} neurons × {n_classes} cue directions)")

# NOTE: neuron_info is a MATLAB table stored as an opaque object that scipy
# cannot deserialise.  We recover metadata from the exported CSV files instead.

# ═══════════════════════════════════════════════════════════════════════════════
# 2. COUNT TRIALS PER NEURON  (directly from the .mat struct arrays)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nCounting trials per neuron from odr_data_new ...")
trials_per_neuron = np.zeros(n_neurons_raw, dtype=int)
trials_per_class  = np.zeros((n_neurons_raw, n_classes), dtype=int)

for i in range(n_neurons_raw):
    for c in range(n_classes):
        cell = odr_data[i, c]
        if cell is None or (isinstance(cell, np.ndarray) and cell.size == 0):
            trials_per_class[i, c] = 0
        else:
            # struct array: number of trials = number of elements
            trials_per_class[i, c] = max(cell.shape)
    trials_per_neuron[i] = trials_per_class[i].sum()

print(f"  Trials/neuron: mean={trials_per_neuron.mean():.1f}  "
      f"median={np.median(trials_per_neuron):.0f}  "
      f"min={trials_per_neuron.min()}  max={trials_per_neuron.max()}")
print(f"  Total trials across all neurons: {trials_per_neuron.sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOAD PER-NEURON CSV  (contains subject ID, age, session info)
#    tau_fix_all_trial_all_neuron.csv has one row per neuron (2102 neurons)
#    This is the unfiltered dataset (before selecting sig neurons)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PER-NEURON METADATA  (from tau_fix_all_trial_all_neuron.csv)")
print("=" * 80)
tau_df = pd.read_csv(os.path.join(GAM_DIR, "tau_fix_all_trial_all_neuron.csv"))
print(f"  Rows (neurons): {len(tau_df)}")
print(f"  Columns: {list(tau_df.columns)}")
print(f"  Subjects: {sorted(tau_df['ID'].unique())}")
print(f"\n  Neurons per subject:")
for subj in sorted(tau_df["ID"].unique()):
    n = (tau_df["ID"] == subj).sum()
    ages = tau_df.loc[tau_df["ID"] == subj, "age"]
    mat_ages = tau_df.loc[tau_df["ID"] == subj, "mature"]
    print(f"    {subj:5s}: {n:5d} neurons  "
          f"(age: {ages.min():.1f}–{ages.max():.1f} months, "
          f"maturation-aligned: {mat_ages.min():.1f} to {mat_ages.max():.1f} months)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SESSION-LEVEL DATA  (rate_sess_all_trial_all_neuron_odr.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SESSION-LEVEL DATA  (from rate_sess_all_trial_all_neuron_odr.csv)")
print("=" * 80)
sess_df = pd.read_csv(os.path.join(GAM_DIR, "rate_sess_all_trial_all_neuron_odr.csv"))
print(f"  Total sessions: {len(sess_df)}")
print(f"  Subjects: {sorted(sess_df['ID'].unique())}")

print(f"\n  Sessions per subject:")
for subj in sorted(sess_df["ID"].unique()):
    sub = sess_df[sess_df["ID"] == subj]
    ages = sub["age"]
    print(f"    {subj:5s}: {len(sub):4d} sessions  "
          f"(age: {ages.min():.1f}–{ages.max():.1f} months)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. BEHAVIOURAL DATA  (beh_data.csv - ODR task performance)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("BEHAVIOURAL DATA  (from behavior/beh_data.csv)")
print("=" * 80)
beh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behavior", "beh_data.csv")
beh_df = pd.read_csv(beh_path)
print(f"  Total behavioural sessions: {len(beh_df)}")
print(f"  Subjects: {sorted(beh_df['Monkey'].unique())}")
print(f"  Tasks: {sorted(beh_df['Task'].unique())}")

print(f"\n  Behavioural sessions per subject × task:")
for subj in sorted(beh_df["Monkey"].unique()):
    sub = beh_df[beh_df["Monkey"] == subj]
    tasks = sub["Task"].value_counts().to_dict()
    ages_m = sub["age"] / 365.0 * 12.0   # 'age' column is in days
    print(f"    {subj:5s}: {len(sub):4d} sessions  tasks={tasks}  "
          f"(age: {ages_m.min():.1f}–{ages_m.max():.1f} months)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ODR-DISTRACTOR DATASET
# ═══════════════════════════════════════════════════════════════════════════════
odrd_path = os.path.join(DATA_DIR, "odrd_data_sig_on_best_20231018.mat")
if os.path.exists(odrd_path):
    print("\n" + "=" * 80)
    print("ODR-DISTRACTOR DATASET  (odrd_data_sig_on_best_20231018.mat)")
    print("=" * 80)
    mat2 = sio.loadmat(odrd_path, squeeze_me=False)
    odrd_keys = [k for k in mat2 if not k.startswith("__")]
    # find the data array
    for k in odrd_keys:
        v = mat2[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.ndim == 2:
            print(f"  {k}: {v.shape[0]} neurons × {v.shape[1]} conditions")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PEV DATA  (per-neuron, has more metadata)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PEV (Population Explained Variance) per neuron")
print("=" * 80)
for label, fn in [("Cue period (all)", "PEV_cue_all.csv"),
                  ("Delay 3s (all)", "PEV_del3_all.csv")]:
    path = os.path.join(GAM_DIR, fn)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n  {label} ({fn}):")
        print(f"    Rows: {len(df)}, Columns: {list(df.columns)}")
        if "ID" in df.columns:
            for subj in sorted(df["ID"].unique()):
                n = (df["ID"] == subj).sum()
                print(f"      {subj}: {n} entries")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. TUNING WIDTH DATA  (per-neuron, significant neurons only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TUNING WIDTH per neuron (significant neurons only)")
print("=" * 80)
for label, fn in [("Cue-responsive", "tuning_width_cue_all_neuron_with_r2_fixrate.csv"),
                  ("Delay-responsive", "tuning_width_del_all_neuron_with_r2_fixrate.csv")]:
    path = os.path.join(GAM_DIR, fn)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n  {label} ({fn}):")
        print(f"    Neurons: {len(df)}")
        if "ID" in df.columns:
            for subj in sorted(df["ID"].unique()):
                n = (df["ID"] == subj).sum()
                print(f"      {subj}: {n} neurons")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. GRAND SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("GRAND SUMMARY")
print("=" * 80)

# Unique subjects across all data sources
all_subjects = set()
all_subjects.update(tau_df["ID"].unique())
all_subjects.update(sess_df["ID"].unique())
all_subjects.update(beh_df["Monkey"].unique())

print(f"\n  Subjects (all sources): {sorted(all_subjects)}")
print(f"\n  Raw .mat file:")
print(f"    odr_data_new: {n_neurons_raw} neurons × {n_classes} cue directions")
print(f"    Total trials: {trials_per_neuron.sum():,}")
print(f"    Trials/neuron: {trials_per_neuron.mean():.1f} ± {trials_per_neuron.std():.1f}")

print(f"\n  Per-neuron CSV (tau): {len(tau_df)} neurons")
print(f"  Session-level CSV: {len(sess_df)} sessions")
print(f"  Behavioural CSV: {len(beh_df)} sessions")

print(f"\n  Neuron & session counts per subject:")
print(f"  {'Subject':>8s}  {'Neurons':>8s}  {'Sessions':>9s}  {'Beh.Sess':>9s}  {'Age range (months)':>20s}")
for subj in sorted(all_subjects):
    n_neur = (tau_df["ID"] == subj).sum() if subj in tau_df["ID"].values else 0
    n_sess = (sess_df["ID"] == subj).sum() if subj in sess_df["ID"].values else 0
    n_beh  = (beh_df["Monkey"] == subj).sum() if subj in beh_df["Monkey"].values else 0

    # age range in months (neural CSVs already in months; beh CSV is in days)
    ages_months = []
    if subj in tau_df["ID"].values:
        ages_months.extend(tau_df.loc[tau_df["ID"] == subj, "age"].tolist())
    if subj in sess_df["ID"].values:
        ages_months.extend(sess_df.loc[sess_df["ID"] == subj, "age"].tolist())
    if subj in beh_df["Monkey"].values:
        # beh_data.csv 'age' is in days → convert to months
        ages_months.extend(
            (beh_df.loc[beh_df["Monkey"] == subj, "age"] / 365.0 * 12.0).tolist()
        )

    if ages_months:
        age_str = f"{min(ages_months):.1f} – {max(ages_months):.1f}"
    else:
        age_str = "N/A"

    print(f"  {subj:>8s}  {n_neur:>8d}  {n_sess:>9d}  {n_beh:>9d}  {age_str:>20s}")

print("\nDone.")
