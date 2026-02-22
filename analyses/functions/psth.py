"""
Compute PSTHs and tuning curves from raw spike data.
"""

import numpy as np


def compute_single_trial_rates(odr_data, bin_edges):
    """
    Compute single-trial firing rates for each neuron and condition.

    For each trial, spike times are aligned to cue onset and binned.

    Parameters
    ----------
    odr_data : ndarray, shape (n_neurons, n_conditions), dtype=object
    bin_edges : ndarray, shape (n_bins + 1,)
        Time bin edges in milliseconds.

    Returns
    -------
    rates : list of lists
        rates[neuron][condition] = ndarray (n_trials, n_bins) in Hz.
    """
    n_neurons, n_conditions = odr_data.shape
    bin_width_s = (bin_edges[1] - bin_edges[0]) / 1000.0
    n_bins = len(bin_edges) - 1

    rates = []
    for i in range(n_neurons):
        neuron_rates = []
        for c in range(n_conditions):
            cell = odr_data[i, c]

            if cell is None or cell.size == 0:
                neuron_rates.append(np.empty((0, n_bins)))
                continue

            n_trials = max(cell.shape)
            ts_arr = cell["TS"].flatten()
            cue_arr = cell["Cue_onT"].flatten()

            trial_rates = []
            for t in range(n_trials):
                ts = _flatten(ts_arr[t])
                cue = _flatten_scalar(cue_arr[t])

                if ts.size > 0:
                    aligned_ms = (ts - cue) * 1000.0
                    hist, _ = np.histogram(aligned_ms, bins=bin_edges)
                else:
                    hist = np.zeros(n_bins)

                trial_rates.append(hist / bin_width_s)

            neuron_rates.append(np.array(trial_rates))
        rates.append(neuron_rates)

        if i % 500 == 0:
            print(f"  neuron {i}/{n_neurons}")

    return rates


def compute_tuning_curves(rates, bin_centers, epoch_windows):
    """
    Average firing rates within time epochs to get tuning curves.

    Parameters
    ----------
    rates : list of lists
        rates[neuron][condition] = ndarray (n_trials, n_bins).
    bin_centers : ndarray, shape (n_bins,)
    epoch_windows : dict
        e.g. {"cue": (0, 500), "delay": (500, 2000)}

    Returns
    -------
    tuning : ndarray, shape (n_neurons, n_conditions, n_epochs)
        Trial-averaged firing rate per neuron, condition, epoch.
    epoch_names : list of str
    """
    n_neurons = len(rates)
    n_conditions = len(rates[0])
    epoch_names = list(epoch_windows.keys())
    n_epochs = len(epoch_names)

    # Precompute bin masks for each epoch
    epoch_masks = {}
    for name, (t0, t1) in epoch_windows.items():
        epoch_masks[name] = (bin_centers >= t0) & (bin_centers < t1)

    tuning = np.full((n_neurons, n_conditions, n_epochs), np.nan)

    for i in range(n_neurons):
        for c in range(n_conditions):
            trial_data = rates[i][c]  # (n_trials, n_bins)
            if trial_data.shape[0] == 0:
                continue
            # Average across trials first
            mean_rate = np.nanmean(trial_data, axis=0)  # (n_bins,)
            # Then average within each epoch
            for ei, name in enumerate(epoch_names):
                tuning[i, c, ei] = np.nanmean(mean_rate[epoch_masks[name]])

    return tuning, epoch_names


def compute_flat_tuning(data, t_range, epochs, bin_ms=50):
    """Compute flattened tuning curves from raw spike data.

    bins -> compute_single_trial_rates -> compute_tuning_curves -> flatten.

    Parameters
    ----------
    data : ndarray, shape (n_neurons, n_conditions), dtype=object
    t_range : tuple
        (start_ms, end_ms) for bin edges.
    epochs : dict
        e.g. {"cue": (0, 500), "delay": (500, 2000)}.
    bin_ms : int
        Bin width in milliseconds.

    Returns
    -------
    flat : ndarray, shape (n_neurons, n_conditions * n_epochs)
    rates : list of lists
    bc : ndarray
        Bin centers.
    """
    bins = np.arange(t_range[0], t_range[1] + bin_ms, bin_ms)
    bc = (bins[:-1] + bins[1:]) / 2.0
    rates = compute_single_trial_rates(data, bins)
    tuning, _ = compute_tuning_curves(rates, bc, epochs)
    flat = tuning.reshape(tuning.shape[0], -1)
    return flat, rates, bc


def pooled_tuning_by_group(task_data_dict, epochs, age_edges, bin_ms=25):
    """Compute tuning curves pooled across tasks, grouped by monkey x age group.

    Parameters
    ----------
    task_data_dict : dict
        {task_name: dict(data, ids, abs_age)}
        ``data`` should already be filtered to the common conditions
        (e.g. 4 cardinal directions).
    epochs : dict
        e.g. {'cue': (0, 500), 'delay': (500, 1700)}.
    age_edges : tuple
        Bin edges for age groups (passed to assign_age_groups).
    bin_ms : int

    Returns
    -------
    grouped : dict
        {monkey: {age_group: ndarray (n_neurons, n_conditions, n_epochs)}}
    epoch_names : list of str
    """
    from .analysis import assign_age_groups

    # Determine time range from epoch bounds
    t_min = min(t0 for t0, _ in epochs.values())
    t_max = max(t1 for _, t1 in epochs.values())
    # Add pre-cue buffer for binning
    t_range = (min(t_min, -500), t_max + bin_ms)

    bins = np.arange(t_range[0], t_range[1] + bin_ms, bin_ms)
    bc = (bins[:-1] + bins[1:]) / 2.0

    # Compute tuning per task, collect with monkey/group labels
    all_tuning = []
    all_ids = []
    all_groups = []

    for task_name, td in task_data_dict.items():
        rates = compute_single_trial_rates(td['data'], bins)
        tuning, epoch_names = compute_tuning_curves(rates, bc, epochs)
        ag = assign_age_groups(td['abs_age'], age_edges)

        all_tuning.append(tuning)
        all_ids.append(td['ids'])
        all_groups.append(ag)

    all_tuning = np.concatenate(all_tuning, axis=0)
    all_ids = np.concatenate(all_ids)
    all_groups = np.concatenate(all_groups)

    # Group by monkey x age group
    grouped = {}
    for mid in sorted(set(all_ids)):
        grouped[mid] = {}
        for g in sorted(set(all_groups)):
            mask = (all_ids == mid) & (all_groups == g)
            if mask.sum() == 0:
                continue
            grouped[mid][g] = all_tuning[mask]

    return grouped, epoch_names


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten(x):
    if hasattr(x, "flatten"):
        return x.flatten()
    return np.array([x])


def _flatten_scalar(x):
    if hasattr(x, "flatten"):
        x = x.flatten()
        return x[0] if x.size > 0 else x
    return x
