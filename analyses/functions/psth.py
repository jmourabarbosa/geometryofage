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
            mean_rate = trial_data.mean(axis=0)  # (n_bins,)
            # Then average within each epoch
            for ei, name in enumerate(epoch_names):
                tuning[i, c, ei] = mean_rate[epoch_masks[name]].mean()

    return tuning, epoch_names


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
