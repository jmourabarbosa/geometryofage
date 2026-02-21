"""
Load ODR data from .mat files and extract neuron metadata
from the MATLAB workspace byte stream.
"""

import scipy.io as sio
import numpy as np


def load_odr_data(mat_path):
    """
    Load the ODR .mat file.

    Returns
    -------
    odr_data : ndarray, shape (n_neurons, 8), dtype=object
        Each cell is a struct array of trials.
    workspace : bytes
        Raw workspace bytes (contains neuron_info table).
    """
    mat = sio.loadmat(mat_path, squeeze_me=False)
    odr_data = mat["odr_data_new"]
    workspace = mat["__function_workspace__"].tobytes()
    return odr_data, workspace


def load_odrd_data(mat_path):
    """
    Load the ODRd (distractor task) .mat file.

    Returns
    -------
    odrd_data : ndarray, shape (n_neurons, 4), dtype=object
    workspace : bytes
    """
    mat = sio.loadmat(mat_path, squeeze_me=False)
    odrd_data = mat["odrd_data_new"]
    workspace = mat["__function_workspace__"].tobytes()
    return odrd_data, workspace


def split_odrd_by_distractor(odrd_data):
    """
    Split ODRd data from (n_neurons, 4) to (n_neurons, 20).

    Each of the 4 cue-direction columns contains trials with 5 distractor
    locations encoded in OriginalClass (1-5, 6-10, 11-15, 16-20).
    Returns one column per OriginalClass value.
    """
    n_neurons = odrd_data.shape[0]
    new_data = np.empty((n_neurons, 20), dtype=object)

    for i in range(n_neurons):
        for c in range(4):
            cell = odrd_data[i, c]

            if cell is None or (hasattr(cell, 'size') and cell.size == 0):
                for s in range(5):
                    new_data[i, c * 5 + s] = None
                continue

            flat = cell.flatten()
            oc_raw = cell["OriginalClass"].flatten()
            oc_vals = np.array([np.asarray(x).flat[0] for x in oc_raw])

            for s in range(5):
                target_oc = c * 5 + s + 1
                mask = oc_vals == target_oc
                if mask.sum() > 0:
                    new_data[i, c * 5 + s] = flat[mask]
                else:
                    new_data[i, c * 5 + s] = None

    return new_data


def extract_metadata(workspace, n_neurons):
    """
    Extract neuron metadata from the MATLAB workspace bytes.

    Automatically detects byte offsets for neuron_age and mature_age
    by scanning for miDOUBLE arrays of the expected length.

    Parameters
    ----------
    workspace : bytes
    n_neurons : int

    Returns
    -------
    ids : ndarray of str, shape (n_neurons,)
    neuron_age : ndarray, shape (n_neurons,)
    mature_age : ndarray, shape (n_neurons,)
    delay_duration : ndarray, shape (n_neurons,) or None
        None if not found (ODRd has no delay_duration field).
    """
    import struct

    ids = _extract_ids(workspace, n_neurons)

    # Find all miDOUBLE arrays with exactly n_neurons elements
    expected_size = n_neurons * 8
    offsets = []
    for i in range(len(workspace) - 8):
        tag = struct.unpack_from('<II', workspace, i)
        if tag[0] == 9 and tag[1] == expected_size:
            offsets.append(i)

    # Identify neuron_age: first array with large negative values (days from maturation)
    neuron_age = None
    mature_age = None
    delay_duration = None

    for off in offsets:
        vals = _read_doubles(workspace, offset=off, n=n_neurons)
        if np.any(np.isnan(vals)):
            continue
        vmin, vmax, nuniq = vals.min(), vals.max(), len(np.unique(vals))

        if neuron_age is None and vmin < -100 and vmax > 100 and nuniq > 50:
            neuron_age = vals
        elif neuron_age is not None and mature_age is None and vmin > 1000 and nuniq <= 8:
            mature_age = vals
        elif mature_age is not None and delay_duration is None:
            unique_vals = set(np.unique(vals))
            if unique_vals <= {0.0, 1.5, 3.0}:
                delay_duration = vals

    assert neuron_age is not None, "Could not find neuron_age in workspace"
    assert mature_age is not None, "Could not find mature_age in workspace"
    return ids, neuron_age, mature_age, delay_duration


# ── Helpers ───────────────────────────────────────────────────────────────────

MONKEY_NAMES = ["OLI", "PIC", "QUA", "ROS", "SON", "TRI", "UNI", "VIK"]


def _extract_ids(workspace, n_neurons):
    """Find all monkey ID strings (UTF-16-LE) and return them in order."""
    hits = []
    for name in MONKEY_NAMES:
        encoded = name.encode("utf-16-le")
        start = 0
        while True:
            pos = workspace.find(encoded, start)
            if pos == -1:
                break
            hits.append((pos, name))
            start = pos + len(encoded)
    hits.sort()
    ids = np.array([h[1] for h in hits])
    assert len(ids) == n_neurons, (
        f"Expected {n_neurons} IDs, found {len(ids)}"
    )
    return ids


def _read_doubles(workspace, offset, n):
    """Read n doubles from workspace, skipping 8-byte MATLAB element header."""
    start = offset + 8
    end = start + n * 8
    return np.frombuffer(workspace[start:end], dtype="<f8").copy()


def _abs_age_months(age_days, mature_days):
    """Convert age-from-maturation and maturation date to absolute age in months."""
    return (age_days + mature_days) / 365.0 * 12.0


def load_all_task_data(data_dir):
    """Load ODR and ODRd data, split by task, compute absolute ages.

    Parameters
    ----------
    data_dir : str
        Path to directory containing .mat files.

    Returns
    -------
    task_data : dict
        {task_name: dict(data, ids, abs_age)} for 'ODR 1.5s', 'ODR 3.0s', 'ODRd'.
    """
    import os

    odr_data_all, ws_odr = load_odr_data(
        os.path.join(data_dir, 'odr_data_both_sig_is_best_20240109.mat'))
    ids_all, age_all, mature_all, delay_all = extract_metadata(
        ws_odr, odr_data_all.shape[0])

    odrd_raw, ws_odrd = load_odrd_data(
        os.path.join(data_dir, 'odrd_data_sig_on_best_20231018.mat'))
    odrd_ids, odrd_age, odrd_mat, _ = extract_metadata(ws_odrd, odrd_raw.shape[0])
    odrd_data = split_odrd_by_distractor(odrd_raw)

    task_data = {}
    for delay, name in [(1.5, 'ODR 1.5s'), (3.0, 'ODR 3.0s')]:
        mask = delay_all == delay
        task_data[name] = dict(
            data=odr_data_all[mask],
            ids=ids_all[mask],
            abs_age=_abs_age_months(age_all[mask], mature_all[mask]),
        )

    task_data['ODRd'] = dict(
        data=odrd_data,
        ids=odrd_ids,
        abs_age=_abs_age_months(odrd_age, odrd_mat),
    )

    return task_data
