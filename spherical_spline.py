"""Cached spherical splines for reference estimation and exact local repair."""

import mne
import numpy as np
from mne.bem import _check_origin
from mne.channels.interpolation import _calc_g
from scipy.linalg import solve

from flatline import _flatline_configurations, _validate_flatlines


def _spline_geometry(raw):
    """Build the full EEG kernel once, keeping indices in original Raw order."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    positions = np.array([raw.info["chs"][i]["loc"][:3] for i in picks], dtype=float)
    valid = np.isfinite(positions).all(axis=1) & (np.linalg.norm(positions, axis=1) > 0)
    if not valid.all():
        names = [raw.ch_names[i] for i in picks[~valid]]
        raise ValueError(f"EEG channels need valid sensor positions; set a montage first: {names}.")
    with mne.utils.use_log_level("warning"):
        positions -= _check_origin("auto", raw.info)
    norms = np.linalg.norm(positions, axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("EEG sensor positions must differ from the fitted sphere origin.")
    positions /= norms[:, None]
    row_of = np.full(len(raw.ch_names), -1, dtype=int)
    row_of[picks] = np.arange(len(picks))
    return {"picks": picks, "row_of": row_of, "positions": positions,
            "kernel": _calc_g(positions @ positions.T), "weights": {}, "mean_weights": {}}


def _spline_weights(geometry, donors, targets):
    """Map genuine donor channels to targets; both use full Raw channel indices."""
    donors, targets = tuple(donors), tuple(targets)
    key = donors, targets
    if key not in geometry["weights"]:
        if not donors:
            raise ValueError("At least one genuine EEG donor is required for spherical interpolation.")
        donor_rows = geometry["row_of"][list(donors)]
        target_rows = geometry["row_of"][list(targets)]
        if (donor_rows < 0).any() or (target_rows < 0).any():
            raise ValueError("Spherical interpolation donors and targets must be EEG channels.")
        n = len(donors)
        if targets:
            system = np.ones((n + 1, n + 1), dtype=float)
            system[:n, :n] = geometry["kernel"][np.ix_(donor_rows, donor_rows)]
            system[np.arange(n), np.arange(n)] += 1e-5
            system[-1, -1] = 0
            evaluation = np.ones((len(targets), n + 1), dtype=float)
            evaluation[:, :n] = geometry["kernel"][np.ix_(target_rows, donor_rows)]
            weights = solve(system, evaluation.T, assume_a="sym").T[:, :n]
        else:
            weights = np.empty((0, n), dtype=float)
        geometry["weights"][key] = weights
    return geometry["weights"][key]


def _mean_configuration(geometry, invalid):
    """Cache the projector for the mean of measured donors and repaired targets."""
    invalid = tuple(invalid)
    if invalid not in geometry["mean_weights"]:
        donors = geometry["picks"][~np.isin(geometry["picks"], invalid)]
        weights = _spline_weights(geometry, donors, invalid)
        coefficients = (1 + weights.sum(axis=0)) / len(geometry["picks"])
        geometry["mean_weights"][invalid] = donors, coefficients
    return geometry["mean_weights"][invalid]


def _flatline_aware_mean(raw, bads, flatlines=None, geometry=None):
    """Mean of the full EEG reconstruction without making an interpolated Raw copy.

    Whole-channel bads are repaired throughout the record. Local flat channels
    join them only in their exact intervals outside global BAD annotations.
    Every configuration reads its genuine donors from the original recording.
    """
    # Bound temporary donor-data copies (8 MiB for 128 float64 channels).
    sample_chunk = 8192
    if not raw.preload:
        raise ValueError("raw must be preloaded.")
    _validate_flatlines(raw, flatlines)
    if geometry is None:
        geometry = _spline_geometry(raw)
    bads = tuple(i for i in geometry["picks"] if raw.ch_names[i] in bads)
    reference = np.empty(raw.n_times, dtype=float)
    donors, coefficients = _mean_configuration(geometry, bads)
    for start in range(0, raw.n_times, sample_chunk):
        stop = min(start + sample_chunk, raw.n_times)
        reference[start:stop] = coefficients @ raw._data[donors, start:stop]
    for start, stop, flat in _flatline_configurations(raw, flatlines):
        if flat:
            donors, coefficients = _mean_configuration(
                geometry, np.union1d(np.asarray(bads, dtype=int), flat)
            )
            for first in range(start, stop, sample_chunk):
                last = min(first + sample_chunk, stop)
                reference[first:last] = coefficients @ raw._data[donors, first:last]
    return reference


def _interpolate_bads_and_flatlines_inplace(raw, bads, flatlines=None, geometry=None):
    """Repair whole bad channels and exact local spans without changing metadata.

    Apply the same configurations used by _flatline_aware_mean, before the
    caller subtracts that mean. Previously repaired channels are never donors
    for another repair in the same interval.
    """
    # Bound temporary donor-data copies (8 MiB for 128 float64 channels).
    sample_chunk = 8192
    if not raw.preload:
        raise ValueError("raw must be preloaded.")
    _validate_flatlines(raw, flatlines)
    if geometry is None:
        geometry = _spline_geometry(raw)
    bads = tuple(i for i in geometry["picks"] if raw.ch_names[i] in bads)
    if bads:
        donors, _ = _mean_configuration(geometry, bads)
        weights = _spline_weights(geometry, donors, bads)
        for start in range(0, raw.n_times, sample_chunk):
            stop = min(start + sample_chunk, raw.n_times)
            raw._data[list(bads), start:stop] = weights @ raw._data[donors, start:stop]
    for start, stop, flat in _flatline_configurations(raw, flatlines):
        if flat:
            invalid = np.union1d(np.asarray(bads, dtype=int), flat)
            donors, _ = _mean_configuration(geometry, invalid)
            weights = _spline_weights(geometry, donors, invalid)
            for first in range(start, stop, sample_chunk):
                last = min(first + sample_chunk, stop)
                raw._data[invalid, first:last] = weights @ raw._data[donors, first:last]
