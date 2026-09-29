"""ELBO step 2: bad-channel and bad-time detection.

Bad-channel criteria may be local or cross-channel, and time invariant or
time dependent. Bad-time detection and iteration between the two parts will
be developed after the initial bad-channel criteria.

Each bad_ch_by_* function adds its detections to raw.info["bads"] without
duplicates and returns {criterion: [channel names]} for that call's detections.
Signal data and input annotations are unchanged.

The 2-second window detectors accept return_details=True to return
(report, details), including window metrics and original flagged waveforms.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter
from step1_line_noise_removal import Interpolate_fft, remove_line_noise_inplace


# %% Part 1: Bad-channel detection

# Local, time-invariant criteria
def bad_ch_by_impedance(
    raw,
    initial_impedance,
    end_impedance=None,
    reject_by="initial",
    imp_cutoff=500,
):
    """Return {"impedance": names} for channels exceeding imp_cutoff.

    Impedances and imp_cutoff are in kOhm. Each supplied list must contain one
    value per channel, in raw.ch_names order, including non-EEG and existing
    bad channels. Values must be finite and nonnegative, or NaN. Initial
    readings are required; end readings are optional unless reject_by is
    "end" or "both".

    "initial" uses only initial readings; "end" uses only end readings;
    "both" flags a channel if either reading exceeds the cutoff. NaN readings
    (e.g., for the reference channel) are ignored; a valid reading can still
    flag a channel in "both" mode. A reading equal to the cutoff is accepted.
    Return only this criterion's detections, in Raw channel order. These
    flags apply to the entire recording and are added to raw.info["bads"],
    preserving existing labels without duplicates.
    """
    if reject_by not in ("initial", "end", "both"):
        raise ValueError('reject_by must be "initial", "end", or "both".')
    if not np.isfinite(imp_cutoff) or imp_cutoff <= 0:
        raise ValueError("imp_cutoff must be finite and greater than 0 (kOhm).")
    if reject_by in ("end", "both") and end_impedance is None:
        raise ValueError(f'end_impedance is required for reject_by="{reject_by}".')

    readings = {"initial": initial_impedance}
    if end_impedance is not None:
        readings["end"] = end_impedance
    for name, values in readings.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (len(raw.ch_names),):
            raise ValueError(
                f"{name}_impedance must be a 1D list with one value per "
                f"Raw channel ({len(raw.ch_names)} values)."
            )
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(
                f"{name}_impedance must contain finite, nonnegative values "
                "(kOhm) or NaN."
            )
        readings[name] = values

    if reject_by == "both":
        bad = (readings["initial"] > imp_cutoff) | (readings["end"] > imp_cutoff)
    else:
        bad = readings[reject_by] > imp_cutoff
    bad_names = [name for name, flagged in zip(raw.ch_names, bad) if flagged]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    return {"impedance": bad_names}


def bad_ch_by_nan(raw):
    """Return {"nan": names} for every channel with any NaN or +/-inf sample.

    Check all channel types and all samples, including annotated spans and
    the reference channel. Add detections to raw.info["bads"], preserving
    existing labels without duplicates. Data and annotations are unchanged.
    """
    bad_names = []
    for i, name in enumerate(raw.ch_names):
        data = raw.get_data(picks=[i], reject_by_annotation=None)
        if not np.isfinite(data).all():
            bad_names.append(name)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    return {"nan": bad_names}


def _check_average_reference(raw):
    """Require the non-bad EEG mean to be within 0.1 nV of zero at every sample."""
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("No non-bad EEG channels remain to check the average reference.")
    mean = np.zeros(raw.n_times)
    for i in picks:
        data = raw.get_data(picks=[i], reject_by_annotation=None)[0]
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite data in eligible channels; run bad_ch_by_nan first.")
        mean += data / len(picks)
    if not np.allclose(mean, 0, rtol=0, atol=1e-10):
        raise ValueError(
            "Non-bad EEG channels are not common-average referenced: their sample-wise "
            f"mean reaches {np.max(np.abs(mean)) * 1e6:g} uV (tolerance 0.0001 uV). "
            "Apply the average reference over the current non-bad EEG channels."
        )


def _resolve_reference_channel(raw, reference_channel, *, require_reference=False):
    """Return an all-zero EEG reference, or None after verifying average reference.

    A supplied name limits the zero-signal search to that EEG channel. If it
    is no longer zero (e.g., after re-referencing), check the average reference
    instead and do not exclude it. Without a name, search all EEG channels,
    including existing bads. require_reference keeps flatline detection strict.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if reference_channel is not None:
        if reference_channel not in [raw.ch_names[i] for i in eeg_picks]:
            raise ValueError("reference_channel must name an EEG channel in raw.")
        eeg_picks = [raw.ch_names.index(reference_channel)]

    candidates = []
    for i in eeg_picks:
        data = raw.get_data(picks=[i], reject_by_annotation=None)
        if data.size and np.all(data == 0):
            candidates.append(raw.ch_names[i])
    if len(candidates) == 1:
        return candidates[0]
    if require_reference:
        raise ValueError("Expected exactly one entirely zero EEG reference channel; "
                         f"found {candidates}. Supply reference_channel explicitly.")
    _check_average_reference(raw)
    return None


def _prepare_channel_windows(
    raw, reference_channel, window_length, occurrence_cutoff, *,
    require_reference=False, require_average=False,
):
    """Select eligible EEG channels and complete, non-overlapping sample windows."""
    if not np.isfinite(window_length) or window_length <= 0:
        raise ValueError("window_length must be finite and greater than 0 (seconds).")
    if not np.isfinite(occurrence_cutoff) or not 0 <= occurrence_cutoff <= 1:
        raise ValueError("occurrence_cutoff must be between 0 and 1.")
    window_samples = int(round(window_length * raw.info["sfreq"]))
    if window_samples < 2:
        raise ValueError("window_length must cover at least two samples.")
    n_windows = raw.n_times // window_samples
    if n_windows == 0:
        raise ValueError("Data must contain at least one complete window.")
    if require_average:
        _check_average_reference(raw)
        reference_channel = None
    else:
        reference_channel = _resolve_reference_channel(
            raw, reference_channel, require_reference=require_reference
        )
    exclude = raw.info["bads"] + ([] if reference_channel is None else [reference_channel])
    picks = mne.pick_types(
        raw.info, eeg=True, exclude=exclude
    )
    if len(picks) < 2:
        raise ValueError("At least two non-bad EEG channels after reference exclusion "
                         "are required.")
    return picks, window_samples, n_windows


def _init_window_details(raw, picks, window_samples, n_windows, metric_names):
    """Allocate complete-window diagnostics without closing annotation gaps."""
    names = [raw.ch_names[i] for i in picks]
    details = {
        "ch_names": names,
        "start_s": np.arange(n_windows) * window_samples / raw.info["sfreq"],
        "stop_s": np.arange(1, n_windows + 1) * window_samples / raw.info["sfreq"],
        "retained": np.zeros(n_windows, dtype=bool),
        "flagged_data_uv": {name: [] for name in names},
    }
    for metric in ("violates", *metric_names):
        details[metric] = np.full((len(picks), n_windows), np.nan)
    return details


def _record_window_details(details, start, window_samples, data, violates):
    """Store this window's flags and violating waveforms; return its index."""
    window = start // window_samples
    details["retained"][window] = True
    details["violates"][:, window] = violates
    for i in np.flatnonzero(violates):
        details["flagged_data_uv"][details["ch_names"][i]].append(data[i] * 1e6)
    return window


def _finish_window_details(details, window_samples):
    """Convert waveform lists to (violating windows, samples) arrays."""
    details["flagged_data_uv"] = {
        name: np.asarray(windows).reshape(-1, window_samples)
        for name, windows in details["flagged_data_uv"].items()
    }
    return details


# Local signal properties with time-dependent occurrence criteria
def bad_ch_by_smallness(
    raw,
    window_length=2.0,
    std_cutoff=2.0,
    occurrence_cutoff=0.25,
    reference_channel=None,
    *,
    return_details=False,
):
    """Return {"smallness": names} for persistently low-variance EEG channels.

    Require the mean over currently non-bad EEG channels to be within 0.1 nV
    of zero at every sample, even if an all-zero native reference exists.
    Exclude existing bads and non-EEG channels. reference_channel is retained
    for call compatibility but does not bypass the check or exclude a channel.
    Use complete non-overlapping windows, starting at the first sample of
    this Raw; window_length is in seconds, rounded to samples. Ignore the trailing
    partial window; annotations do not exclude any samples.

    In each window, compute each channel's variance (ddof=0). Remove values
    strictly above mean + std_cutoff * std, then recompute mean/std on the
    retained variances (ddof=0). Flag retained values strictly below the new
    mean - std_cutoff * std. Upper-tail removal is a single pass and does not
    itself flag a channel. Return channels flagged in strictly more than
    occurrence_cutoff of complete windows, and add them to raw.info["bads"].
    Preserve existing labels without duplicates; data/annotations are unchanged.

    With return_details=True, return (report, details). Common fields follow
    bad_ch_by_HFnoise: ch_names, crop-relative start_s/stop_s, retained,
    violates, and flagged_data_uv. All complete windows are retained, even
    when annotated. variance_uv2 and baseline_retained (0/1) have shape
    (channels, windows); upper_cutoff_uv2 and lower_cutoff_uv2 have one value
    per window. Variances and cutoffs are in squared microvolts.
    """
    if not np.isfinite(std_cutoff) or std_cutoff <= 0:
        raise ValueError("std_cutoff must be finite and greater than 0.")
    picks, window_samples, n_windows = _prepare_channel_windows(
        raw, reference_channel, window_length, occurrence_cutoff, require_average=True
    )
    counts = np.zeros(len(picks), dtype=int)
    if return_details:
        details = _init_window_details(
            raw, picks, window_samples, n_windows, ("variance_uv2", "baseline_retained")
        )
        for metric in ("upper_cutoff_uv2", "lower_cutoff_uv2"):
            details[metric] = np.full(n_windows, np.nan)
    for start in range(0, n_windows * window_samples, window_samples):
        data = raw.get_data(
            picks=picks, start=start, stop=start + window_samples,
            reject_by_annotation=None,
        )
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite data in eligible channels; run bad_ch_by_nan first.")
        variances = data.var(axis=1)
        high_cutoff = variances.mean() + std_cutoff * variances.std()
        keep = variances <= high_cutoff
        retained = variances[keep]
        low_cutoff = retained.mean() - std_cutoff * retained.std()
        violates = keep & (variances < low_cutoff)
        counts += violates
        if return_details:
            window = _record_window_details(details, start, window_samples, data, violates)
            details["variance_uv2"][:, window] = variances * 1e12
            details["baseline_retained"][:, window] = keep
            details["upper_cutoff_uv2"][window] = high_cutoff * 1e12
            details["lower_cutoff_uv2"][window] = low_cutoff * 1e12

    bad_names = [raw.ch_names[i] for i in picks[counts / n_windows > occurrence_cutoff]]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"smallness": bad_names}
    if return_details:
        return report, _finish_window_details(details, window_samples)
    return report


def bad_ch_by_flatline(
    raw,
    window_length=2.0,
    occurrence_cutoff=0.25,
    reference_channel=None,
    *,
    return_details=False,
):
    """Return {"flatline": names} for persistently flat EEG channels.

    Require a supplied or uniquely detected all-zero EEG reference: run this
    before average referencing. Channel selection and complete-window rules
    otherwise follow bad_ch_by_smallness. A window is flat if all samples are
    exactly equal, including constant nonzero signals. Suppress its flag if
    at least 50% of the other eligible EEG channels are also flat in that
    window. The target channel is excluded from this comparison, as are
    existing bads/reference.

    Return channels flagged in strictly more than occurrence_cutoff of all
    complete windows, including suppressed windows in the denominator.
    Add detections to raw.info["bads"], preserving existing labels without
    duplicates. Data and annotations are unchanged.

    With return_details=True, return (report, details), with the common fields
    described in bad_ch_by_HFnoise. All complete windows are retained, even
    when annotated or suppressed. flat (0/1), other_flat_fraction, suppressed
    (0/1), and violates (0/1) have shape (channels, windows). A suppressed flag
    denotes a flat channel whose other_flat_fraction is at least 0.5.
    flagged_data_uv includes only windows where violates == 1.
    """
    picks, window_samples, n_windows = _prepare_channel_windows(
        raw, reference_channel, window_length, occurrence_cutoff, require_reference=True
    )
    counts = np.zeros(len(picks), dtype=int)
    if return_details:
        details = _init_window_details(
            raw, picks, window_samples, n_windows, ("flat", "other_flat_fraction", "suppressed")
        )
    for start in range(0, n_windows * window_samples, window_samples):
        data = raw.get_data(
            picks=picks, start=start, stop=start + window_samples,
            reject_by_annotation=None,
        )
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite data in eligible channels; run bad_ch_by_nan first.")
        flat = np.all(data == data[:, :1], axis=1)
        other_flat_count = flat.sum() - flat.astype(int)
        violates = flat & (other_flat_count < 0.5 * (len(picks) - 1))
        counts += violates
        if return_details:
            window = _record_window_details(details, start, window_samples, data, violates)
            details["flat"][:, window] = flat
            details["other_flat_fraction"][:, window] = other_flat_count / (len(picks) - 1)
            details["suppressed"][:, window] = flat & ~violates

    bad_names = [raw.ch_names[i] for i in picks[counts / n_windows > occurrence_cutoff]]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"flatline": bad_names}
    if return_details:
        return report, _finish_window_details(details, window_samples)
    return report


def mask_segment_by_voltage(
    raw,
    voltage_threshold_uv=500.0,
    channel_fraction_cutoff=0.25,
    window_length=0.1,
    reference_channel=None,
    *,
    require_average=False,
):
    """Append BAD_voltage_mask annotations to raw in place; return None.

    Exclude existing bads, non-EEG channels, and any all-zero reference. If
    no reference is identified, verify average reference as in bad_ch_by_smallness.
    With require_average=True, always verify average reference and do not
    exclude any channel by its reference name or all-zero signal.
    A channel exceeds the threshold in a window if any sample is strictly outside
    +/-voltage_threshold_uv. Annotate the entire window when strictly more
    than channel_fraction_cutoff of eligible channels exceed it, even if
    their exceedances are not simultaneous. Merge adjacent masked windows;
    include the final partial window with its actual duration.

    These annotations are temporary when called by the four noisy-channel
    detectors below, which pass a copy of their input Raw.
    """
    if not np.isfinite(voltage_threshold_uv) or voltage_threshold_uv <= 0:
        raise ValueError("voltage_threshold_uv must be finite and greater than 0.")
    if not np.isfinite(channel_fraction_cutoff) or not 0 <= channel_fraction_cutoff <= 1:
        raise ValueError("channel_fraction_cutoff must be between 0 and 1.")
    if not np.isfinite(window_length) or window_length <= 0:
        raise ValueError("window_length must be finite and greater than 0 (seconds).")
    sfreq = raw.info["sfreq"]
    window_samples = int(round(window_length * sfreq))
    if window_samples < 1:
        raise ValueError("window_length must cover at least one sample.")
    if require_average:
        _check_average_reference(raw)
        reference_channel = None
    else:
        reference_channel = _resolve_reference_channel(raw, reference_channel)
    exclude = raw.info["bads"] + ([] if reference_channel is None else [reference_channel])
    picks = mne.pick_types(
        raw.info, eeg=True, exclude=exclude
    )
    if len(picks) == 0:
        raise ValueError("No usable EEG channels remain after exclusions.")

    # Read one channel at a time; retain only counts for the short mask windows.
    starts = np.arange(0, raw.n_times, window_samples)
    counts = np.zeros(len(starts), dtype=int)
    for i in picks:
        data = raw.get_data(picks=[i], reject_by_annotation=None)[0]
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite data in eligible channels; run bad_ch_by_nan first.")
        exceeds = np.abs(data) > voltage_threshold_uv / 1e6
        counts += np.logical_or.reduceat(exceeds, starts)
    bad = counts / len(picks) > channel_fraction_cutoff
    edges = np.flatnonzero(np.diff(np.r_[False, bad, False]))
    spans = edges.reshape(-1, 2) * window_samples
    spans[:, 1] = np.minimum(spans[:, 1], raw.n_times)
    raw.annotations.append(
        raw.first_time + spans[:, 0] / sfreq,
        (spans[:, 1] - spans[:, 0]) / sfreq,
        ["BAD_voltage_mask"] * len(spans),
    )


def _temporary_voltage_mask(raw, reference_channel, *, require_average=False):
    """Mask a copy after checking for a native or average reference."""
    if require_average:
        _check_average_reference(raw)
        reference_channel = None
    else:
        reference_channel = _resolve_reference_channel(raw, reference_channel)
    masked = raw.copy()
    mask_segment_by_voltage(
        masked, reference_channel=reference_channel, require_average=require_average
    )
    return masked, reference_channel


def _unmasked_windows(raw, picks, window_samples, n_windows, *, return_start=False):
    """Yield complete windows with no BAD-annotated samples, without closing gaps."""
    for start in range(0, n_windows * window_samples, window_samples):
        data = raw.get_data(
            picks=picks, start=start, stop=start + window_samples,
            reject_by_annotation="omit", verbose=False,
        )
        if data.shape[1] != window_samples:
            continue
        yield (start, data) if return_start else data


def _iqr(values, axis=0):
    """IQR using MATLAB's midpoint (Hazen) quantile convention."""
    q25, q75 = np.percentile(values, [25, 75], axis=axis, method="hazen")
    return q75 - q25


def _mad(values, axis=0):
    """Unscaled median absolute deviation, as in MATLAB mad(values, 1)."""
    return np.median(np.abs(values - np.median(values, axis=axis, keepdims=True)), axis=axis)


def _robust_channel_z(values, use_iqr=False):
    """One cross-channel baseline per metric; floor a zero scale at machine epsilon."""
    center = np.median(values, axis=0)
    scale = 0.7413 * _iqr(values) if use_iqr else 1.4826 * _mad(values)
    return (values - center) / np.maximum(scale, np.finfo(float).eps)


def _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff):
    """Return flagged names after evaluating at least one retained window."""
    if n_retained == 0:
        raise ValueError("No complete unmasked windows remain for channel detection.")
    return [raw.ch_names[i] for i in picks[counts / n_retained > occurrence_cutoff]]


def bad_ch_by_voltage(
    raw,
    voltage_threshold_uv=200.0,
    occurrence_cutoff=0.25,
    reference_channel=None,
):
    """Return {"voltage": names} for excessive voltage outside temporary masks.

    Flag if strictly more than occurrence_cutoff of retained samples exceed
    +/-voltage_threshold_uv. Exclude existing bads, the reference, non-EEG
    channels, existing BAD annotations, and the temporary 0.1-second voltage
    mask. Add detections to raw.info["bads"], preserving existing labels
    without duplicates. Input data and annotations are unchanged.
    Accept an all-zero native reference, otherwise verify average reference
    as in bad_ch_by_smallness.

    Show all channels over the full time range, including existing bads,
    the reference, and masked samples. Plot in microvolts with dotted cutoff
    lines and a different color per channel. Highlight and label only this
    call's detections in the legend. Shade temporary voltage masks in light
    gray and scale the y-axis from finite samples outside all BAD annotations.
    """
    if not np.isfinite(voltage_threshold_uv) or voltage_threshold_uv <= 0:
        raise ValueError("voltage_threshold_uv must be finite and greater than 0.")
    if not np.isfinite(occurrence_cutoff) or not 0 <= occurrence_cutoff <= 1:
        raise ValueError("occurrence_cutoff must be between 0 and 1.")
    masked, reference_channel = _temporary_voltage_mask(raw, reference_channel)
    exclude = masked.info["bads"] + ([] if reference_channel is None else [reference_channel])
    picks = mne.pick_types(
        masked.info, eeg=True, exclude=exclude
    )
    bad_names = []
    for i in picks:
        data = masked.get_data(picks=[i], reject_by_annotation="omit", verbose=False)[0]
        if data.size == 0:
            raise ValueError("No unmasked samples remain for channel detection.")
        if np.mean(np.abs(data) > voltage_threshold_uv / 1e6) > occurrence_cutoff:
            bad_names.append(raw.ch_names[i])
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    times = raw.times
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(raw.ch_names)))
    ymin, ymax = np.inf, -np.inf
    for i, name in enumerate(raw.ch_names):
        data_uv = raw.get_data(picks=[i], reject_by_annotation=None)[0] * 1e6
        retained_uv = masked.get_data(
            picks=[i], reject_by_annotation="omit", verbose=False
        )[0] * 1e6
        retained_uv = retained_uv[np.isfinite(retained_uv)]
        if retained_uv.size:
            ymin = min(ymin, retained_uv.min())
            ymax = max(ymax, retained_uv.max())
        if name in bad_names:
            ax.plot(times, data_uv, color=colors[i], linewidth=0.8, label=name, zorder=2)
        else:
            ax.plot(times, data_uv, color=colors[i], linewidth=0.5, alpha=0.7, zorder=1)
    for annotation in masked.annotations:
        if annotation["description"] == "BAD_voltage_mask":
            start = annotation["onset"] - masked.first_time
            ax.axvspan(start, start + annotation["duration"], color="lightgray", zorder=0)
    padding = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - padding, ymax + padding)
    for cutoff in (-voltage_threshold_uv, voltage_threshold_uv):
        ax.axhline(cutoff, color="black", linestyle=":", linewidth=1, zorder=3)
    ax.set(xlabel="Time (s)", ylabel="Voltage (µV)", title="Bad channels by voltage")
    if bad_names:
        ax.legend(title="Flagged channels", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.show()
    return {"voltage": bad_names}


def bad_ch_by_std(
    raw,
    window_length=2.0,
    robust_z_cutoff=5.0,
    occurrence_cutoff=0.25,
    reference_channel=None,
    *,
    return_details=False,
):
    """Return {"std": names} for persistently high robust standard deviation.

    PREP findNoisyChannels.m, Method 1: d = 0.7413 * IQR over time;
    z = (d - median(d)) / (0.7413 * IQR(d)) across channels. Use z strictly
    above robust_z_cutoff, never abs(z). Compute a fresh baseline once per
    complete window; exclude existing bads and non-EEG channels. Require
    average reference as in bad_ch_by_smallness; reference_channel is ignored.

    Apply the temporary 0.1-second voltage mask on a copy. Omit every window
    overlapping any BAD annotation and the trailing partial window. Flag
    channels offending in strictly more than occurrence_cutoff of retained
    windows. Add detections to raw.info["bads"], preserving existing labels
    without duplicates. Input data and annotations are unchanged.

    With return_details=True, return (report, details), with the common fields
    described in bad_ch_by_HFnoise. robust_std_uv, robust_z, and violates have
    shape (channels, complete windows); masked windows contain NaN. The
    robust standard deviation is reported in microvolts. flagged_data_uv
    stores the original violating window waveforms in chronological order.
    """
    if not np.isfinite(robust_z_cutoff) or robust_z_cutoff <= 0:
        raise ValueError("robust_z_cutoff must be finite and greater than 0.")
    masked, reference_channel = _temporary_voltage_mask(
        raw, reference_channel, require_average=True
    )
    picks, window_samples, n_windows = _prepare_channel_windows(
        masked, reference_channel, window_length, occurrence_cutoff, require_average=True
    )
    counts = np.zeros(len(picks), dtype=int)
    n_retained = 0
    if return_details:
        details = _init_window_details(
            raw, picks, window_samples, n_windows, ("robust_std_uv", "robust_z")
        )
    for start, data in _unmasked_windows(
        masked, picks, window_samples, n_windows, return_start=True
    ):
        deviation = 0.7413 * _iqr(data, axis=1)
        z = _robust_channel_z(deviation, use_iqr=True)
        violates = z > robust_z_cutoff
        counts += violates
        n_retained += 1
        if return_details:
            window = _record_window_details(details, start, window_samples, data, violates)
            details["robust_std_uv"][:, window] = deviation * 1e6
            details["robust_z"][:, window] = z
    bad_names = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"std": bad_names}
    if return_details:
        return report, _finish_window_details(details, window_samples)
    return report


def _prep_hf_ratio(data, sfreq, *, return_mads=False):
    """PREP's MAD(residual)/MAD(lowpass), filtering this window in isolation.

    Port the 100th-order design_fir and filtfilt_fast FIR paths called by
    findNoisyChannels.m, Method 2: 45 Hz passband, 50 Hz stopband, cubic
    response interpolation, Hamming window, and forward/backward filtering.
    The implemented PREP filter is low-pass only, despite its 1 Hz comment.
    Floor zero low-pass MAD at machine epsilon to keep flat signals finite.
    With return_mads=True, also return the HF and LF MAD arrays, in input units.
    """
    order, n_fft = 100, 512
    frequencies = np.array([0, 45, 50, sfreq / 2]) / (sfreq / 2)
    bins = np.floor(frequencies * n_fft + 0.5).astype(int)
    if np.any(np.diff(bins) <= 0):
        raise ValueError("Sampling rate cannot resolve PREP's 45-50 Hz filter grid.")
    grid = np.arange(n_fft + 1)
    response = PchipInterpolator(bins, [1, 1, 0, 0])(grid)
    response = response * np.exp(-0.5j * order * np.pi * grid / n_fft)
    kernel = np.fft.irfft(response, n=2 * n_fft)[:order + 1] * np.hamming(order + 1)
    width, n_times = len(kernel), data.shape[1]
    left = 2 * data[:, :1] - data[:, np.arange(width, 0, -1) % n_times]
    right = 2 * data[:, -1:] - data[:, np.arange(n_times - 2, n_times - width - 2, -1) % n_times]
    padded = np.concatenate([left, data, right], axis=1)
    forward = lfilter(kernel, [1], padded, axis=1)
    lowpass = lfilter(kernel, [1], forward[:, ::-1], axis=1)[:, ::-1]
    lowpass = lowpass[:, width:width + n_times]
    hf_mad = _mad(data - lowpass, axis=1)
    lf_mad = _mad(lowpass, axis=1)
    ratios = hf_mad / np.maximum(lf_mad, np.finfo(float).eps)
    return (ratios, hf_mad, lf_mad) if return_mads else ratios


def bad_ch_by_HFnoise(
    raw,
    window_length=2.0,
    robust_z_cutoff=5.0,
    occurrence_cutoff=0.25,
    reference_channel=None,
    *,
    return_details=False,
):
    """Return {"HFnoise": names} for persistently high PREP HF-noise ratios.

    Use MAD(signal - lowpass) / MAD(lowpass) per channel, then subtract the
    cross-channel median and divide by 1.4826 * cross-channel MAD. A window
    violates when z > robust_z_cutoff. A channel must violate in strictly more
    than occurrence_cutoff of retained windows. Then compare the median HF
    and LF MADs in its violating (F) versus non-violating (N) windows: require
    log(H_F / H_N) > max(0, -log(L_F / L_N)). Numerical ties do not confirm
    the flag. Floor MADs at machine epsilon before taking log ratios.
    Keep candidates that violate in every retained window, since no
    non-violating baseline is available.
    Each retained window is filtered separately to keep masked bursts out of
    neighboring windows. Accept an all-zero native reference, otherwise verify
    average reference. Temporary masking and the occurrence denominator follow
    bad_ch_by_std. Require sfreq > 100 Hz
    to evaluate noise above 50 Hz. Add detections to raw.info["bads"], preserving
    existing labels without duplicates. Input data/annotations are unchanged.

    With return_details=True, return (report, details). Details contain channel
    names, crop-relative start/stop seconds, a retained-window mask, and arrays
    of shape (channels, complete windows): ratio, robust_z, hf_mad_uv, lf_mad_uv,
    and violates (0/1). Masked windows contain NaN, not a passing flag. For each
    channel, flagged_data_uv contains its original window waveforms in
    chronological order, matching the windows where violates == 1. The
    diagnostics describe this pass's channel set, before its new bad labels.
    Window violations remain the initial z flags, including for channels that
    do not pass the channel-level HF/LF confirmation.
    """
    if not np.isfinite(robust_z_cutoff) or robust_z_cutoff <= 0:
        raise ValueError("robust_z_cutoff must be finite and greater than 0.")
    if raw.info["sfreq"] <= 100:
        raise ValueError("bad_ch_by_HFnoise requires a sampling rate above 100 Hz.")
    masked, reference_channel = _temporary_voltage_mask(raw, reference_channel)
    picks, window_samples, n_windows = _prepare_channel_windows(
        masked, reference_channel, window_length, occurrence_cutoff
    )
    counts = np.zeros(len(picks), dtype=int)
    n_retained = 0
    hf_mads, lf_mads, window_violations = [], [], []
    if return_details:
        names = [raw.ch_names[i] for i in picks]
        details = {
            "ch_names": names,
            "start_s": np.arange(n_windows) * window_samples / raw.info["sfreq"],
            "stop_s": np.arange(1, n_windows + 1) * window_samples / raw.info["sfreq"],
            "retained": np.zeros(n_windows, dtype=bool),
            "flagged_data_uv": {name: [] for name in names},
        }
        for metric in ("ratio", "robust_z", "hf_mad_uv", "lf_mad_uv", "violates"):
            details[metric] = np.full((len(picks), n_windows), np.nan)
    for start, data in _unmasked_windows(
        masked, picks, window_samples, n_windows, return_start=True
    ):
        ratios, hf_mad, lf_mad = _prep_hf_ratio(data, raw.info["sfreq"], return_mads=True)
        z = _robust_channel_z(ratios)
        violates = z > robust_z_cutoff
        counts += violates
        n_retained += 1
        hf_mads.append(hf_mad)
        lf_mads.append(lf_mad)
        window_violations.append(violates)
        if return_details:
            window = start // window_samples
            details["retained"][window] = True
            for metric, values in (("ratio", ratios),
                                   ("robust_z", z),
                                   ("hf_mad_uv", hf_mad * 1e6),
                                   ("lf_mad_uv", lf_mad * 1e6),
                                   ("violates", violates)):
                details[metric][:, window] = values
            for i in np.flatnonzero(violates):
                details["flagged_data_uv"][names[i]].append(data[i] * 1e6)
    candidates = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
    hf_mads = np.asarray(hf_mads).T
    lf_mads = np.asarray(lf_mads).T
    window_violations = np.asarray(window_violations).T
    bad_names = []
    eps = np.finfo(float).eps
    for i, pick in enumerate(picks):
        name = raw.ch_names[pick]
        if name not in candidates:
            continue
        flagged = window_violations[i]
        if flagged.all():
            bad_names.append(name)
            continue
        hf_flagged = np.median(hf_mads[i, flagged])
        hf_baseline = np.median(hf_mads[i, ~flagged])
        lf_flagged = np.median(lf_mads[i, flagged])
        lf_baseline = np.median(lf_mads[i, ~flagged])
        hf_log_change = np.log(max(hf_flagged, eps) / max(hf_baseline, eps))
        lf_log_change = np.log(max(lf_flagged, eps) / max(lf_baseline, eps))
        required_increase = max(0.0, -lf_log_change)
        if (hf_log_change > required_increase
                and not np.isclose(hf_log_change, required_increase, rtol=1e-12, atol=0.0)):
            bad_names.append(name)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"HFnoise": bad_names}
    if return_details:
        details["flagged_data_uv"] = {
            name: np.asarray(windows).reshape(-1, window_samples)
            for name, windows in details["flagged_data_uv"].items()
        }
        return report, details
    return report


def bad_ch_by_BBnoise(
    raw,
    window_length=2.0,
    robust_z_cutoff=3.0,
    min_bad_bands=3,
    occurrence_cutoff=0.25,
    reference_channel=None,
    *,
    return_details=False,
):
    """Return {"BBnoise": names} for persistently broadband high-power EEG channels.

    Per retained window, compute Welch PSD (one full-window Hamming segment,
    DC removed), average power inside each of SPECTRAL_BANDS' five open
    intervals, and convert to dB. Compute median/MAD z-scores across channels
    separately for each band. Flag every channel with z > robust_z_cutoff in
    at least min_bad_bands, in a single pass with no iterative removal.

    Require average reference as in bad_ch_by_smallness; reference_channel is
    ignored. Selection, temporary masking, and occurrence denominator follow
    bad_ch_by_std. Add detections to raw.info["bads"], preserving existing labels
    without duplicates. Input data and annotations are unchanged.

    With return_details=True, return (report, details), with the common fields
    described in bad_ch_by_HFnoise. bands_hz gives the five open intervals.
    band_mean_psd_db (dB relative to 1 V^2/Hz), robust_z, and band_violates
    (0/1) have shape (channels, complete windows, bands). n_bad_bands and
    violates (0/1) have shape (channels, complete windows). Masked windows
    contain NaN in all metric/flag arrays. flagged_data_uv stores original
    waveforms only where the channel violates in at least min_bad_bands.
    """
    SPECTRAL_BANDS = ((0, 4), (4, 8), (8, 12), (12, 30), (30, 45))

    if not np.isfinite(robust_z_cutoff) or robust_z_cutoff <= 0:
        raise ValueError("robust_z_cutoff must be finite and greater than 0.")
    if not isinstance(min_bad_bands, int) or not 1 <= min_bad_bands <= len(SPECTRAL_BANDS):
        raise ValueError("min_bad_bands must be an integer between 1 and 5.")
    if raw.info["sfreq"] / 2 < SPECTRAL_BANDS[-1][1]:
        raise ValueError("Sampling rate must cover all spectral bands through 45 Hz.")
    masked, reference_channel = _temporary_voltage_mask(
        raw, reference_channel, require_average=True
    )
    picks, window_samples, n_windows = _prepare_channel_windows(
        masked, reference_channel, window_length, occurrence_cutoff, require_average=True
    )
    counts = np.zeros(len(picks), dtype=int)
    n_retained = 0
    if return_details:
        details = _init_window_details(raw, picks, window_samples, n_windows, ("n_bad_bands",))
        details["bands_hz"] = SPECTRAL_BANDS
        for metric in ("band_mean_psd_db", "robust_z", "band_violates"):
            details[metric] = np.full((len(picks), n_windows, len(SPECTRAL_BANDS)), np.nan)
    for start, data in _unmasked_windows(
        masked, picks, window_samples, n_windows, return_start=True
    ):
        psds, freqs = mne.time_frequency.psd_array_welch(
            data, sfreq=raw.info["sfreq"], fmin=0, fmax=45,
            n_fft=window_samples, n_per_seg=window_samples, n_overlap=0,
            window="hamming", remove_dc=True, verbose=False,
        )
        bands = [(freqs > low) & (freqs < high) for low, high in SPECTRAL_BANDS]
        if any(not band.any() for band in bands):
            raise ValueError("window_length is too short to resolve all spectral bands.")
        power = np.column_stack([psds[:, band].mean(axis=1) for band in bands])
        power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
        z = _robust_channel_z(power_db)
        band_violates = z > robust_z_cutoff
        n_bad_bands = band_violates.sum(axis=1)
        violates = n_bad_bands >= min_bad_bands
        counts += violates
        n_retained += 1
        if return_details:
            window = _record_window_details(details, start, window_samples, data, violates)
            details["band_mean_psd_db"][:, window] = power_db
            details["robust_z"][:, window] = z
            details["band_violates"][:, window] = band_violates
            details["n_bad_bands"][:, window] = n_bad_bands
    bad_names = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"BBnoise": bad_names}
    if return_details:
        return report, _finish_window_details(details, window_samples)
    return report


def plot_bad_ch_report(raw, bad_ch_report):
    """Show bad-channel categories on an EEG sensor topoplot; return the figure.

    Use the EEG sensor locations in raw.info, including existing bad channels.
    Each report category gets a different colored ring; concentric rings show
    channels flagged by multiple criteria. Omit categories without EEG
    detections from the legend. Non-EEG entries are not plotted. Unflagged
    sensors are gray, and raw and bad_ch_report are unchanged.
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    info = mne.pick_info(raw.info, picks)
    info["bads"] = []  # Category rings provide the bad-channel colors.
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    mne.viz.plot_sensors(
        info, kind="topomap", ch_type="eeg", axes=ax, show=False,
        show_names=True, pointsize=18, linewidth=0.4,
        title="Bad channels by criterion",
    )
    sensors = ax.collections[0]
    sensors.set_facecolor("0.75")
    sensors.set_edgecolor("0.55")
    positions = sensors.get_offsets()
    ring_counts = np.zeros(len(picks), dtype=int)
    colors = plt.get_cmap("tab10")
    for category_index, (category, names) in enumerate(bad_ch_report.items()):
        indices = np.flatnonzero(np.isin(info["ch_names"], names))
        if not len(indices):
            continue
        ax.scatter(
            positions[indices, 0], positions[indices, 1],
            s=(10 + 4 * ring_counts[indices]) ** 2, facecolors="none",
            edgecolors=[colors(category_index % 10)], linewidths=2,
            label=category, zorder=2, clip_on=False,
        )
        ring_counts[indices] += 1
    for label in ax.texts:
        label.set_fontsize(8)
    if ring_counts.any():
        ax.legend(title="Criterion", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.show()
    return fig


# %% Part 2: Definitive bad-time detection (to be developed)


if __name__ == "__main__":
    # %% Load example data
    subject_id = "sub-02"
    project_dir = Path(__file__).resolve().parents[3]
    set_file = (project_dir / subject_id / "eeg"
                / f"{subject_id}_task-ColorPrecMemStudy_run-1_eeg.set")
    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    assert raw.info["sfreq"] == 1000, "Expected 1000 Hz sampling rate."

    enci_onsets = raw.annotations.onset[raw.annotations.description == "enci"]
    encm_onsets = raw.annotations.onset[raw.annotations.description == "encm"]
    if not len(enci_onsets) or not len(encm_onsets):
        raise ValueError("The example crop requires both enci and encm annotations.")
    crop_tmin = enci_onsets.min() - raw.first_time - 3.0
    crop_tmax = encm_onsets.max() - raw.first_time + 2.5
    raw.crop(tmin=crop_tmin, tmax=crop_tmax)

    # %% Line noise removal
    raw_prep = raw.copy()
    raw_prep.filter(l_freq=1.0, h_freq=None, verbose=False)
    clean = Interpolate_fft(raw_prep, line_freq=[30.0, 90.0])
    clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=4.0)
    # Remove the estimated noise from the original data without high-pass filtering it
    remove_line_noise_inplace(raw, raw_prep, clean)

    # %% Bad-channel detection
    raw_bads = raw.copy()
    raw_bads.filter(l_freq=0.3, h_freq=None, verbose=False)

    bad_ch_report = {}
    bad_ch_report.update(bad_ch_by_nan(raw_bads))
    bad_ch_report.update(bad_ch_by_flatline(raw_bads))
    bad_ch_report.update(bad_ch_by_voltage(raw_bads))
    bad_ch_report.update(bad_ch_by_HFnoise(raw_bads))
    plot_bad_ch_report(raw_bads, bad_ch_report)
