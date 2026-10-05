"""ELBO step 2: bad-channel and bad-time detection.

Bad-channel criteria may be local or cross-channel, and time invariant or
time dependent. Bad-time detection and iteration between the two parts will
be developed after the initial bad-channel criteria.

Each bad_ch_by_* function adds its detections to raw.info["bads"] without
duplicates and returns {criterion: [channel names]} for that call's detections.
Signal data and input annotations are unchanged.

The 2-second window detectors accept return_details=True to return
(report, details), including window metrics and original flagged waveforms.
bad_ch_by_pop counts individual electrode pops instead; its details include
each channel's pop times and step sizes.

bad_ch_by_correlation and bad_ch_by_RANSAC compare each channel with the others
on a copy band-limited to 1-45 Hz. The first flags channels that intermittently
lose the signal they share with their neighbours (PREP's correlation
criterion); the second flags channels that random subsets of the others,
through spherical splines, persistently fail to predict (RANSAC).

The native reference is the one EEG channel whose samples are all exactly
zero. bad_ch_by_nan, bad_ch_by_flatline, and pca_cleaning_inplace require it;
bad_ch_by_smallness, bad_ch_by_std, bad_ch_by_BBnoise, bad_ch_by_correlation,
and bad_ch_by_RANSAC require re-referenced data, with no entirely zero EEG
channel; bad_ch_by_voltage, bad_ch_by_HFnoise, and bad_ch_by_pop accept either
and exclude the native reference if present.

pca_cleaning_inplace modifies the data in place: after the initial bad-channel
pass, it removes the PC1 of each 1-second window where PC1 explains at least 90%
of the variance, on the input or on 2 Hz high-passed data, and exceeds 100 µV on
some channel, suppressing large artifacts shared across channels. With
return_removed=True, remove_pca_signal_inplace can subtract the same removed
signals from a copy high-passed at a lower cutoff (filter first, then subtract).

robust_average_reference_inplace then iterates bad-channel detection under a
robust average reference (the mean with bad channels interpolated), interpolates
the bad channels, and subtracts the final robust mean.
"""

from functools import partial
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import mne
from mne.bem import _check_origin  # private: the origin interpolate_bads fits
from mne.channels.interpolation import _make_interpolation_matrix  # private: MNE's spherical spline
import numpy as np
from scipy.signal import fftconvolve, firwin
from step1_line_noise_removal import ZapLine_notch, Interpolate_fft, remove_line_noise_inplace  # noqa: F401


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


def bad_ch_by_nan(raw, reference_channel=None):
    """Return {"nan": names} for every channel with any NaN or +/-inf sample.

    Require the native reference, as in bad_ch_by_flatline. Check all channel
    types and all samples, including annotated spans. Add detections to
    raw.info["bads"], preserving existing labels without duplicates. Data and
    annotations are unchanged.
    """
    _reference_channel(raw, "native", reference_channel)
    bad_names = []
    for i, name in enumerate(raw.ch_names):
        data = raw.get_data(picks=[i], reject_by_annotation=None)
        if not np.isfinite(data).all():
            bad_names.append(name)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    return {"nan": bad_names}


def _zero_eeg_channels(raw, picks):
    """Names of the picked channels whose samples are all exactly zero."""
    names = []
    for i in picks:
        data = raw.get_data(picks=[i], reject_by_annotation=None)
        if data.size and np.all(data == 0):
            names.append(raw.ch_names[i])
    return names


def _reference_channel(raw, reference, reference_channel=None):
    """Check the reference a method needs; return the native reference to exclude.

    "native": a supplied reference_channel must be an entirely zero EEG
    channel, bad or not, and other all-zero channels are then ordinary
    channels; without a name, exactly one non-bad EEG channel must be entirely
    zero. Return its name. "average": no EEG channel, bad or not, may be
    entirely zero; return None. "either": return the one entirely zero non-bad
    EEG channel, or None if there is none; raise if there are several.
    """
    eeg_all = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_good = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if reference == "native":
        if reference_channel is not None:
            if reference_channel not in [raw.ch_names[i] for i in eeg_all]:
                raise ValueError("reference_channel must name an EEG channel in raw.")
            if not _zero_eeg_channels(raw, [raw.ch_names.index(reference_channel)]):
                raise ValueError(f"reference_channel {reference_channel} is not entirely "
                                 "zero; this method requires the native reference.")
            return reference_channel
        zeros = _zero_eeg_channels(raw, eeg_good)
        if len(zeros) != 1:
            raise ValueError("Expected exactly one entirely zero non-bad EEG channel (the "
                             f"native reference); found {zeros}. Supply reference_channel "
                             "explicitly.")
        return zeros[0]
    if reference == "average":
        zeros = _zero_eeg_channels(raw, eeg_all)
        if zeros:
            raise ValueError(f"EEG channels {zeros} are entirely zero, as the native "
                             "reference is; re-reference every EEG channel first, e.g., "
                             "with robust_average_reference_inplace.")
        return None
    zeros = _zero_eeg_channels(raw, eeg_good)
    if len(zeros) > 1:
        raise ValueError(f"Several non-bad EEG channels are entirely zero: {zeros}. "
                         "Run bad_ch_by_flatline first.")
    return zeros[0] if zeros else None


def _prepare_channel_windows(
    raw, window_length, occurrence_cutoff, reference, reference_channel=None,
):
    """Select eligible EEG channels and complete, non-overlapping sample windows.

    reference and reference_channel are checked by _reference_channel.
    """
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
    reference_channel = _reference_channel(raw, reference, reference_channel)
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
    robust_z_cutoff=2.0,
    occurrence_cutoff=0.25,
    *,
    return_details=False,
):
    """Return {"smallness": names} for persistently low-variance EEG channels.

    Require re-referenced data: raise if any EEG channel, bad or not, is
    entirely zero, as the native reference is. Exclude existing bads and
    non-EEG channels. Use complete non-overlapping windows, starting at the
    first sample of this Raw; window_length is in seconds, rounded to samples.
    Ignore the trailing partial window; annotations do not exclude any samples.

    In each window, compute each channel's variance (ddof=0) and its robust z
    across channels, (variance - median) / (1.4826 * MAD), flooring a zero
    scale at machine epsilon. Remove values with robust z strictly above
    robust_z_cutoff, then recompute the median and MAD on the retained
    variances. Flag retained values with new robust z strictly below
    -robust_z_cutoff. Upper-tail removal is a single pass, keeps
    strong-artifact channels from inflating the MAD, and does not itself flag
    a channel. Return channels flagged in strictly more than occurrence_cutoff
    of complete windows, and add them to raw.info["bads"]. Preserve existing
    labels without duplicates; data/annotations are unchanged.

    With return_details=True, return (report, details). Common fields follow
    bad_ch_by_HFnoise: ch_names, crop-relative start_s/stop_s, retained,
    violates, and flagged_data_uv. All complete windows are retained, even
    when annotated. variance_uv2 and baseline_retained (0/1) have shape
    (channels, windows); upper_cutoff_uv2 and lower_cutoff_uv2 have one value
    per window. Variances and cutoffs are in squared microvolts.
    """
    if not np.isfinite(robust_z_cutoff) or robust_z_cutoff <= 0:
        raise ValueError("robust_z_cutoff must be finite and greater than 0.")
    picks, window_samples, n_windows = _prepare_channel_windows(
        raw, window_length, occurrence_cutoff, "average"
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
        high_cutoff = _robust_z_value(variances, robust_z_cutoff)
        keep = variances <= high_cutoff
        low_cutoff = _robust_z_value(variances[keep], -robust_z_cutoff)
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

    Require the native reference: run this before re-referencing. Also run it
    before filtering or line-noise removal: both perturb a constant nonzero
    signal by round-off, so its samples are no longer exactly equal, while
    entirely zero channels stay zero. A supplied
    reference_channel must be entirely zero, bad or not; other entirely zero
    channels are then evaluated like any other. Without a name, exactly one
    non-bad EEG channel must be entirely zero. Channel selection and
    complete-window rules otherwise follow bad_ch_by_smallness, also excluding
    the reference. A window is flat if all samples are
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
        raw, window_length, occurrence_cutoff, "native", reference_channel
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
    *,
    require_average=False,
):
    """Append BAD_voltage_mask annotations to raw in place; return None.

    Exclude existing bads and non-EEG channels. Accept the native reference,
    excluded as in bad_ch_by_voltage, or re-referenced data. With
    require_average=True, require re-referenced data as in bad_ch_by_smallness.
    A channel exceeds the threshold in a window if any sample is strictly outside
    +/-voltage_threshold_uv. Annotate the entire window when strictly more
    than channel_fraction_cutoff of eligible channels exceed it, even if
    their exceedances are not simultaneous. Merge adjacent masked windows;
    include the final partial window with its actual duration.

    These annotations are temporary when called by the five noisy-channel
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
    reference_channel = _reference_channel(raw, "average" if require_average else "either")
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


def _temporary_voltage_mask(raw, *, require_average=False):
    """Check the reference, then mask a copy; return it and any native reference."""
    reference_channel = _reference_channel(raw, "average" if require_average else "either")
    masked = raw.copy()
    mask_segment_by_voltage(masked, require_average=require_average)
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


def _robust_z_value(values, z):
    """Value at robust z across channels, with _robust_channel_z's MAD scale."""
    return np.median(values) + z * max(1.4826 * _mad(values), np.finfo(float).eps)


def _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff):
    """Return flagged names after evaluating at least one retained window."""
    if n_retained == 0:
        raise ValueError("No complete unmasked windows remain for channel detection.")
    return [raw.ch_names[i] for i in picks[counts / n_retained > occurrence_cutoff]]


def bad_ch_by_voltage(
    raw,
    voltage_threshold_uv=200.0,
    occurrence_cutoff=0.25,
    *,
    plot=True,
    return_details=False,
):
    """Return {"voltage": names} for excessive voltage outside temporary masks.

    Flag if strictly more than occurrence_cutoff of retained samples exceed
    +/-voltage_threshold_uv. Exclude existing bads, the reference, non-EEG
    channels, existing BAD annotations, and the temporary 0.1-second voltage
    mask. Add detections to raw.info["bads"], preserving existing labels
    without duplicates. Input data and annotations are unchanged.
    Accept the native reference or re-referenced data: exclude the one
    entirely zero non-bad EEG channel, the native reference, if present, and
    raise if there are several (run bad_ch_by_flatline first).

    With plot=True, show all channels except existing bads over the full time
    range, including the reference and masked samples. Plot in microvolts with
    dotted cutoff lines and a different color per channel. Highlight and label
    only this call's detections in the legend. Shade temporary voltage masks in
    light gray and scale the y-axis from the plotted channels' finite samples
    outside all BAD annotations.

    With return_details=True, return (report, details): ch_names of the
    evaluated channels and exceed_fraction, each one's fraction of retained
    samples beyond +/-voltage_threshold_uv.
    """
    if not np.isfinite(voltage_threshold_uv) or voltage_threshold_uv <= 0:
        raise ValueError("voltage_threshold_uv must be finite and greater than 0.")
    if not np.isfinite(occurrence_cutoff) or not 0 <= occurrence_cutoff <= 1:
        raise ValueError("occurrence_cutoff must be between 0 and 1.")
    masked, reference_channel = _temporary_voltage_mask(raw)
    exclude = masked.info["bads"] + ([] if reference_channel is None else [reference_channel])
    picks = mne.pick_types(
        masked.info, eeg=True, exclude=exclude
    )
    fractions = np.empty(len(picks))
    for row, i in enumerate(picks):
        data = masked.get_data(picks=[i], reject_by_annotation="omit", verbose=False)[0]
        if data.size == 0:
            raise ValueError("No unmasked samples remain for channel detection.")
        fractions[row] = np.mean(np.abs(data) > voltage_threshold_uv / 1e6)
    bad_names = [raw.ch_names[i] for i in picks[fractions > occurrence_cutoff]]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    result = {"voltage": bad_names}
    if return_details:
        result = result, {"ch_names": [raw.ch_names[i] for i in picks],
                          "exceed_fraction": fractions}
    if not plot:
        return result

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    times = raw.times
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(raw.ch_names)))
    ymin, ymax = np.inf, -np.inf
    for i, name in enumerate(raw.ch_names):
        if name in masked.info["bads"]:  # Bads before this call.
            continue
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
    return result


def bad_ch_by_std(
    raw,
    window_length=2.0,
    robust_z_cutoff=5.0,
    occurrence_cutoff=0.25,
    median_z_cutoff=2.0,
    *,
    return_details=False,
):
    """Return {"std": names} for persistently high robust standard deviation.

    PREP findNoisyChannels.m, Method 1: d = 0.7413 * IQR over time;
    z = (d - median(d)) / (0.7413 * IQR(d)) across channels. Use z strictly
    above robust_z_cutoff, never abs(z). Compute a fresh baseline once per
    complete window; exclude existing bads and non-EEG channels. Require
    re-referenced data as in bad_ch_by_smallness.

    Apply the temporary 0.1-second voltage mask on a copy. Omit every window
    overlapping any BAD annotation and the trailing partial window. Flag
    channels offending in strictly more than occurrence_cutoff of retained
    windows, or whose median z over retained windows is strictly above
    median_z_cutoff, which catches channels moderately high in most windows.
    Add detections to raw.info["bads"], preserving existing labels without
    duplicates. Input data and annotations are unchanged.

    With return_details=True, return (report, details), with the common fields
    described in bad_ch_by_HFnoise. robust_std_uv, robust_z, and violates have
    shape (channels, complete windows); masked windows contain NaN.
    median_robust_z has one value per channel. The robust standard deviation
    is reported in microvolts. flagged_data_uv stores the original violating
    window waveforms in chronological order.
    """
    if not np.isfinite(robust_z_cutoff) or robust_z_cutoff <= 0:
        raise ValueError("robust_z_cutoff must be finite and greater than 0.")
    if not np.isfinite(median_z_cutoff) or median_z_cutoff <= 0:
        raise ValueError("median_z_cutoff must be finite and greater than 0.")
    masked, _ = _temporary_voltage_mask(raw, require_average=True)
    picks, window_samples, n_windows = _prepare_channel_windows(
        masked, window_length, occurrence_cutoff, "average"
    )
    counts = np.zeros(len(picks), dtype=int)
    n_retained = 0
    window_z = []
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
        window_z.append(z)
        if return_details:
            window = _record_window_details(details, start, window_samples, data, violates)
            details["robust_std_uv"][:, window] = deviation * 1e6
            details["robust_z"][:, window] = z
    frequent = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
    median_z = np.median(window_z, axis=0)
    bad_names = [raw.ch_names[i] for i, median in zip(picks, median_z)
                 if raw.ch_names[i] in frequent or median > median_z_cutoff]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"std": bad_names}
    if return_details:
        details["median_robust_z"] = median_z
        return report, _finish_window_details(details, window_samples)
    return report


def _lowpass_kernel(sfreq):
    """FIR with a 45 Hz passband edge and a 50 Hz stopband edge at any sampling rate.

    A Hamming-window FIR (firwin, cutoff 47.5 Hz) whose length follows from
    the 5 Hz transition band, 3.3 * sfreq / 5 taps made odd, the rule MNE and
    EEGLAB's firfilt use. PREP's findNoisyChannels.m, Method 2, fixes the order
    at 100 instead, so its band moves with the sampling rate.
    """
    n_taps = int(round(3.3 * sfreq / (50 - 45)))
    n_taps += n_taps % 2 == 0
    return firwin(n_taps, 47.5, window="hamming", fs=sfreq)


def _lowpass_45_hz(data, sfreq):
    """Low-pass data (channels, times) with _lowpass_kernel, forward and backward.

    As in PREP's filtfilt_fast, filter forward and backward after
    odd-reflection padding; the zero-phase gain stays within 1% of 1 up to
    45 Hz and below 2e-5 from 50 Hz. An output sample depends on input samples
    up to one filter length minus one sample away on either side. data must be
    longer than the filter (about 0.66 s).
    """
    kernel = _lowpass_kernel(sfreq)
    width, n_times = len(kernel), data.shape[1]
    if n_times <= width:
        raise ValueError(f"Data must be longer than the {width / sfreq:.2f} s low-pass filter.")
    left = 2 * data[:, :1] - data[:, np.arange(width, 0, -1) % n_times]
    right = 2 * data[:, -1:] - data[:, np.arange(n_times - 2, n_times - width - 2, -1) % n_times]
    padded = np.concatenate([left, data, right], axis=1)
    # FFT convolution truncated to the causal output, as lfilter, but faster for long kernels
    forward = fftconvolve(padded, kernel[None], axes=1)[:, :padded.shape[1]]
    lowpass = fftconvolve(forward[:, ::-1], kernel[None], axes=1)[:, :padded.shape[1]][:, ::-1]
    return lowpass[:, width:width + n_times]


def _prep_hf_ratio(data, sfreq, *, return_mads=False):
    """PREP's MAD(residual)/MAD(lowpass), filtering this window in isolation.

    Low-pass with _lowpass_45_hz; like PREP's, the filter is low-pass only.
    Floor zero low-pass MAD at machine epsilon to keep flat signals finite.
    With return_mads=True, also return the HF and LF MAD arrays, in input units.
    """
    lowpass = _lowpass_45_hz(data, sfreq)
    hf_mad = _mad(data - lowpass, axis=1)
    lf_mad = _mad(lowpass, axis=1)
    ratios = hf_mad / np.maximum(lf_mad, np.finfo(float).eps)
    return (ratios, hf_mad, lf_mad) if return_mads else ratios


def bad_ch_by_HFnoise(
    raw,
    window_length=2.0,
    robust_z_cutoff=5.0,
    occurrence_cutoff=0.25,
    *,
    return_details=False,
):
    """Return {"HFnoise": names} for persistently high PREP HF-noise ratios.

    Use MAD(signal - lowpass) / MAD(lowpass) per channel, then subtract the
    cross-channel median and divide by 1.4826 * cross-channel MAD. A window
    violates when z > robust_z_cutoff. A channel must violate in strictly more
    than occurrence_cutoff of retained windows. Take each eligible channel's
    median HF MAD over this candidate's violating windows. Keep the candidate
    if its median is at or above the cross-channel median of those values,
    including the candidate. Only candidates below that median need HF/LF
    confirmation: compare their median HF and LF MADs in violating (F) versus
    non-violating (N) windows and require
    log(H_F / H_N) > max(0, -log(L_F / L_N)). Numerical ties do not confirm
    the flag. Floor MADs at machine epsilon before taking log ratios.
    Keep candidates that violate in every retained window, since no
    non-violating baseline is available.
    Each retained window is filtered separately to keep masked bursts out of
    neighboring windows. Accept the native reference or re-referenced data as
    in bad_ch_by_voltage. Temporary masking and the occurrence denominator follow
    bad_ch_by_std. Require sfreq > 100 Hz
    to evaluate noise above 50 Hz, and windows longer than the 0.66 s low-pass
    filter. Add detections to raw.info["bads"], preserving
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
    masked, _ = _temporary_voltage_mask(raw)
    picks, window_samples, n_windows = _prepare_channel_windows(
        masked, window_length, occurrence_cutoff, "either"
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
        hf_channel_median = np.median(np.median(hf_mads[:, flagged], axis=1))
        if hf_flagged >= hf_channel_median:
            bad_names.append(name)
            continue
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
    robust_z_cutoff=5.0,
    min_bad_bands=3,
    occurrence_cutoff=0.25,
    *,
    return_details=False,
):
    """Return {"BBnoise": names} for persistently broadband high-power EEG channels.

    Per retained window, compute Welch PSD (one full-window Hamming segment,
    DC removed), average power inside each of SPECTRAL_BANDS' five open
    intervals, and convert to dB. Compute median/MAD z-scores across channels
    separately for each band. Flag every channel with z > robust_z_cutoff in
    at least min_bad_bands, in a single pass with no iterative removal.

    Require re-referenced data as in bad_ch_by_smallness. Selection, temporary
    masking, and occurrence denominator follow bad_ch_by_std. Add detections to
    raw.info["bads"], preserving existing labels without duplicates. Input data
    and annotations are unchanged.

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
    masked, _ = _temporary_voltage_mask(raw, require_average=True)
    picks, window_samples, n_windows = _prepare_channel_windows(
        masked, window_length, occurrence_cutoff, "average"
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


def _level_change(data, jumps, gap, width):
    """Mean over width samples after each jump minus before, skipping gap samples on each side.

    A jump at t lies between samples t and t + 1. data has time on the last axis;
    the result has shape data.shape[:-1] + (len(jumps),).
    """
    after = jumps[:, None] + 1 + gap + np.arange(width)
    before = jumps[:, None] - gap - width + np.arange(width)
    return data[..., after].mean(axis=-1) - data[..., before].mean(axis=-1)


def bad_ch_by_pop(
    raw,
    jump_sd_cutoff=20.0,
    step_floor_uv=30.0,
    neighbor_ratio=3.0,
    sharpness_cutoff=0.7,
    rate_cutoff=1.0,
    *,
    return_details=False,
):
    """Return {"pop": names} for EEG channels with frequent electrode pops.

    A pop is a sharp, sustained voltage step on a single electrode. A candidate
    is a single-sample jump |x(t + 1) - x(t)| above both 20 uV and
    jump_sd_cutoff times the channel's robust SD of sample differences
    (1.4826 * MAD over retained samples). Its level change is the mean over
    10-60 ms after the jump minus the mean over 10-60 ms before it. Confirm the
    candidate when the level change (1) exceeds step_floor_uv in magnitude, so
    brief spikes that return to baseline do not count; (2) exceeds
    neighbor_ratio times the median magnitude of the same level change on the
    6 nearest eligible sensors (3D positions in raw.info), so shared artifacts
    do not count; and (3) is mostly complete within about 20 ms: the level
    change over 10-20 ms windows is at least sharpness_cutoff times the
    10-60 ms one, so slow drifts do not count. Confirmed jumps no more than
    100 ms apart form one pop.

    Flag channels with strictly more than rate_cutoff pops per minute of
    retained data. Accept the native reference or re-referenced data as in
    bad_ch_by_voltage. Exclude existing bads, the reference, and non-EEG
    channels. Apply the temporary voltage mask on a copy, as in bad_ch_by_std;
    masked and BAD-annotated samples are not retained, and candidates whose
    evaluation spans touch them are skipped. Require sfreq >= 100 Hz. Add
    detections to raw.info["bads"], preserving existing labels without
    duplicates. Input data and annotations are unchanged.

    With return_details=True, return (report, details): ch_names, neighbors
    (name -> 6 nearest eligible names), retained_min, per-channel arrays
    jump_sd_uv, n_pops, and pops_per_min, and dicts pop_times_s and
    pop_steps_uv keyed by channel name. Each pop is described by its first
    confirmed jump: crop-relative time of the sample before it, and its signed
    10-60 ms level change in microvolts.
    """
    for name, value in (("jump_sd_cutoff", jump_sd_cutoff), ("step_floor_uv", step_floor_uv),
                        ("neighbor_ratio", neighbor_ratio)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than 0.")
    if not np.isfinite(sharpness_cutoff) or not 0 <= sharpness_cutoff <= 1:
        raise ValueError("sharpness_cutoff must be between 0 and 1.")
    if not np.isfinite(rate_cutoff) or rate_cutoff < 0:
        raise ValueError("rate_cutoff must be finite and nonnegative (pops per minute).")
    sfreq = raw.info["sfreq"]
    if sfreq < 100:
        raise ValueError("bad_ch_by_pop requires a sampling rate of at least 100 Hz.")
    gap, width, short, merge = (int(round(s * sfreq)) for s in (0.01, 0.05, 0.01, 0.1))
    masked, reference_channel = _temporary_voltage_mask(raw)
    exclude = raw.info["bads"] + ([] if reference_channel is None else [reference_channel])
    picks = mne.pick_types(raw.info, eeg=True, exclude=exclude)
    if len(picks) < 7:
        raise ValueError("At least seven non-bad EEG channels after reference exclusion "
                         "are required.")
    positions = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    if not np.isfinite(positions).all() or not np.abs(positions).sum(axis=1).all():
        raise ValueError("Eligible EEG channels need sensor positions; set a montage first.")
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :6]
    # The voltage mask already verified finite data, so NaN marks annotated samples.
    retained = ~np.isnan(
        masked.get_data(picks=picks[:1], reject_by_annotation="NaN", verbose=False)[0]
    )
    retained_min = retained.sum() / sfreq / 60
    if retained_min == 0:
        raise ValueError("No unmasked samples remain for channel detection.")

    names = [raw.ch_names[i] for i in picks]
    jump_sd = np.empty(len(picks))
    n_pops = np.zeros(len(picks), dtype=int)
    pop_times, pop_steps = {}, {}
    for row, pick in enumerate(picks):
        # Row 0 is the channel, rows 1-6 its neighbors.
        data = raw.get_data(picks=[pick, *picks[neighbors[row]]], reject_by_annotation=None)
        data[:, ~retained] = np.nan
        diffs = np.diff(data[0])
        jump_sd[row] = max(1.4826 * _mad(diffs[np.isfinite(diffs)]), np.finfo(float).eps)
        jumps = np.flatnonzero(
            (np.abs(diffs) > jump_sd_cutoff * jump_sd[row]) & (np.abs(diffs) > 20e-6)
        )
        jumps = jumps[(jumps >= gap + width) & (jumps + 1 + gap + width <= raw.n_times)]
        step = _level_change(data, jumps, gap, width)
        sharp_step = _level_change(data[0], jumps, gap, short)
        with np.errstate(divide="ignore", invalid="ignore"):
            confirmed = (
                (np.abs(step[0]) > step_floor_uv / 1e6)
                & (np.abs(step[0]) > neighbor_ratio * np.median(np.abs(step[1:]), axis=0))
                & (sharp_step / step[0] >= sharpness_cutoff)
            )
        first = np.diff(jumps[confirmed], prepend=-np.inf) > merge
        n_pops[row] = first.sum()
        pop_times[names[row]] = jumps[confirmed][first] / sfreq
        pop_steps[names[row]] = step[0, confirmed][first] * 1e6
    pops_per_min = n_pops / retained_min
    bad_names = [name for name, rate in zip(names, pops_per_min) if rate > rate_cutoff]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"pop": bad_names}
    if return_details:
        return report, {
            "ch_names": names,
            "neighbors": {name: [names[j] for j in neighbors[row]]
                          for row, name in enumerate(names)},
            "retained_min": retained_min,
            "jump_sd_uv": jump_sd * 1e6,
            "n_pops": n_pops,
            "pops_per_min": pops_per_min,
            "pop_times_s": pop_times,
            "pop_steps_uv": pop_steps,
        }
    return report


# Cross-channel criteria, scored on a copy band-limited to 1-45 Hz

def _band_limit(data, sfreq):
    """Return data (channels, times) band-limited to 1-45 Hz and the filters' reach.

    High-pass with MNE's default FIR for l_freq=1 (Hamming, 1 Hz passband
    edge, -6 dB at 0.5 Hz, 3.3 s long, zero phase), then low-pass with
    _lowpass_45_hz. An output sample depends on input samples up to the
    returned reach away on either side, in samples: about 2.3 s at any rate.
    """
    highpass = mne.filter.create_filter(None, sfreq, l_freq=1.0, h_freq=None, verbose=False)
    data = mne.filter.filter_data(data, sfreq, l_freq=1.0, h_freq=None, verbose=False)
    return _lowpass_45_hz(data, sfreq), len(highpass) // 2 + len(_lowpass_kernel(sfreq)) - 1


def _widen_voltage_mask(raw, seconds):
    """Extend each BAD_voltage_mask span of raw by seconds on both sides, within the data."""
    annotations = raw.annotations
    mask = np.flatnonzero(annotations.description == "BAD_voltage_mask")
    starts = np.maximum(annotations.onset[mask] - seconds, raw.first_time)
    stops = np.minimum(annotations.onset[mask] + annotations.duration[mask] + seconds,
                       raw.first_time + raw.n_times / raw.info["sfreq"])
    annotations.delete(mask)
    annotations.append(starts, stops - starts, ["BAD_voltage_mask"] * len(mask))


def _band_limited_windows(raw, window_length, occurrence_cutoff):
    """Shared setup of the cross-channel detectors.

    Require sfreq > 100 Hz and re-referenced data. Mask a copy of the input,
    select channels and complete windows as in bad_ch_by_std, band-limit the
    selected channels with _band_limit, and widen the masks by the filters'
    reach. Return the masked copy, picks, window_samples, n_windows, and the
    band-limited data of picks.
    """
    sfreq = raw.info["sfreq"]
    if sfreq <= 100:
        raise ValueError("The 45-50 Hz low-pass requires a sampling rate above 100 Hz.")
    masked, _ = _temporary_voltage_mask(raw, require_average=True)
    picks, window_samples, n_windows = _prepare_channel_windows(
        masked, window_length, occurrence_cutoff, "average"
    )
    filtered, reach = _band_limit(masked.get_data(picks=picks, reject_by_annotation=None), sfreq)
    _widen_voltage_mask(masked, reach / sfreq)
    return masked, picks, window_samples, n_windows, filtered


def _require_variance(values, names, what, start, sfreq):
    """Raise if a row of values (channels, samples) is constant: its r would be undefined."""
    constant = np.ptp(values, axis=1) == 0
    if constant.any():
        raise ValueError(
            f"{what} of {[names[i] for i in np.flatnonzero(constant)]} is constant in the "
            f"window starting at {start / sfreq:g} s, so its r is undefined. Re-referenced "
            "data should never be constant; inspect the recording."
        )


def _row_pearson(x, y):
    """Pearson r between matching rows of x and y."""
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return (x * y).sum(axis=1) / np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum(axis=1))


def bad_ch_by_correlation(
    raw,
    window_length=2.0,
    correlation_cutoff=0.4,
    quantile=0.98,
    occurrence_cutoff=0.01,
    *,
    return_details=False,
):
    """Return {"correlation": names} for channels that intermittently lose their shared signal.

    PREP findNoisyChannels.m, Method 3, with signed r and 2-s windows. In each
    retained window, correlate (Pearson r) every evaluated channel with every
    other one and score it by the quantile of its r with the others (numpy's
    default linear method; the self-correlation is excluded). At 0.98 with
    110-128 partners, the score lies between the channel's 3rd and 4th best
    partner, so it drops only when the channel loses the signal it shares even
    with its closest neighbours, e.g., from a broken contact. Unlike PREP's
    |r|, r is signed, so anticorrelated channels do not count as partners. A
    window violates when the score is strictly below correlation_cutoff; flag
    channels violating in strictly more than occurrence_cutoff of retained
    windows, a few windows at the 1% default.

    Score a copy band-limited to 1-45 Hz (_band_limit). Require re-referenced
    data as in bad_ch_by_smallness, and sfreq > 100 Hz. Exclude existing bads
    and non-EEG channels. Compute the temporary voltage mask on the input, as
    in bad_ch_by_std, then widen each masked span on both sides by the
    filters' reach, since filtering spreads masked artifacts into neighbouring
    windows. Retain complete windows that overlap no widened mask or BAD
    annotation. Raise if an evaluated channel is constant in a retained window,
    in the input or after filtering: its r would be undefined, and
    re-referenced data should never be constant. Add detections to
    raw.info["bads"], preserving existing labels without duplicates. Input data
    and annotations are unchanged.

    With return_details=True, return (report, details), with the common fields
    described in bad_ch_by_HFnoise. score and violates have shape (channels,
    complete windows); unretained windows contain NaN. flagged_data_uv stores
    the input, not band-limited, violating waveforms.
    """
    if not np.isfinite(correlation_cutoff) or not -1 <= correlation_cutoff <= 1:
        raise ValueError("correlation_cutoff must be between -1 and 1.")
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1.")
    masked, picks, window_samples, n_windows, filtered = _band_limited_windows(
        raw, window_length, occurrence_cutoff
    )
    names = [raw.ch_names[i] for i in picks]
    others = ~np.eye(len(picks), dtype=bool)
    counts = np.zeros(len(picks), dtype=int)
    n_retained = 0
    if return_details:
        details = _init_window_details(raw, picks, window_samples, n_windows, ("score",))
    for start, data in _unmasked_windows(
        masked, picks, window_samples, n_windows, return_start=True
    ):
        window = filtered[:, start:start + window_samples]
        _require_variance(data, names, "The input", start, raw.info["sfreq"])
        _require_variance(window, names, "The band-limited signal", start, raw.info["sfreq"])
        r = np.corrcoef(window)[others].reshape(len(picks), -1)
        score = np.quantile(r, quantile, axis=1)
        violates = score < correlation_cutoff
        counts += violates
        n_retained += 1
        if return_details:
            index = _record_window_details(details, start, window_samples, data, violates)
            details["score"][:, index] = score
    bad_names = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"correlation": bad_names}
    if return_details:
        return report, _finish_window_details(details, window_samples)
    return report


def _ransac_predictors(raw, picks, n_subsets, n_predictors, random_state):
    """Draw each evaluated channel's predictor subsets and their spline weights.

    For every EEG channel in raw, bad or not, in Raw order, draw n_subsets
    random orderings of all other EEG channels from one generator seeded with
    random_state, so the orderings do not depend on which channels are bad.
    Each subset of an evaluated channel (one of picks) is the first
    n_predictors evaluated channels of one of its orderings. The weights map a
    subset to its target as interpolate_bads does: positions centred on the
    origin interpolate_bads fits, then MNE's _make_interpolation_matrix.
    Return subsets as rows of picks, shape (len(picks), n_subsets,
    n_predictors), and the weights, of the same shape.
    """
    positions = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    if not np.isfinite(positions).all() or not np.abs(positions).sum(axis=1).all():
        raise ValueError("Evaluated EEG channels need sensor positions; set a montage first.")
    with mne.utils.use_log_level("warning"):
        positions = positions - _check_origin("auto", raw.info)
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    row_of = np.full(len(raw.ch_names), -1)
    row_of[picks] = np.arange(len(picks))
    rng = np.random.default_rng(random_state)
    subsets = np.empty((len(picks), n_subsets, n_predictors), dtype=int)
    for index, target in enumerate(eeg):
        orderings = rng.permuted(np.tile(np.delete(eeg, index), (n_subsets, 1)), axis=1)
        if row_of[target] >= 0:
            for k, ordering in enumerate(orderings):
                subsets[row_of[target], k] = row_of[ordering[row_of[ordering] >= 0][:n_predictors]]
    weights = np.array([
        [_make_interpolation_matrix(positions[subset], positions[[row]])[0]
         for subset in subsets[row]]
        for row in range(len(picks))
    ])
    return subsets, weights


def _ransac_median(weight_stack, window):
    """Per-sample median of every channel's subset predictions in one window.

    weight_stack has shape (n_subsets, channels, channels). Predict in float32,
    whose sort is several times faster than any float64 median here, and
    average the middle order statistics: the usual median of those values.
    """
    predictions = np.sort(weight_stack @ window.astype(np.float32), axis=0)
    middle = len(predictions)
    return (predictions[(middle - 1) // 2].astype(float) + predictions[middle // 2]) / 2


def bad_ch_by_RANSAC(
    raw,
    window_length=2.0,
    method="threshold",
    correlation_cutoff=0.75,
    robust_z_cutoff=5.0,
    occurrence_cutoff=0.25,
    n_subsets=50,
    subset_fraction=0.25,
    random_state=435656,
    *,
    return_details=False,
):
    """Return {"RANSAC": names} for channels the other channels persistently fail to predict.

    PREP findNoisyChannels.m, Method 4, and EEGLAB clean_channels, changed so
    that no channel predicts itself, the subset size is fixed, the median is
    the usual one, r is Pearson's rather than their uncentred cosine, and
    windows are 2 s long.

    Each subset has round(subset_fraction * n) channels, n being all EEG
    channels in raw, bad or not, so its size is the same in every recording:
    32 for 129 channels. Every evaluated channel gets n_subsets subsets drawn
    uniformly from the other evaluated channels (_ransac_predictors): they never
    contain the channel itself, are reproducible from random_state, and change
    little across robust-reference passes, since a newly bad channel is just
    replaced by the next eligible one in its ordering. Map each subset to the
    channel with MNE's spherical spline, take the per-sample median of the
    n_subsets predicted traces, and correlate it with the channel in each
    retained window (Pearson r).

    method="threshold" flags a window when r is strictly below
    correlation_cutoff (PREP style). method="robust_z" pools r over all
    evaluated channels and retained windows, computes z = (median - r) /
    (1.4826 * MAD), flooring a zero scale at machine epsilon, so that a low r
    gives a high z, and flags a window when z is strictly above
    robust_z_cutoff. Either way, flag channels violating in strictly more than
    occurrence_cutoff of retained windows.

    Selection, band-limiting, masking, the re-reference requirement, and the
    error for constant channels follow bad_ch_by_correlation; also raise if a
    median prediction is constant in a retained window. Require 3D positions
    of the evaluated channels and a full subset besides each of them. Add
    detections to raw.info["bads"], preserving existing labels without
    duplicates. Input data and annotations are unchanged.

    With return_details=True, return (report, details), with the common fields
    described in bad_ch_by_HFnoise. r, robust_z (for both methods), and
    violates (of this method) have shape (channels, complete windows);
    unretained windows contain NaN. subsets holds each channel's predictor
    names, shape (channels, n_subsets, subset size). noise_gain is each
    channel's median, over its subsets, of the Euclidean norm of the spline
    weights: how strongly independent predictor noise enters its prediction,
    highest at the rim. flagged_data_uv stores the input violating waveforms;
    flagged_filtered_uv and flagged_predicted_uv store the band-limited channel
    and its median prediction in the same windows.
    """
    if method not in ("threshold", "robust_z"):
        raise ValueError('method must be "threshold" or "robust_z".')
    if not np.isfinite(correlation_cutoff) or not -1 <= correlation_cutoff <= 1:
        raise ValueError("correlation_cutoff must be between -1 and 1.")
    if not np.isfinite(robust_z_cutoff) or robust_z_cutoff <= 0:
        raise ValueError("robust_z_cutoff must be finite and greater than 0.")
    if not isinstance(n_subsets, int) or n_subsets < 1:
        raise ValueError("n_subsets must be a positive integer.")
    if not np.isfinite(subset_fraction) or not 0 < subset_fraction <= 1:
        raise ValueError("subset_fraction must be greater than 0 and at most 1.")
    masked, picks, window_samples, n_windows, filtered = _band_limited_windows(
        raw, window_length, occurrence_cutoff
    )
    n_predictors = int(round(subset_fraction * len(mne.pick_types(raw.info, eeg=True, exclude=[]))))
    if n_predictors < 1:
        raise ValueError("subset_fraction is too small to give a subset of at least one channel.")
    if len(picks) <= n_predictors:
        raise ValueError(f"RANSAC needs {n_predictors} predictors besides each channel, so at "
                         f"least {n_predictors + 1} non-bad EEG channels; found {len(picks)}.")
    subsets, weights = _ransac_predictors(raw, picks, n_subsets, n_predictors, random_state)
    weight_stack = np.zeros((n_subsets, len(picks), len(picks)), dtype=np.float32)
    for k in range(n_subsets):
        weight_stack[k, np.arange(len(picks))[:, None], subsets[:, k]] = weights[:, k]
    names = [raw.ch_names[i] for i in picks]
    sfreq = raw.info["sfreq"]
    r = np.full((len(picks), n_windows), np.nan)
    retained = np.zeros(n_windows, dtype=bool)
    for start, data in _unmasked_windows(
        masked, picks, window_samples, n_windows, return_start=True
    ):
        window = filtered[:, start:start + window_samples]
        _require_variance(data, names, "The input", start, sfreq)
        _require_variance(window, names, "The band-limited signal", start, sfreq)
        median = _ransac_median(weight_stack, window)
        _require_variance(median, names, "The median prediction", start, sfreq)
        r[:, start // window_samples] = _row_pearson(window, median)
        retained[start // window_samples] = True
    if not retained.any():
        raise ValueError("No complete unmasked windows remain for channel detection.")
    pooled = r[:, retained]
    z = (np.median(pooled) - r) / max(1.4826 * _mad(pooled, axis=None), np.finfo(float).eps)
    violates = (r < correlation_cutoff if method == "threshold" else z > robust_z_cutoff) & retained
    bad_names = _noisy_window_bads(raw, picks, violates.sum(axis=1), retained.sum(),
                                   occurrence_cutoff)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    report = {"RANSAC": bad_names}
    if not return_details:
        return report

    details = _init_window_details(raw, picks, window_samples, n_windows, ("r", "robust_z"))
    details["r"], details["robust_z"] = r, z
    details["subsets"] = np.array(names)[subsets]
    details["noise_gain"] = np.median(np.linalg.norm(weights, axis=2), axis=1)
    predicted = {"flagged_filtered_uv": {name: [] for name in names},
                 "flagged_predicted_uv": {name: [] for name in names}}
    for index in np.flatnonzero(retained):
        start = index * window_samples
        data = masked.get_data(picks=picks, start=start, stop=start + window_samples,
                               reject_by_annotation=None)
        _record_window_details(details, start, window_samples, data, violates[:, index])
        rows = np.flatnonzero(violates[:, index])
        if len(rows):
            window = filtered[:, start:start + window_samples]
            median = _ransac_median(weight_stack, window)
            for i in rows:
                predicted["flagged_filtered_uv"][names[i]].append(window[i] * 1e6)
                predicted["flagged_predicted_uv"][names[i]].append(median[i] * 1e6)
    for field, waveforms in predicted.items():
        details[field] = {name: np.asarray(windows).reshape(-1, window_samples)
                          for name, windows in waveforms.items()}
    return report, _finish_window_details(details, window_samples)


def plot_bad_ch_report(raw, bad_ch_report):
    """Show bad-channel categories on an EEG sensor topoplot; return the figure.

    Use the EEG sensor locations in raw.info, including existing bad channels.
    Each report category gets a different colored ring; concentric rings show
    channels flagged by multiple criteria. Omit categories without EEG
    detections from the legend. Non-EEG entries are not plotted. Unflagged
    sensors are gray, and raw and bad_ch_report are unchanged.
    """
    print(bad_ch_report)

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


# %% Shared-artifact removal by local PCA

def pca_cleaning_inplace(
    raw, window_length=1.0, variance_cutoff=0.9, fade_length=0.25, highpass_freq=2.0,
    amplitude_floor_uv=100.0, reference_channel=None, *, return_removed=False,
):
    """Remove dominant local PC1s from preloaded raw in place; return details.

    Split the recording into non-overlapping windows from the first sample of
    this Raw; window_length is in seconds, rounded to samples, and the trailing
    partial window joins the last complete window. Each window gets two PCA
    passes across every non-bad EEG channel, each channel mean-centered: one on
    the input data ("input") and one on a copy high-passed at highpass_freq
    ("highpass"), where brief shared spikes stand out from slow activity. A pass
    flags the window when its PC1 explains at least variance_cutoff of the
    variance. The pass qualifies only if the back-projection u u^T x(t) in its
    own data exceeds amplitude_floor_uv on at least one channel, so small shared
    signals, such as rhythms under the reference, are kept.

    Each cleaned window subtracts one PC1 back-projection from the input data.
    When the input pass qualifies, it is u u^T x(t) with that pass's weights and
    the input data; otherwise it is u_hp u_hp^T x_hp(t) with the high-pass
    weights and high-passed data, so activity the filter rejects, including the
    artifact's slow part, is kept. Window means are not subtracted, so an
    input-pass removal includes the artifact's average level.
    Consecutive cleaned windows form runs. A run's removal fades in (raised
    cosine) over the fade_length seconds before its first window and out over
    the fade_length seconds after its last. Between consecutive cleaned windows,
    the removal hands over from one window's u u^T to the next over
    +/- fade_length around their shared edge; the two weights sum to one, so the
    total removal weight never exceeds one. fade_length may be 0 (hard edges)
    up to half of window_length.

    Require the native reference, as in bad_ch_by_flatline: run this before
    re-referencing, which changes PC1 variance fractions and peaks, so the
    cutoffs apply to native-reference data only. Exclude existing bads,
    non-EEG channels, and the reference, which stay unchanged. Bad labels and
    annotations are unchanged.

    Return ch_names, crop-relative start_s/stop_s, first_samp, n_times, sfreq,
    cleaned (bool), removed_pass ("input", "highpass", or "" per window), and for
    each pass ("input", "highpass") a dict of per-window pc1_variance_fraction,
    pc1_peak_uv, flagged and qualified (bool), plus pc1_loadings with shape
    (windows, channels): unit norm, arbitrary sign, computed before cleaning.
    With return_removed=True, also return removed_spans: one (start, stop,
    signal) per run including its fades, in samples from the Raw start, with the
    subtracted signal in volts, shape (len(ch_names), stop - start). Pass the
    details to remove_pca_signal_inplace to subtract the same signal from other
    data.
    """
    if not np.isfinite(variance_cutoff) or not 0 < variance_cutoff < 1:
        raise ValueError("variance_cutoff must be between 0 and 1.")
    if not np.isfinite(fade_length) or not 0 <= fade_length <= window_length / 2:
        raise ValueError("fade_length must be between 0 and half of window_length (seconds).")
    if not np.isfinite(highpass_freq) or not 0 < highpass_freq < raw.info["sfreq"] / 2:
        raise ValueError("highpass_freq must be between 0 Hz and the Nyquist frequency.")
    if not np.isfinite(amplitude_floor_uv) or amplitude_floor_uv < 0:
        raise ValueError("amplitude_floor_uv must be finite and nonnegative (microvolts).")
    if not raw.preload:
        raise ValueError("raw must be preloaded.")
    picks, window_samples, n_windows = _prepare_channel_windows(
        raw, window_length, 0.25, "native", reference_channel
    )
    fade = int(round(fade_length * raw.info["sfreq"]))
    starts = np.arange(n_windows) * window_samples
    stops = np.append(starts[1:], raw.n_times)
    highpassed = raw._data[picks]
    if not np.isfinite(highpassed).all():
        raise ValueError("Nonfinite data in eligible channels; run bad_ch_by_nan first.")
    highpassed = mne.filter.filter_data(
        highpassed, raw.info["sfreq"], highpass_freq, None, copy=False, verbose=False
    )
    details = {
        "ch_names": [raw.ch_names[i] for i in picks],
        "start_s": starts / raw.info["sfreq"], "stop_s": stops / raw.info["sfreq"],
        "first_samp": raw.first_samp, "n_times": raw.n_times, "sfreq": raw.info["sfreq"],
    }
    if return_removed:
        details["removed_spans"] = []
    for name in ("input", "highpass"):
        details[name] = {
            "pc1_variance_fraction": np.empty(n_windows),
            "pc1_peak_uv": np.empty(n_windows),
            "pc1_loadings": np.empty((n_windows, len(picks))),
        }
    for window, (start, stop) in enumerate(zip(starts, stops)):
        for name, data in (("input", raw._data[picks, start:stop]),
                           ("highpass", highpassed[:, start:stop])):
            loadings, singular, _ = np.linalg.svd(
                data - data.mean(axis=1, keepdims=True), full_matrices=False
            )
            pc1 = loadings[:, 0]
            details[name]["pc1_variance_fraction"][window] = singular[0] ** 2 / np.sum(singular ** 2)
            details[name]["pc1_peak_uv"][window] = np.abs(np.outer(pc1, pc1 @ data)).max() * 1e6
            details[name]["pc1_loadings"][window] = pc1
    for name in ("input", "highpass"):
        passed = details[name]
        passed["flagged"] = passed["pc1_variance_fraction"] >= variance_cutoff
        passed["qualified"] = passed["flagged"] & (passed["pc1_peak_uv"] > amplitude_floor_uv)
    cleaned = details["input"]["qualified"] | details["highpass"]["qualified"]
    details["cleaned"] = cleaned
    details["removed_pass"] = np.where(
        details["input"]["qualified"], "input", np.where(cleaned, "highpass", "")
    )
    print(f"PC1 removed in {cleaned.sum()} of {cleaned.size} {window_length:g}-s windows")

    def ramp(samples, onset, length):
        """Raised cosine rising from 0 to 1 over length samples from onset."""
        if length == 0:
            return (samples >= onset).astype(float)
        return 0.5 - 0.5 * np.cos(np.pi * np.clip((samples - onset + 0.5) / length, 0, 1))

    # Runs are separated by at least one uncleaned window, so their fades never
    # overlap and each run can be cleaned from its own copy of the original data.
    edges = np.flatnonzero(np.diff(np.r_[False, cleaned, False])).reshape(-1, 2)
    for first, last in edges - [0, 1]:
        span_start = max(starts[first] - fade, 0)
        span_stop = min(stops[last] + fade, raw.n_times)
        samples = np.arange(span_start, span_stop)
        data = raw._data[picks, span_start:span_stop]
        sources = {"input": data, "highpass": highpassed[:, span_start:span_stop]}
        removal = np.zeros_like(data)
        for window in range(first, last + 1):
            rise = (starts[window] - fade, fade if window == first else 2 * fade)
            fall = (stops[window] if window == last else stops[window] - fade,
                    fade if window == last else 2 * fade)
            weight = ramp(samples, *rise) * (1 - ramp(samples, *fall))
            name = details["removed_pass"][window]
            pc1 = details[name]["pc1_loadings"][window]
            removal += weight * np.outer(pc1, pc1 @ sources[name])
        raw._data[picks, span_start:span_stop] = data - removal
        if return_removed:
            details["removed_spans"].append((span_start, span_stop, removal))
    return details


def remove_pca_signal_inplace(raw, pca_details):
    """Subtract pca_cleaning_inplace's removed signals from another preloaded Raw in place.

    Use this for a copy of the same recording high-passed at or below the cutoff
    of the data pca_cleaning_inplace saw (e.g., 0.1 Hz when it saw 0.3 Hz):
    filter first, then subtract. Subtracting after the same cutoff reproduces
    the cleaned data exactly; after a lower cutoff, the result is the cleaned
    data plus the recording's own band between the two cutoffs. For higher
    cutoffs (e.g., 1 Hz for ICA), filter the cleaned data instead of calling this.

    pca_details must come from pca_cleaning_inplace(..., return_removed=True) on data
    covering the same samples (first_samp, n_times, sfreq) and containing the
    same channel names. Only those channels within the cleaned runs change. The
    stored signals are subtracted as computed, not re-projected from this Raw.
    Return None.
    """
    if "removed_spans" not in pca_details:
        raise ValueError("Run pca_cleaning_inplace with return_removed=True.")
    if not raw.preload:
        raise ValueError("raw must be preloaded.")
    if (raw.first_samp, raw.n_times, raw.info["sfreq"]) != (
        pca_details["first_samp"], pca_details["n_times"], pca_details["sfreq"]
    ):
        raise ValueError("raw must cover the same samples as the data passed to "
                         "pca_cleaning_inplace.")
    picks = [raw.ch_names.index(name) for name in pca_details["ch_names"]]
    for start, stop, removal in pca_details["removed_spans"]:
        raw._data[picks, start:stop] -= removal


def plot_pca_cleaning(raw, cleaned_raw, pca_details):
    """Show the whole recording before and after pca_cleaning_inplace; return both figures.

    The first figure overlays the channels that entered the PCA (non-bad EEG,
    reference excluded; one color each) before and after cleaning on a shared
    voltage axis. Each trace is drawn as its min/max envelope in 20-ms bins, so
    brief peaks survive at full-recording scale without millions of vertices.
    The second figure shows channels after cleaning at full resolution, one panel
    each: the three PCA channels whose directions from the head origin are
    nearest midline targets about 53 degrees in front of, at, and behind the
    vertex (frontal, middle, posterior), then the first bad EEG channel in
    cleaned_raw.info["bads"], which was not cleaned, if there is one. Blue
    shading marks runs of cleaned windows; each run's removal also fades in and
    out just outside it. The bad channel's panel shades the same runs pink
    instead, since nothing was removed there. In the second figure, finite
    samples are drawn and gray shading with a legend marks NaN or infinite
    samples, which can occur only in the bad channel. Times are relative to the
    cropped Raw.
    """
    sfreq = raw.info["sfreq"]
    names = pca_details["ch_names"]
    positions = np.array([cleaned_raw.info["chs"][cleaned_raw.ch_names.index(name)]["loc"][:3]
                          for name in names])
    if not np.isfinite(positions).all() or not np.abs(positions).sum(axis=1).all():
        raise ValueError("PCA channels need sensor positions; set a montage first.")
    directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    available = list(range(len(names)))
    channels = []
    # MNE head coordinates: +y anterior, +z superior.
    for region, target in (("frontal", [0, 0.8, 0.6]), ("middle", [0, 0, 1]),
                           ("posterior", [0, -0.8, 0.6])):
        pick = available[int(np.argmax(directions[available] @ target))]
        available.remove(pick)
        channels.append((names[pick], region))
    eeg_info = mne.pick_info(cleaned_raw.info, mne.pick_types(cleaned_raw.info, eeg=True, exclude=[]))
    channels += [(name, "bad, not cleaned") for name in eeg_info["bads"][:1]]
    cleaned = pca_details["cleaned"]
    edges = np.flatnonzero(np.diff(np.r_[False, cleaned, False])).reshape(-1, 2)
    runs = [(pca_details["start_s"][first], pca_details["stop_s"][stop - 1])
            for first, stop in edges]
    window_length = pca_details["stop_s"][0] - pca_details["start_s"][0]

    def shade(ax, color="lightsteelblue"):
        for start, stop in runs:
            ax.axvspan(start, stop, color=color, alpha=0.5, zorder=0)

    bin_samples = int(round(0.02 * sfreq))
    n_bins = raw.n_times // bin_samples
    envelope_times = np.repeat(np.arange(n_bins) * bin_samples / sfreq, 2)
    overview, axes = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True, sharey=True, layout="constrained"
    )
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(names)))
    for ax, inst, stage in zip(axes, (raw, cleaned_raw), ("Before", "After")):
        for color, name in zip(colors, names):
            data = inst.get_data(picks=[name], reject_by_annotation=None)[0] * 1e6
            binned = data[:n_bins * bin_samples].reshape(n_bins, bin_samples)
            envelope = np.column_stack([binned.min(axis=1), binned.max(axis=1)]).ravel()
            ax.plot(envelope_times, envelope, color=color, linewidth=0.4, alpha=0.7)
        shade(ax)
        ax.set_ylabel(f"{stage} PCA cleaning\nVoltage (µV)")
    axes[-1].set(xlabel="Time from cropped recording start (s)", xlim=(0, raw.times[-1]))
    overview.suptitle(
        f"{len(names)} channels in the PCA (bad channels and the reference not shown); "
        f"blue = {cleaned.sum()} cleaned {window_length:g}-s windows in {len(runs)} runs"
    )

    traces, axes = plt.subplots(
        len(channels), 1, figsize=(16, 2.4 * len(channels)), sharex=True,
        squeeze=False, layout="constrained",
    )
    for row, (ax, (name, label)) in enumerate(zip(axes[:, 0], channels)):
        data = cleaned_raw.get_data(picks=[name], reject_by_annotation=None)[0] * 1e6
        ax.plot(cleaned_raw.times, data, color=f"C{row % 10}", linewidth=0.5)
        shade(ax, "lightsteelblue" if name in names else "pink")
        missing = np.flatnonzero(np.diff(np.r_[False, ~np.isfinite(data), False])).reshape(-1, 2)
        # A 3-pt edge keeps runs far shorter than a pixel visible at full-recording scale.
        for i, (start, stop) in enumerate(missing / sfreq):
            ax.axvspan(start, stop, color="0.6", alpha=0.6, linewidth=3, zorder=0,
                       label="_nolegend_" if i else "NaN/inf samples (not drawn)")
        if len(missing):
            ax.legend(loc="upper right")
        ax.set_ylabel(f"{name} ({label})\nVoltage (µV)")
    axes[-1, 0].set(xlabel="Time from cropped recording start (s)", xlim=(0, raw.times[-1]))
    bad_note = "; pink = the same windows in the bad channel, which was not cleaned"
    traces.suptitle("After PCA cleaning, full resolution; blue = cleaned windows"
                    + (bad_note if len(channels) > 3 else ""))
    plt.show(block=False)
    return overview, traces


# %% Robust average reference

def _robust_mean(raw, bads):
    """Mean over all EEG channels of a copy with the named bads spherically interpolated."""
    interpolated = raw.copy()
    interpolated.info["bads"] = list(bads)
    if bads:
        interpolated.interpolate_bads(reset_bads=False, on_bad_position="raise", verbose=False)
    eeg = mne.pick_types(interpolated.info, eeg=True, exclude=[])
    return interpolated._data[eeg].mean(axis=0)


def robust_average_reference_inplace(
    raw, reference_channel=None, max_iterations=10, *, ransac_method="threshold",
    return_details=False,
):
    """Robust-average-reference preloaded raw in place; return (report, bads, reference).

    Require the native reference, as in bad_ch_by_flatline. Existing bads stay
    bad and are never re-evaluated. Non-bad EEG data must be finite.

    Each pass takes the robust mean: on a copy, interpolate the existing EEG
    bads and the channels the previous pass flagged (none in the first pass)
    with MNE's spherical splines at default settings, then average all EEG
    channels, including the native reference (all zero) and the interpolated
    channels. Subtract this mean from every EEG channel of another copy, whose
    bad channels are not interpolated, and run bad_ch_by_voltage (no plot),
    bad_ch_by_HFnoise, bad_ch_by_pop, bad_ch_by_smallness, bad_ch_by_std,
    bad_ch_by_BBnoise, bad_ch_by_correlation, and bad_ch_by_RANSAC (with
    method=ransac_method) on it in that order: the detectors that require
    average reference after the others, and the cross-channel detectors last,
    so that correlation's flags are not RANSAC predictors. Each pass starts
    from the existing bads, and each detector also excludes the channels
    flagged before it in the same pass. Stop when a pass flags exactly the
    channels its robust mean interpolated. If a pass
    instead flags the set an earlier robust mean interpolated (none for the
    first), the passes would cycle: warn, treat every channel flagged within
    the cycle as bad, and recompute the robust mean with them interpolated.
    Raise if neither has happened within max_iterations passes.

    Then interpolate every bad EEG channel in raw, subtract the last robust
    mean from every EEG channel, and remove the EEG channels from
    raw.info["bads"], so all EEG channels average to zero. Non-EEG channels are
    unchanged. Unless the reference channel was interpolated, verify that it
    now holds exactly minus the robust mean.

    Return the last pass's {criterion: names} (after a cycle, each criterion's
    channels across the cycling passes), the interpolated EEG channels in
    Raw order (existing and new bads), and the subtracted robust mean in volts,
    shape (n_times,). Adding it back to every EEG channel restores the native
    reference.

    With return_details=True, also return a list with one dict per pass:
    interpolated (the channels interpolated for its robust mean), flagged (the
    channels it flagged), and each criterion's details from its detector with
    return_details=True. Each detector's details cover the channels it
    evaluated, without the existing bads and the channels flagged before it in
    that pass. A non-bad channel's final trace in raw is the signal they saw.
    """
    if not raw.preload:
        raise ValueError("raw must be preloaded.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if ransac_method not in ("threshold", "robust_z"):
        raise ValueError('ransac_method must be "threshold" or "robust_z".')
    reference_channel = _reference_channel(raw, "native", reference_channel)
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_names = [raw.ch_names[i] for i in eeg]
    orig_bads = list(raw.info["bads"])
    initial = [name for name in eeg_names if name in orig_bads]
    if not np.isfinite(raw._data[mne.pick_types(raw.info, eeg=True, exclude="bads")]).all():
        raise ValueError("Nonfinite data in non-bad EEG channels; run bad_ch_by_nan first.")

    detectors = (
        # the first three can also run on native reference
        partial(bad_ch_by_voltage, plot=False),
        bad_ch_by_HFnoise,
        bad_ch_by_pop,
        # the next five require average reference
        bad_ch_by_smallness,
        bad_ch_by_std,
        bad_ch_by_BBnoise,
        # cross-channel detectors last; RANSAC's predictors exclude correlation's flags
        bad_ch_by_correlation,
        partial(bad_ch_by_RANSAC, method=ransac_method),
    )
    flagged, used, history, reports, details = [], [], [], [], []
    for _ in range(max_iterations):
        used.append(flagged)  # used[p]: channels interpolated for pass p + 1's mean
        reference = _robust_mean(raw, initial + flagged)
        referenced = raw.copy()
        referenced._data[eeg] -= reference
        report, pass_details = {}, {"interpolated": initial + flagged}
        for detector in detectors:
            result = detector(referenced, return_details=return_details)
            if return_details:
                result, detector_details = result
                pass_details[next(iter(result))] = detector_details
            report.update(result)
        new = [name for name in eeg_names
               if name in referenced.info["bads"] and name not in initial]
        pass_details["flagged"] = new
        details.append(pass_details)
        history.append(new)
        reports.append(report)
        print(f"Robust average reference pass {len(history)}: bad channels {new}")
        if set(new) == set(flagged):
            break
        start = next((p for p, earlier in enumerate(used[:-1]) if set(earlier) == set(new)), None)
        if start is not None:
            # The next pass would repeat pass start + 1, so used[start:] keeps recurring.
            cycle = used[start:]
            flagged = [name for name in eeg_names if any(name in names for names in cycle)]
            flipping = [name for name in flagged if not all(name in names for names in cycle)]
            cycle_reports = reports[max(start, 1) - 1:]
            report = {criterion: [name for name in eeg_names
                                  if any(name in r[criterion] for r in cycle_reports)]
                      for criterion in report}
            warnings.warn(f"Robust average reference cycled: pass {len(history) + 1} would "
                          f"repeat pass {start + 1}, with {flipping} flipping. Treating every "
                          f"channel flagged in the cycle as bad: {flagged}.", stacklevel=2)
            reference = _robust_mean(raw, initial + flagged)
            break
        flagged = new
    else:
        raise RuntimeError(f"Robust average reference did not converge in {max_iterations} "
                           f"passes; bad channels flagged per pass: {history}")

    bads = [name for name in eeg_names if name in initial or name in flagged]
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, on_bad_position="raise", verbose=False)
    raw._data[eeg] -= reference
    raw.info["bads"] = [name for name in orig_bads if name not in eeg_names]
    if reference_channel not in bads and not np.array_equal(
        raw._data[raw.ch_names.index(reference_channel)], -reference
    ):
        raise RuntimeError(f"Reference channel {reference_channel} does not hold minus the "
                           "robust mean after re-referencing.")
    if return_details:
        return report, bads, reference, details
    return report, bads, reference


# %% Part 2: Definitive bad-time detection (to be developed)


if __name__ == "__main__":
    # %% Load example data
    subject_id = "sub-02"
    project_dir = Path(__file__).resolve().parents[3]
    set_file = (project_dir / subject_id / "eeg"
                / f"{subject_id}_task-ColorPrecMemStudy_run-1_eeg.set")
    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    assert raw.info["sfreq"] == 1000, "Expected 1000 Hz sampling rate."
    # The amplifier export applied a 100 Hz low-pass that the .set file doesn't record.
    with raw.info._unlock():
        raw.info["lowpass"] = 100.0

    enci_onsets = raw.annotations.onset[raw.annotations.description == "enci"]
    encm_onsets = raw.annotations.onset[raw.annotations.description == "encm"]
    if not len(enci_onsets) or not len(encm_onsets):
        raise ValueError("The example crop requires both enci and encm annotations.")
    crop_tmin = enci_onsets.min() - raw.first_time - 3.0
    crop_tmax = encm_onsets.max() - raw.first_time + 2.5
    raw.crop(tmin=crop_tmin, tmax=crop_tmax)

    # %% Initial bad channels: nonfinite values and flat lines, on the unfiltered data
    # because line-noise removal and filtering break exact flatness
    bad_ch_report = {}
    # bad_ch_report.update(bad_ch_by_impedance(raw, initial_impedance))
    bad_ch_report.update(bad_ch_by_nan(raw))
    bad_ch_report.update(bad_ch_by_flatline(raw))

    # %% Line noise removal
    raw_prep = raw.copy()
    raw_prep.filter(l_freq=1.0, h_freq=None, verbose=False)
    # clean = ZapLine_notch(raw_prep, line_freq=30.0)
    # clean = ZapLine_notch(clean, line_freq=90.0)
    # clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=1.5)
    clean = Interpolate_fft(raw_prep, line_freq=[30.0, 90.0])
    clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=4.0)
    # Remove the estimated noise from the original data without high-pass filtering it
    remove_line_noise_inplace(raw, raw_prep, clean)

    # %% High-pass filter at 0.3 Hz
    raw_bads = raw.copy()
    raw_bads.filter(l_freq=0.3, h_freq=None, verbose=False)

    # %% Bad channels by voltage, high-frequency noise, and electrode pops
    bad_ch_report.update(bad_ch_by_voltage(raw_bads))
    bad_ch_report.update(bad_ch_by_HFnoise(raw_bads))
    bad_ch_report.update(bad_ch_by_pop(raw_bads))
    plot_bad_ch_report(raw_bads, bad_ch_report)  # Before PCA cleaning

    # %% Local PCA cleaning of shared artifacts, excluding the bad channels above.
    # Rerun the detectors on the cleaned data; if they flag new channels, add them
    # to the bads and redo the cleaning on the pre-cleaning data.
    # while True:
    #     raw_pca = raw_bads.copy()
    #     pca_details = pca_cleaning_inplace(raw_pca)
    #     new_bads = {}
    #     new_bads.update(bad_ch_by_voltage(raw_pca, plot=False))
    #     new_bads.update(bad_ch_by_HFnoise(raw_pca))
    #     new_bads.update(bad_ch_by_pop(raw_pca))
    #     print(f"New bad channels after PCA cleaning: {new_bads}")
    #     if not any(new_bads.values()):
    #         break
    #     for criterion, names in new_bads.items():
    #         bad_ch_report[criterion] = bad_ch_report[criterion] + names
    #     raw_bads.info["bads"] = list(raw_pca.info["bads"])
    # plot_bad_ch_report(raw_bads, bad_ch_report)  # After PCA cleaning

    # # Whole-recording traces before and after PCA cleaning
    # pca_overview_figure, pca_channel_figure = plot_pca_cleaning(raw_bads, raw_pca, pca_details)
    # raw_bads = raw_pca  # Keep the cleaned data for the next steps

    # %% Robust average reference: interpolate all bad channels, which info["bads"]
    # no longer lists; bad_ch_report keeps them by criterion.
    robust_report, interpolated_bads, robust_reference = robust_average_reference_inplace(raw_bads)
    for criterion, names in robust_report.items():
        bad_ch_report[criterion] = bad_ch_report.get(criterion, []) + names
    plot_bad_ch_report(raw_bads, bad_ch_report)

    # Band-power topomaps after robust average referencing (bads interpolated). The
    # median Welch PSD shows persistent channel power; brief large events barely count.
    psd_topomap_figure = raw_bads.compute_psd(fmax=45.0, average="median", verbose=False).plot_topomap()
