"""ELBO step 2: bad-channel and bad-time detection.

Bad-channel criteria may be local or cross-channel, and time invariant or
time dependent. Bad-time detection and iteration between the two parts will
be developed after the initial bad-channel criteria.

Each bad_ch_by_* function adds its detections to raw.info["bads"] without
duplicates and returns {criterion: [channel names]} for that call's detections.
Signal data and input annotations are unchanged.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter
from step1_line_noise_removal import Interpolate_fft, remove_line_noise_inplace
from somata.multitaper.multitaper_spectrogram_python import multitaper_spectrogram


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


# Local signal properties with time-dependent occurrence criteria
def bad_ch_by_smallness(
    raw,
    window_length=2.0,
    std_cutoff=2.0,
    occurrence_cutoff=0.25,
    reference_channel=None,
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
    """
    if not np.isfinite(std_cutoff) or std_cutoff <= 0:
        raise ValueError("std_cutoff must be finite and greater than 0.")
    picks, window_samples, n_windows = _prepare_channel_windows(
        raw, reference_channel, window_length, occurrence_cutoff, require_average=True
    )
    counts = np.zeros(len(picks), dtype=int)
    for start in range(0, n_windows * window_samples, window_samples):
        data = raw.get_data(
            picks=picks, start=start, stop=start + window_samples,
            reject_by_annotation=None,
        )
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite data in eligible channels; run bad_ch_by_nan first.")
        variances = data.var(axis=1)
        keep = variances <= variances.mean() + std_cutoff * variances.std()
        retained = variances[keep]
        low_cutoff = retained.mean() - std_cutoff * retained.std()
        counts += keep & (variances < low_cutoff)

    bad_names = [raw.ch_names[i] for i in picks[counts / n_windows > occurrence_cutoff]]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    return {"smallness": bad_names}


def bad_ch_by_flatline(
    raw,
    window_length=2.0,
    occurrence_cutoff=0.25,
    reference_channel=None,
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
    """
    picks, window_samples, n_windows = _prepare_channel_windows(
        raw, reference_channel, window_length, occurrence_cutoff, require_reference=True
    )
    counts = np.zeros(len(picks), dtype=int)
    for start in range(0, n_windows * window_samples, window_samples):
        data = raw.get_data(
            picks=picks, start=start, stop=start + window_samples,
            reject_by_annotation=None,
        )
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite data in eligible channels; run bad_ch_by_nan first.")
        flat = np.all(data == data[:, :1], axis=1)
        other_flat_count = flat.sum() - flat.astype(int)
        counts += flat & (other_flat_count < 0.5 * (len(picks) - 1))

    bad_names = [raw.ch_names[i] for i in picks[counts / n_windows > occurrence_cutoff]]
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    return {"flatline": bad_names}


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
    call's detections in the legend.
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
    for i, name in enumerate(raw.ch_names):
        data_uv = raw.get_data(picks=[i], reject_by_annotation=None)[0] * 1e6
        if name in bad_names:
            ax.plot(times, data_uv, color=colors[i], linewidth=0.8, label=name, zorder=2)
        else:
            ax.plot(times, data_uv, color=colors[i], linewidth=0.5, alpha=0.7, zorder=1)
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
    for data in _unmasked_windows(masked, picks, window_samples, n_windows):
        deviation = 0.7413 * _iqr(data, axis=1)
        counts += _robust_channel_z(deviation, use_iqr=True) > robust_z_cutoff
        n_retained += 1
    bad_names = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    return {"std": bad_names}


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
    cross-channel median and divide by 1.4826 * cross-channel MAD. Flag only
    z > robust_z_cutoff. Each retained window is filtered separately to keep
    masked bursts out of neighboring windows. Accept an all-zero native
    reference, otherwise verify average reference. Temporary masking and the
    occurrence denominator follow bad_ch_by_std. Require sfreq > 100 Hz
    to evaluate noise above 50 Hz. Add detections to raw.info["bads"], preserving
    existing labels without duplicates. Input data/annotations are unchanged.

    With return_details=True, return (report, details). Details contain channel
    names, crop-relative start/stop seconds, a retained-window mask, and arrays
    of shape (channels, complete windows): ratio, robust_z, hf_mad_uv, lf_mad_uv,
    and violates (0/1). Masked windows contain NaN, not a passing flag. For each
    channel, flagged_data_uv contains its original window waveforms in
    chronological order, matching the windows where violates == 1. The
    diagnostics describe this pass's channel set, before its new bad labels.
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
        if return_details:
            ratios, hf_mad, lf_mad = _prep_hf_ratio(data, raw.info["sfreq"], return_mads=True)
        else:
            ratios = _prep_hf_ratio(data, raw.info["sfreq"])
        z = _robust_channel_z(ratios)
        violates = z > robust_z_cutoff
        counts += violates
        n_retained += 1
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
    bad_names = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
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
    for data in _unmasked_windows(masked, picks, window_samples, n_windows):
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
        counts += (z > robust_z_cutoff).sum(axis=1) >= min_bad_bands
        n_retained += 1
    bad_names = _noisy_window_bads(raw, picks, counts, n_retained, occurrence_cutoff)
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad_names))
    return {"BBnoise": bad_names}


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
    project_dir = Path(__file__).resolve().parents[2]
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

    # %% Freshly loaded traces, before any filtering or line-noise removal
    unfiltered_sfreq = float(raw.info["sfreq"])
    unfiltered_start = int(round(590.0 * unfiltered_sfreq))
    unfiltered_stop = int(round(604.0 * unfiltered_sfreq))
    unfiltered_times = np.arange(unfiltered_start, unfiltered_stop) / unfiltered_sfreq
    # Retain these two original channels for later inspection of HF-flagged windows.
    unfiltered_pair_uv = raw.get_data(picks=["E7", "E106"], units="uV")
    unfiltered_traces_uv = unfiltered_pair_uv[:, unfiltered_start:unfiltered_stop]
    unfiltered_trace_fig, ax = plt.subplots(figsize=(14, 4), layout="constrained")
    for channel, trace, color in zip(("E7", "E106"), unfiltered_traces_uv,
                                     ("tab:red", "tab:blue")):
        ax.plot(unfiltered_times, trace, label=channel, color=color, linewidth=0.8)
    ax.set(xlim=(590, 604), ylabel="Voltage (µV)",
           xlabel="Time from start of cropped recording (s)",
           title="E7 and E106: freshly loaded data, before filtering or line-noise removal",
           xticks=np.arange(590, 605, 2))
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()

    # %% All freshly loaded channels, overlaid with 50% transparency
    all_unfiltered_traces_uv = raw.get_data(
        picks=raw.ch_names, start=unfiltered_start, stop=unfiltered_stop, units="uV",
    )
    all_unfiltered_trace_fig, ax = plt.subplots(figsize=(16, 5), layout="constrained")
    channel_colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(raw.ch_names)))
    for channel, trace, color in zip(raw.ch_names, all_unfiltered_traces_uv, channel_colors):
        ax.plot(unfiltered_times, trace, color=color, alpha=0.5, linewidth=0.7, label=channel)
    ax.set(xlim=(590, 604), ylabel="Voltage (µV)",
           xlabel="Time from start of cropped recording (s)",
           title=f"All {len(raw.ch_names)} channels: freshly loaded, unfiltered data · 50% transparency",
           xticks=np.arange(590, 605, 2))
    ax.grid(alpha=0.2)
    plt.show()

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
    hf_report, hf_details = bad_ch_by_HFnoise(raw_bads, return_details=True)
    bad_ch_report.update(hf_report)
    plot_bad_ch_report(raw_bads, bad_ch_report)

    # %% Temporary HF-noise diagnostics from the same detection pass
    # e.g. hf_flagged_windows["E7"]["data_uv"][0] is its first flagged waveform.
    hf_flagged_windows = {
        name: {
            "start_s": hf_details["start_s"][hf_details["violates"][i] == 1],
            "stop_s": hf_details["stop_s"][hf_details["violates"][i] == 1],
            "data_uv": hf_details["flagged_data_uv"][name],
        }
        for i, name in enumerate(hf_details["ch_names"])
        if np.any(hf_details["violates"][i] == 1)
    }
    for channel in ("E7", "E26", "E106"):
        i = hf_details["ch_names"].index(channel)
        flags = hf_details["violates"][i]
        print(f"{channel}: {np.nansum(flags):.0f}/{hf_details['retained'].sum()} "
              f"windows exceed z > 5 ({np.nanmean(flags):.1%}); "
              f"median ratio={np.nanmedian(hf_details['ratio'][i]):.3f}, "
              f"HF MAD={np.nanmedian(hf_details['hf_mad_uv'][i]):.3f} uV, "
              f"LF MAD={np.nanmedian(hf_details['lf_mad_uv'][i]):.3f} uV")

    sfreq = float(raw_bads.info["sfreq"])
    spectrogram_params = dict(
        frequency_range=[1, sfreq / 2],
        time_bandwidth=2,
        num_tapers=3,
        window_params=[2, 2],  # Match the detector's non-overlapping windows.
        min_nfft=int(2 ** np.ceil(np.log2(sfreq / 0.05))),
        detrend_opt="constant",
        plot_on=False,
        verbose=False,
    )
    hf_spectra_db = {}
    for channel in ("E7", "E106"):
        power, times, freqs = multitaper_spectrogram(
            raw_bads.get_data(picks=[channel], units="uV")[0],
            sfreq, **spectrogram_params,
        )
        hf_spectra_db[channel] = 10 * np.log10(np.where(power > 0, power, np.nan))
    np.testing.assert_allclose(times, (hf_details["start_s"] + hf_details["stop_s"]) / 2)

    e7 = hf_details["ch_names"].index("E7")
    window_edges = np.r_[hf_details["start_s"], hf_details["stop_s"][-1]]
    visible_freqs = freqs <= 150
    e7_spectrum = hf_spectra_db["E7"][visible_freqs]
    vmin, vmax = np.nanpercentile(e7_spectrum, [5, 98])
    hf_timeline_fig, axes = plt.subplots(
        3, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1.5]}, layout="constrained",
    )
    mesh = axes[0].pcolormesh(times, freqs[visible_freqs], e7_spectrum, shading="auto",
                              vmin=vmin, vmax=vmax, cmap="cet_rainbow4", rasterized=True)
    axes[0].axhline(50, color="black", linestyle=":", linewidth=1)
    axes[0].set(title="E7: line-noise cleaned, 0.3 Hz high-pass; 2 s windows",
                ylabel="Frequency (Hz)")
    hf_timeline_fig.colorbar(mesh, ax=axes, label=r"PSD (dB re 1 $\mu V^2$/Hz)")
    flags = hf_details["violates"][e7]
    axes[1].step(window_edges, np.r_[flags, flags[-1]], where="post", color="tab:red")
    axes[1].set(yticks=[0, 1], yticklabels=["Pass", "Violate"], ylim=(-0.15, 1.15),
                ylabel="z > 5")
    z = hf_details["robust_z"][e7]
    axes[2].step(window_edges, np.r_[z, z[-1]], where="post", color="tab:blue")
    axes[2].axhline(5, color="black", linestyle=":", label="Cutoff = 5")
    axes[2].set(ylabel="Robust z", xlabel="Time from start of cropped recording (s)",
                xlim=(0, window_edges[-1]))
    for start, stop in zip(hf_details["start_s"][~hf_details["retained"]],
                           hf_details["stop_s"][~hf_details["retained"]]):
        for ax in axes[1:]:
            ax.axvspan(start, stop, color="0.85", zorder=0)
    axes[1].set_title("Gray intervals: masked, not evaluated", fontsize=9, loc="left")
    axes[2].legend(loc="upper right")
    plt.show()

    hf_spectra_fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True,
                                        layout="constrained")
    for ax, keep, title in zip(
        axes, (np.ones(len(times), dtype=bool), hf_details["retained"]),
        ("All complete windows", "Only windows retained by HF detection"),
    ):
        for channel, color in (("E7", "tab:red"), ("E106", "tab:blue")):
            ax.semilogx(freqs, hf_spectra_db[channel][:, keep].mean(axis=1),
                        label=channel, color=color, linewidth=1.3)
        ax.axvline(50, color="black", linestyle=":", label="50 Hz split")
        ax.set(title=title, xlabel="Frequency (Hz)", xlim=(1, sfreq / 2))
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel(r"Mean PSD in dB (re 1 $\mu V^2$/Hz)")
    hf_spectra_fig.suptitle("E7 vs E106: time average of multitaper PSD in dB")
    plt.show()

    # Separate numerator and denominator to explain a high ratio without an HF burst.
    hf_components_fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                                           layout="constrained")
    for ax, metric, label in zip(axes, ("hf_mad_uv", "lf_mad_uv", "ratio"),
                                 ("HF MAD (µV)", "LF MAD (µV)", "HF / LF MAD")):
        for channel, color in (("E7", "tab:red"), ("E106", "tab:blue")):
            i = hf_details["ch_names"].index(channel)
            values = hf_details[metric][i]
            ax.step(window_edges, np.r_[values, values[-1]], where="post",
                    color=color, label=channel, linewidth=1)
        ax.set(ylabel=label, yscale="log")
        ax.grid(alpha=0.2)
    ratio_cutoff = np.full(len(times), np.nan)
    retained_ratios = hf_details["ratio"][:, hf_details["retained"]]
    ratio_cutoff[hf_details["retained"]] = (
        np.median(retained_ratios, axis=0)
        + 5 * np.maximum(1.4826 * _mad(retained_ratios, axis=0), np.finfo(float).eps)
    )
    axes[2].step(window_edges, np.r_[ratio_cutoff, ratio_cutoff[-1]], where="post",
                 color="black", linestyle=":", label="Window-specific z = 5 cutoff")
    axes[0].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    axes[2].set(xlabel="Time from start of cropped recording (s)",
                xlim=(0, window_edges[-1]))
    hf_components_fig.suptitle("E7 vs E106: the two MADs and their ratio (masked windows are gaps)")
    plt.show()

    # %% Zoom: sustained E7 HF-noise violations near 600 s
    comparison_start, comparison_stop = 590.0, 604.0
    comparison_windows = ((hf_details["start_s"] >= comparison_start)
                          & (hf_details["stop_s"] <= comparison_stop))
    hf_comparison_fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                                           sharey="row", layout="constrained")
    for row, fmax in enumerate((150, 15)):
        selected_freqs = (freqs >= 1) & (freqs <= fmax)
        spectra = [hf_spectra_db[channel][np.ix_(selected_freqs, comparison_windows)]
                   for channel in ("E7", "E106")]
        vmin, vmax = np.nanpercentile(np.stack(spectra), [5, 98])
        for col, (channel, spectrum) in enumerate(zip(("E7", "E106"), spectra)):
            ax = axes[row, col]
            mesh = ax.pcolormesh(times[comparison_windows], freqs[selected_freqs],
                                 spectrum, shading="auto", cmap="cet_rainbow4",
                                 vmin=vmin, vmax=vmax, rasterized=True)
            ax.set(xlim=(comparison_start, comparison_stop), ylim=(1, fmax),
                   title=f"{channel}: {'1–150 Hz' if row == 0 else 'LF detail, 1–15 Hz'}")
            if row == 0:
                ax.axhline(50, color="black", linestyle=":", linewidth=1)
            else:
                ax.set_xlabel("Time from start of cropped recording (s)")
        axes[row, 0].set_ylabel("Frequency (Hz)")
        hf_comparison_fig.colorbar(mesh, ax=list(axes[row]),
                                   label=r"PSD (dB re 1 $\mu V^2$/Hz)")
    hf_comparison_fig.suptitle(
        "sub-02, 590–604 s: E7 vs E106\n"
        "Line-noise cleaned, 0.3 Hz high-pass; shared color scale within each row"
    )
    plt.show()

    # %% Filtered time traces over the same interval (no filtering after cropping)
    interval_start_sample = int(round(comparison_start * sfreq))
    interval_stop_sample = int(round(comparison_stop * sfreq))
    interval_times = np.arange(interval_start_sample, interval_stop_sample) / sfreq
    interval_traces_uv = raw_bads.get_data(
        picks=["E7", "E106"], start=interval_start_sample,
        stop=interval_stop_sample, units="uV",
    )
    hf_interval_traces_fig, ax = plt.subplots(figsize=(14, 4), layout="constrained")
    for channel, trace, color in zip(("E7", "E106"), interval_traces_uv,
                                     ("tab:red", "tab:blue")):
        ax.plot(interval_times, trace, label=channel, color=color, linewidth=0.8)
    ax.set(xlim=(comparison_start, comparison_stop), ylabel="Voltage (µV)",
           xlabel="Time from start of cropped recording (s)",
           title="E7 and E106: line-noise cleaned, 0.3 Hz high-pass (no vertical offsets)",
           xticks=np.arange(comparison_start, comparison_stop + 1, 2))
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()

    # %% Cross-channel z-score distributions, one panel per 2-second window
    comparison_indices = np.flatnonzero(comparison_windows)
    interval_z = hf_details["robust_z"][:, comparison_windows]
    z_bins = np.arange(np.floor(np.nanmin(interval_z)), np.ceil(np.nanmax(interval_z)) + 1)
    hf_z_distributions_fig, axes = plt.subplots(
        len(comparison_indices), 1, figsize=(13, 13), sharex=True, layout="constrained",
    )
    for ax, window in zip(axes, comparison_indices):
        z = hf_details["robust_z"][:, window]
        ax.hist(z, bins=z_bins, color="0.65", edgecolor="white")
        ax.axvline(5, color="black", linestyle=":", label="Cutoff = 5")
        for channel, color, height in (("E7", "tab:red", 0.90), ("E106", "tab:blue", 0.76)):
            value = z[hf_details["ch_names"].index(channel)]
            ax.scatter(value, height, transform=ax.get_xaxis_transform(), s=35,
                       color=color, edgecolor="white", label=channel, zorder=3)
        exceeding = [name for name, value in zip(hf_details["ch_names"], z) if value > 5]
        names_text = "\n".join(", ".join(exceeding[i:i + 6]) for i in range(0, len(exceeding), 6))
        ax.text(1.02, 0.90, f"z > 5: {len(exceeding)} channels\n{names_text or 'None'}",
                transform=ax.transAxes, va="top", fontsize=10)
        start, stop = hf_details["start_s"][window], hf_details["stop_s"][window]
        ax.set_ylabel(f"{start:g}–{stop:g} s\nChannels")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("HF/LF robust z-score across channels")
    hf_z_distributions_fig.suptitle("sub-02: HF-noise z-score distributions, 590-604 s")
    plt.show()

    # %% Spectrograms around the head, independently scaled within each channel
    # Spread panel centers for readability; gray lines link to actual sensor locations.
    panel_centers = {
        "E22": (0.36, 0.84), "E9": (0.64, 0.84),
        "E26": (0.12, 0.73), "E122": (0.88, 0.68),
        "E33": (0.12, 0.55), "E7": (0.36, 0.60), "E106": (0.64, 0.60),
        "E36": (0.12, 0.37), "E104": (0.88, 0.44),
        "E52": (0.36, 0.37), "E92": (0.64, 0.37),
        "E70": (0.36, 0.16), "E83": (0.64, 0.16),
    }
    hf_spatial_fig = plt.figure(figsize=(18, 14))
    head_ax = hf_spatial_fig.add_axes([0.05, 0.07, 0.90, 0.83])
    sensor_info = raw_bads.info.copy()
    sensor_info["bads"] = []
    mne.viz.plot_sensors(sensor_info, ch_type="eeg", axes=head_ax, show=False,
                         show_names=False, pointsize=5, linewidth=0.1, title="")
    head_ax.collections[0].set_alpha(0.15)
    for line in head_ax.lines:
        line.set(color="0.8", linewidth=1)
    eeg_names = [raw_bads.ch_names[i] for i in mne.pick_types(sensor_info, eeg=True, exclude=[])]
    sensor_positions = head_ax.collections[0].get_offsets()
    reference_position = sensor_positions[eeg_names.index("Cz")]
    head_ax.scatter(*reference_position, s=25, color="0.4", zorder=3)
    head_ax.annotate("Cz (reference)", reference_position, xytext=(0, -12),
                     textcoords="offset points", ha="center", color="0.4", fontsize=9)
    spatial_params = dict(spectrogram_params, frequency_range=[1, 150])
    hf_spatial_spectra_db = {}
    for channel, (x, y) in panel_centers.items():
        data_uv = raw_bads.get_data(picks=[channel], start=interval_start_sample,
                                    stop=interval_stop_sample, units="uV")[0]
        power, local_times, spatial_freqs = multitaper_spectrogram(data_uv, sfreq, **spatial_params)
        spectrum = 10 * np.log10(np.where(power > 0, power, np.nan))
        hf_spatial_spectra_db[channel] = spectrum
        vmin, vmax = np.nanpercentile(spectrum, [5, 99.5])
        position = sensor_positions[eeg_names.index(channel)]
        head_ax.annotate("", xy=position, xycoords="data", xytext=(x, y),
                         textcoords=hf_spatial_fig.transFigure,
                         arrowprops=dict(arrowstyle="-", color="0.7", linewidth=0.8))
        head_ax.scatter(*position, s=20, color="0.5", zorder=3)
        ax = hf_spatial_fig.add_axes([x - 0.095, y - 0.065, 0.19, 0.13])
        ax.pcolormesh(local_times + comparison_start, spatial_freqs, spectrum,
                      shading="auto", cmap="cet_rainbow4", vmin=vmin, vmax=vmax, rasterized=True)
        ax.axhline(50, color="black", linestyle=":", linewidth=0.8)
        i = hf_details["ch_names"].index(channel)
        n_flags = int(np.nansum(hf_details["violates"][i, comparison_windows]))
        color = "tab:red" if channel == "E7" else "tab:blue" if channel == "E106" else "black"
        ax.set_title(f"{channel} · {n_flags}/7 flags", fontsize=11, color=color)
        ax.set(yscale="log", ylim=(1, 150), xlim=(comparison_start, comparison_stop),
               xticks=[comparison_start, 597, comparison_stop], yticks=[1, 10, 50, 150],
               yticklabels=["1", "10", "50", "150"])
        ax.tick_params(labelsize=7, length=2, pad=1)
    hf_spatial_fig.suptitle(
        "sub-02: spatial spectrogram comparison, 590–604 s\n"
        "Line-noise cleaned, 0.3 Hz high-pass · anterior at top, left hemisphere at left",
        fontsize=16, y=0.985,
    )
    hf_spatial_fig.text(
        0.5, 0.025, "Each channel uses its own 5th–99.5th percentile dB color scale. "
        "Axes: time (s), frequency (Hz, log scale). Dotted line: 50 Hz.\n"
        "Flags are per-window z > 5; gray connectors show the actual sensor positions.",
        ha="center", fontsize=11,
    )
    plt.show()

    # %% Concatenate every E7 HF-noise violation, with matching E106 windows
    e7_flagged_indices = np.flatnonzero(
        hf_details["violates"][hf_details["ch_names"].index("E7")] == 1
    )
    e7_flagged_start_s = hf_details["start_s"][e7_flagged_indices]
    e7_flagged_stop_s = hf_details["stop_s"][e7_flagged_indices]
    e7_flagged_edges_s = np.r_[0, np.cumsum(e7_flagged_stop_s - e7_flagged_start_s)]
    e7_flagged_pair_uv = []
    hf_flagged_pair_fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True,
                                            sharey=True, layout="constrained")
    for offset, start, stop in zip(e7_flagged_edges_s[:-1], e7_flagged_start_s, e7_flagged_stop_s):
        first, last = int(round(start * sfreq)), int(round(stop * sfreq))
        traces = unfiltered_pair_uv[:, first:last]
        e7_flagged_pair_uv.append(traces)
        concatenated_times = offset + np.arange(last - first) / sfreq
        for ax, trace, color in zip(axes, traces, ("tab:red", "tab:blue")):
            # Separate plot calls avoid artificial jumps between noncontiguous windows.
            ax.plot(concatenated_times, trace, color=color, linewidth=0.6)
    for ax, channel in zip(axes, ("E7", "E106")):
        for boundary in e7_flagged_edges_s[1:-1]:
            ax.axvline(boundary, color="0.45", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set(title=channel, ylabel="Voltage (µV)", xlim=(0, e7_flagged_edges_s[-1]))
    axes[-1].set_xlabel("Concatenated time (s); dotted lines separate original 2-second windows")
    hf_flagged_pair_fig.suptitle(
        f"All {len(e7_flagged_indices)} E7 HF/LF z > 5 windows, in chronological order "
        f"({e7_flagged_edges_s[-1]:g} s total)\n"
        "Freshly loaded, unfiltered traces; windows selected by the line-cleaned, 0.3 Hz high-pass detector"
    )
    plt.show()
