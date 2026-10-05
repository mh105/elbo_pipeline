"""Native-reference flatline segments and their later sample/window masks."""

from collections import defaultdict

import mne
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def _near_constant_segments(data):
    """Union all ten-sample blocks whose full range is at most one nanovolt."""
    if len(data) < 10:
        return []
    high = maximum_filter1d(data, size=10, origin=-5, mode="nearest")[:-9]
    low = minimum_filter1d(data, size=10, origin=-5, mode="nearest")[:-9]
    starts = np.flatnonzero(high - low <= 1e-9)
    if not len(starts):
        return []
    changes = np.zeros(len(data) + 1, dtype=np.int16)
    changes[starts] += 1
    changes[starts + 10] -= 1
    flat = np.cumsum(changes[:-1], dtype=np.int16) > 0
    return np.flatnonzero(np.diff(np.r_[False, flat, False])).reshape(-1, 2).tolist()


def _merge_segments(segments):
    """Merge touching or overlapping half-open sample intervals."""
    merged = []
    for start, stop in sorted(segments):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def _detect_flatline_data(raw, reference_channel):
    """Detect original/1-Hz-high-passed flat segments without changing Raw.

    Include finite existing bad EEG channels, but exempt the native reference.
    An already bad nonfinite channel is unavailable rather than flat. Store
    only flagged channels; all sample bounds are relative to this Raw.
    """
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_names = [raw.ch_names[i] for i in eeg]
    if reference_channel not in eeg_names:
        raise ValueError("reference_channel must name an EEG channel in raw.")
    result = {
        "ch_names": [], "segments": [], "raw_segments": [], "highpass_segments": [],
        "eeg_ch_names": eeg_names, "raw_ch_names": list(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]), "first_samp": int(raw.first_samp),
        "n_times": int(raw.n_times), "reference_channel": reference_channel,
        "min_samples": 10, "tolerance_v": 1e-9, "highpass_hz": 1.0,
        "global_channel_fraction_cutoff": 0.25,
    }
    for pick in eeg:
        name = raw.ch_names[pick]
        if name == reference_channel:
            continue
        data = raw.get_data(picks=[pick], reject_by_annotation=None)[0]
        if not np.isfinite(data).all():
            if name in raw.info["bads"]:
                continue
            raise ValueError("Nonfinite data in EEG channels; run bad_ch_by_nan first.")
        original = _near_constant_segments(data)
        filtered = []
        if len(data) >= 10:
            highpass = mne.filter.filter_data(
                data, raw.info["sfreq"], l_freq=1.0, h_freq=None, verbose=False,
            )
            filtered = _near_constant_segments(highpass)
        combined = _merge_segments(original + filtered)
        if combined:
            result["ch_names"].append(name)
            result["segments"].append(combined)
            result["raw_segments"].append(original)
            result["highpass_segments"].append(filtered)
    return result


def _validate_flatlines(raw, flatlines):
    """Reject changed channel/time alignment; allow changed bad-channel labels."""
    if flatlines is None:
        return
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    expected = {
        "raw_ch_names": list(raw.ch_names),
        "eeg_ch_names": [raw.ch_names[i] for i in eeg],
        "sfreq": float(raw.info["sfreq"]), "first_samp": int(raw.first_samp),
        "n_times": int(raw.n_times),
    }
    if not isinstance(flatlines, dict) or any(
        flatlines.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Flatline metadata does not match Raw channel/sample alignment.")
    names = flatlines.get("ch_names", [])
    segments = flatlines.get("segments", [])
    if (len(set(names)) != len(names) or len(names) != len(segments)
            or any(name not in expected["eeg_ch_names"] for name in names)
            or flatlines.get("reference_channel") not in expected["eeg_ch_names"]
            or flatlines["reference_channel"] in names):
        raise ValueError("Flatline channel names and segments do not match eligible EEG channels.")
    for intervals in segments:
        bounds = np.asarray(intervals)
        if not bounds.size:
            continue
        if (bounds.ndim != 2 or bounds.shape[1] != 2
                or not np.issubdtype(bounds.dtype, np.integer)
                or (bounds[:, 0] < 0).any() or (bounds[:, 1] > raw.n_times).any()
                or (bounds[:, 0] >= bounds[:, 1]).any()
                or (bounds[1:, 0] < bounds[:-1, 1]).any()):
            raise ValueError("Flatline segments must be ordered half-open integer sample intervals.")


def _segments_to_windows(segments, window_samples, n_windows):
    """Mark complete windows with any segment overlap, preserving boundary gaps."""
    flat = np.zeros(n_windows, dtype=bool)
    for start, stop in segments:
        first = start // window_samples
        last = min((stop - 1) // window_samples + 1, n_windows)
        flat[first:last] = True
    return flat


def _flatline_windows(raw, flatlines, picks, window_samples, n_windows):
    """Return a channel-by-window mask in the requested full-Raw pick order."""
    mask = np.zeros((len(picks), n_windows), dtype=bool)
    if flatlines is None:
        return mask
    _validate_flatlines(raw, flatlines)
    by_name = dict(zip(flatlines["ch_names"], flatlines["segments"]))
    for row, pick in enumerate(picks):
        mask[row] = _segments_to_windows(
            by_name.get(raw.ch_names[pick], []), window_samples, n_windows,
        )
    return mask


def _mask_eeg_picks(raw, reference_channel):
    """Select mask numerator/denominator channels, excluding bads and reference."""
    exclude = raw.info["bads"] + ([] if reference_channel is None else [reference_channel])
    picks = mne.pick_types(raw.info, eeg=True, exclude=exclude)
    if len(picks) == 0:
        raise ValueError("No usable EEG channels remain after exclusions.")
    return picks


def _global_flatline_segments(flatlines, eligible_ch_names, channel_fraction_cutoff=0.25):
    """Intervals with strictly more than the cutoff of eligible EEG channels flat.

    Count only the supplied non-bad, non-reference channels, simultaneously
    rather than per scoring window. Use this same pool for the denominator.
    """
    if not np.isfinite(channel_fraction_cutoff) or not 0 <= channel_fraction_cutoff <= 1:
        raise ValueError("channel_fraction_cutoff must be between 0 and 1.")
    if flatlines is None:
        return []
    changes = defaultdict(int)
    eligible = set(eligible_ch_names)
    for name, intervals in zip(flatlines["ch_names"], flatlines["segments"]):
        if name not in eligible:
            continue
        for start, stop in intervals:
            changes[start] += 1
            changes[stop] -= 1
    count, previous, spans = 0, 0, []
    threshold = channel_fraction_cutoff * len(eligible)
    for position in sorted(changes):
        if count > threshold and position > previous:
            if spans and spans[-1][1] == previous:
                spans[-1][1] = position
            else:
                spans.append([previous, position])
        count += changes[position]
        previous = position
    return spans


def _flatline_configurations(raw, flatlines):
    """Yield sample spans and locally flat full-Raw channel IDs.

    Cover the whole recording, merge identical neighbours, and omit local
    channels inside global dropout intervals. Reuse the intervals saved when
    mask_segment_by_flatline annotated the native data, despite later changes
    to bad labels. Without that mask call, derive intervals from current bads.
    Callers combine each local pattern with their whole-channel bads.
    """
    _validate_flatlines(raw, flatlines)
    if flatlines is None:
        yield 0, raw.n_times, ()
        return
    events = defaultdict(list)
    for name, intervals in zip(flatlines["ch_names"], flatlines["segments"]):
        pick = raw.ch_names.index(name)
        for start, stop in intervals:
            events[start].append((pick, True))
            events[stop].append((pick, False))
    global_segments = flatlines.get("global_segments")
    if global_segments is None:
        picks = _mask_eeg_picks(raw, flatlines["reference_channel"])
        global_segments = _global_flatline_segments(
            flatlines, [raw.ch_names[i] for i in picks],
            flatlines.get("global_channel_fraction_cutoff", 0.25),
        )
    for start, stop in global_segments:
        events[start].append((None, True))
        events[stop].append((None, False))
    positions = sorted({0, raw.n_times, *events})
    active, dropout = set(), False
    pending = None
    for start, stop in zip(positions[:-1], positions[1:]):
        for pick, begins in events[start]:
            if pick is None:
                dropout = begins
            elif begins:
                active.add(pick)
            else:
                active.discard(pick)
        pattern = () if dropout else tuple(sorted(active))
        if pending is not None and pending[2] == pattern:
            pending = pending[0], stop, pattern
        else:
            if pending is not None:
                yield pending
            pending = start, stop, pattern
    if pending is not None:
        yield pending
