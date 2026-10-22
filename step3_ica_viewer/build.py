"""Prepare immutable diagnostic assets from an exact fitted ICA and its epochs."""

import base64
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from scipy.ndimage import median_filter
from scipy.signal import periodogram

from .envelope import _write_reconstruction


BUILD_VERSION = "1.2-reconstructed-envelope"
LAGS_MS = (4, 20, 40, 80, 160)
THRESHOLD = 5.0
MERGE_GAP_MS = 300
ICLABEL_CLASSES = (
    "brain", "muscle artifact", "eye blink", "heart beat", "line noise",
    "channel noise", "other",
)


def _robust_parameters(values):
    """Return finite center/scale, reporting empty or zero-MAD support as zero."""
    if not len(values):
        return 0.0, 0.0
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    return center, scale


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.column_stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _groups(intervals, gap_samples):
    groups = []
    for start, end in intervals:
        if groups and start - groups[-1][1] <= gap_samples:
            groups[-1][1] = int(end)
            groups[-1][2] += 1
        else:
            groups.append([int(start), int(end), 1])
    return groups


def _time_segments(epoch_table, n_samples):
    """Keep legitimate epoch joins, and split wherever original time is missing."""
    starts = epoch_table.retained_start_sample.to_numpy(dtype=int)
    ends = epoch_table.retained_end_sample.to_numpy(dtype=int)
    if (not len(starts) or starts[0] != 0 or ends[-1] != n_samples
            or np.any(ends <= starts) or not np.array_equal(starts[1:], ends[:-1])):
        raise ValueError("Epoch mapping must partition the retained source samples.")
    segments, discontinuities = [], []
    segment_start = 0
    for row in range(1, len(epoch_table)):
        if not bool(epoch_table.iloc[row].contiguous_with_previous):
            boundary = int(starts[row])
            segments.append([segment_start, boundary])
            segment_start = boundary
            previous, current = epoch_table.iloc[row - 1], epoch_table.iloc[row]
            discontinuities.append(dict(
                retained_sample=boundary, retained_s=boundary / 250.0,
                original_previous_end_s=float(previous.original_end_s),
                original_next_start_s=float(current.original_start_s),
            ))
    segments.append([segment_start, n_samples])
    return segments, discontinuities


def _analyze_component(source, segments, sfreq):
    """Detect on float64 samples; no delta or grouping may cross a time gap."""
    center, source_mad = _robust_parameters(source)
    # A zero-MAD trace retains its native variations for display. Mark the
    # fallback explicitly; zero-MAD delta scales cannot trigger candidates.
    scale = source_mad if source_mad > 0 else 1.0
    x = (source - center) / scale
    lags = [round(ms * sfreq / 1000) for ms in LAGS_MS]
    deltas = np.full((len(lags), len(x)), np.nan)
    delta_medians, delta_scales, degenerate_lags = [], [], []
    for index, lag in enumerate(lags):
        differences = [x[start + lag:end] - x[start:end - lag]
                       for start, end in segments if end - start > lag]
        values = np.concatenate(differences) if differences else np.empty(0)
        median, mad = _robust_parameters(values)
        delta_medians.append(median)
        delta_scales.append(mad)
        if mad <= 0:
            degenerate_lags.append(LAGS_MS[index])
            continue
        for start, end in segments:
            if end - start <= lag:
                continue
            difference = x[start + lag:end] - x[start:end - lag]
            aligned = start + lag // 2
            deltas[index, aligned:aligned + len(difference)] = (difference - median) / mad

    absolute = np.nan_to_num(np.abs(deltas), nan=0.0)
    score = absolute.max(axis=0)
    intervals, groups = [], []
    alternate_counts = {150: 0, 500: 0}
    for start, end in segments:
        local = _runs(score[start:end] >= THRESHOLD) + start
        intervals.extend(local.tolist())
        groups.extend(_groups(local, round(MERGE_GAP_MS * sfreq / 1000)))
        for milliseconds in alternate_counts:
            # Preserve the prototype's integer truncation at the 150-ms scale.
            gap = int(milliseconds * sfreq / 1000)
            alternate_counts[milliseconds] += len(_groups(local, gap))

    # This descriptive contrast is distinct from the displayed pre/post levels.
    # Median-filter each continuous block separately, excluding missing context.
    contrast_z = np.full(len(x), np.nan)
    contrast_parts = []
    for start, end in segments:
        if end - start <= 50:
            continue
        level = median_filter(x[start:end], size=21, mode="reflect")
        contrast = level[30:] - level[:-30]
        contrast_parts.append((start, end, contrast))
    contrast_values = [contrast[(10 if start else 0):
                                (len(contrast) - 10 if end < len(x) else len(contrast))]
                       for start, end, contrast in contrast_parts]
    contrast_median, contrast_scale = _robust_parameters(
        np.concatenate(contrast_values) if contrast_values else np.empty(0)
    )
    if contrast_scale > 0:
        for start, end, contrast in contrast_parts:
            contrast_z[start + 15:end - 15] = (contrast - contrast_median) / contrast_scale
            # A removed-time join must not provide invented reflected context.
            if start:
                contrast_z[start + 15:start + 25] = np.nan
            if end < len(x):
                contrast_z[end - 25:end - 15] = np.nan

    events, run_records = [], []
    for start, end in intervals:
        peak = start + int(np.argmax(score[start:end]))
        run_records.append(dict(start_s=start / sfreq, end_s=end / sfreq,
                                peak_s=peak / sfreq, peak_score=float(score[peak])))
    segment_index = 0
    for event_id, (start, end, n_runs) in enumerate(groups, 1):
        while start >= segments[segment_index][1]:
            segment_index += 1
        segment_start, segment_end = segments[segment_index]
        peak = start + int(np.argmax(score[start:end]))
        winning = int(np.argmax(absolute[:, peak]))
        fast_z = float(absolute[:2, start:end].max())
        offset_z = float(contrast_z[peak]) if np.isfinite(contrast_z[peak]) else None
        # Clip only at genuine recording edges; incomplete context at an internal
        # removed-time join is unavailable rather than borrowing another epoch.
        pre_available = peak - 25 >= segment_start or segment_start == 0
        post_available = peak + 26 <= segment_end or segment_end == len(x)
        pre = x[max(segment_start, peak - 25):max(segment_start, peak - 5)]
        post = x[min(segment_end, peak + 5):min(segment_end, peak + 26)]
        shape = ("fast edge + level change" if offset_z is not None and abs(offset_z) >= 5
                 else "fast transient") if fast_z >= 5 else "broader change"
        events.append(dict(
            event_id=event_id, start_s=start / sfreq, end_s=end / sfreq,
            duration_s=(end - start) / sfreq, peak_s=peak / sfreq,
            peak_score=float(score[peak]), peak_lag_ms=LAGS_MS[winning],
            n_runs=n_runs, fast_z=fast_z, median_contrast_z=offset_z,
            pre_level=float(np.median(pre)) if pre_available and len(pre) else None,
            post_level=float(np.median(post)) if post_available and len(post) else None,
            shape=shape,
        ))
    durations = [event["duration_s"] for event in events]
    summary = dict(
        n_runs=len(intervals), n_events=len(events),
        gap150_events=alternate_counts[150], gap500_events=alternate_counts[500],
        n_gt1s=sum(value > 1 for value in durations),
        max_duration_s=max(durations, default=0.0),
    )
    return dict(
        source_median=center, source_scale=scale, source_mad=source_mad,
        source_normalization_valid=source_mad > 0,
        delta_medians=delta_medians, delta_scales=delta_scales,
        degenerate_delta_lags_ms=degenerate_lags,
        events=events, threshold_runs=run_records, summary=summary,
    ), x


def _add_shared_context(components, n_samples, sfreq):
    """Count unique covering ICs at each target peak with half-open intervals."""
    presence = np.zeros((len(components), n_samples), dtype=bool)
    for component in components:
        for event in component["events"]:
            start, end = (round(event[key] * sfreq) for key in ("start_s", "end_s"))
            presence[component["id"], start:end] = True
    minimum = len(components) // 4 + 1
    for component in components:
        for event in component["events"]:
            peak = round(event["peak_s"] * sfreq)
            covering = np.flatnonzero(presence[:, peak]).tolist()
            event.update(context_components=covering, context_count=len(covering),
                         context_fraction=len(covering) / len(components),
                         shared=len(covering) >= minimum)
        remaining = [event for event in component["events"] if not event["shared"]]
        component["summary"].update(
            n_events_adjusted=len(remaining),
            n_runs_adjusted=sum(event["n_runs"] for event in remaining),
            max_duration_s_adjusted=max((event["duration_s"] for event in remaining), default=0.0),
            n_gt1s_adjusted=sum(event["duration_s"] > 1 for event in remaining),
            n_shared=len(component["events"]) - len(remaining),
        )
    return dict(
        n_components=len(components), fraction_threshold=0.25, min_components=minimum,
        rule="interval_overlap_at_event_peak", interval_bounds="start <= target peak < end",
        includes_iclabel_excluded=True, count_target_component=True,
        adjusted_runs_definition="Omit all threshold runs in shared grouped events.",
        rate_denominator_seconds=n_samples / sfreq,
    )


def _median_psd(sources, sfreq):
    """Median full-epoch PSD in native ICA units; never join separate epochs.

    sources has shape (epochs, components, samples). Take the pointwise median
    of linear power densities across epochs before converting to dB.
    """
    frequencies, power = periodogram(
        sources, fs=sfreq, window="hamming", detrend="constant",
        scaling="density", axis=-1,
    )
    keep = frequencies <= min(50.0, sfreq / 2)
    median_power = np.median(power[..., keep], axis=0)
    return frequencies[keep], 10 * np.log10(np.maximum(median_power, np.finfo(float).tiny))


def _topomaps(cache_dir, ica):
    maps = ica.get_components()
    topomap_dir = cache_dir / "topomaps"
    topomap_dir.mkdir(exist_ok=True)
    for component in range(ica.n_components_):
        maximum = float(np.max(np.abs(maps[:, component])))
        values = maps[:, component] / maximum if maximum else np.zeros(len(maps))
        fig = Figure(figsize=(2.4, 2.4), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)
        mne.viz.plot_topomap(values, ica.info, axes=ax, vlim=(-1, 1), cmap="RdBu_r",
                             sensors=True, contours=6, outlines="head", res=128, show=False)
        sensors = [item for item in ax.collections if isinstance(item, PathCollection)
                   and item.get_offsets().shape == (len(ica.ch_names), 2)]
        if len(sensors) != 1:
            raise RuntimeError("Could not identify topomap electrode markers.")
        sensors[0].set_edgecolor("#39ff14")
        sensors[0].set_facecolor("#000000")
        sensors[0].set_sizes([10.0])
        sensors[0].set_linewidths([0.65])
        fig.savefig(topomap_dir / f"ICA{component:03d}.png", dpi=150, facecolor="white")
        fig.clear()


def build_viewer(cache_dir, ica, fit_epochs, iclabel_table, *, recording_filename,
                 reference_channel, epoch_table, review_id, initial_excluded):
    """Write a generalized viewer cache without changing inputs or exclusions.

    ``epoch_table`` maps the concatenated fitting samples to original Raw time.
    The caller owns durable identity, input archiving, and the decision JSON.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sfreq = float(fit_epochs.info["sfreq"])
    n_epochs, n_times = len(fit_epochs), len(fit_epochs.times)
    n_components = int(ica.n_components_)
    if sfreq != 250 or n_times != 500 or not n_epochs or not n_components:
        raise ValueError("Viewer v1 requires nonempty 250-Hz, two-second fitting epochs.")
    if ica.noise_cov is not None:
        raise ValueError("Viewer v1 does not support covariance-whitened ICA fits.")
    if len(epoch_table) != n_epochs:
        raise ValueError("Epoch mapping does not match the fitting epochs.")
    labels = iclabel_table.set_index("component")
    if sorted(labels.index.tolist()) != list(range(n_components)):
        raise ValueError("ICLabel table must contain each component exactly once.")
    source_epochs = ica.get_sources(fit_epochs).get_data(copy=True)
    if not np.isfinite(source_epochs).all():
        raise ValueError("ICA sources contain nonfinite samples.")
    psd_frequencies, psd_db = _median_psd(source_epochs, sfreq)
    sources = source_epochs.transpose(1, 0, 2).reshape(n_components, -1)
    segments, discontinuities = _time_segments(epoch_table, sources.shape[1])
    _write_reconstruction(cache_dir, ica, fit_epochs, sources, segments)
    mixing_uv = ica.pre_whitener_ * ica.get_components() * 1e6
    voltage_channels = np.argmax(np.abs(mixing_uv), axis=0)
    automatic = labels.index[labels.iclabel_auto_exclude].astype(int).tolist()
    payload = dict(
        build_version=BUILD_VERSION, recording_filename=str(recording_filename),
        reference_channel=reference_channel, sfreq=sfreq, n_epochs=n_epochs,
        epoch_duration_s=n_times / sfreq, duration_s=sources.shape[1] / sfreq,
        highpass=float(fit_epochs.info["highpass"]), lowpass=float(fit_epochs.info["lowpass"]),
        source_units="ICA activation units (AU)", source_display_scale=5.0,
        psd=dict(
            freqs_hz=psd_frequencies.tolist(), fmax_hz=min(50.0, sfreq / 2),
            method="Median across fitting epochs; full-epoch Hamming periodograms, mean removed.",
            power_units="AU²/Hz", frequency_resolution_hz=sfreq / n_times,
            aggregation="median_linear_power_then_db",
        ),
        lags_ms=list(LAGS_MS), threshold=THRESHOLD, merge_gap_ms=MERGE_GAP_MS,
        review_id=review_id, iclabel_excluded=automatic,
        initial_excluded=list(initial_excluded), segments=segments,
        discontinuities=discontinuities, epoch_table=epoch_table.to_dict("records"),
        components=[],
    )
    data_dir = cache_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for component in range(n_components):
        record, normalized = _analyze_component(sources[component], segments, sfreq)
        row = labels.loc[component]
        record.update(
            id=component, name=f"ICA{component:03d}",
            psd_db=psd_db[component].tolist(),
            voltage_channel=ica.ch_names[voltage_channels[component]],
            voltage_uV_per_source_unit=float(mixing_uv[voltage_channels[component], component]),
            iclabel_auto_exclude=bool(row.iclabel_auto_exclude),
            iclabel_label=str(row.winning_label), iclabel_score=float(row.winning_probability),
            iclabel_probabilities={label: float(row[label]) for label in ICLABEL_CLASSES},
        )
        for event in record["events"]:
            event["component"] = component
        payload["components"].append(record)
        encoded = base64.b64encode(normalized.astype("<f4").tobytes()).decode("ascii")
        (data_dir / f"ICA{component:03d}.js").write_text(
            f'receiveICSignal({component},"{encoded}");\n', encoding="utf-8"
        )
    payload["context"] = _add_shared_context(payload["components"], sources.shape[1], sfreq)
    payload["method"] = (
        "For each IC and lag (4, 20, 40, 80, 160 ms), subtract the temporal median "
        "of valid lagged differences and divide by 1.4826 × MAD. A candidate occurs "
        "where any |delta Z| ≥ 5. Merge contiguous threshold runs when the quiet gap "
        "is ≤300 ms. Deltas, shape windows, runs and groups do not cross removed-time "
        "gaps. Shape labels are descriptive and never filter candidates. “Fast” "
        "means the 4- or 20-ms delta reaches 5; “level change” uses a 21-sample "
        "median-filtered source and its 30-sample difference, robustly normalized, "
        "at the grouped peak. Displayed pre/post medians use −100 to −20 ms and "
        "+20 to +104 ms offsets. Original threshold runs are the primary count. "
        "Parenthesized counts omit entire shared groups. All data come from the "
        "exact annotation-clean 1–100-Hz fitting epochs at 250 Hz. Delta center "
        "alignment floors half-sample centers (4/20-ms lags are 2 ms early). "
        "Zero-MAD source/lag cases are reported and never divided by zero."
    )
    _topomaps(cache_dir, ica)
    summaries = [dict(component=c["id"], name=c["name"], **c["summary"],
                      iclabel_auto_exclude=c["iclabel_auto_exclude"],
                      iclabel_label=c["iclabel_label"], iclabel_score=c["iclabel_score"])
                 for c in payload["components"]]
    events = [dict(event, context_components=json.dumps(event["context_components"]))
              for component in payload["components"] for event in component["events"]]
    pd.DataFrame(summaries).to_csv(cache_dir / "summary.csv", index=False)
    # Keep a usable header even when no component has a detected event.
    event_columns = (list(events[0]) if events else [
        "component", "event_id", "start_s", "end_s", "duration_s", "peak_s",
        "peak_score", "peak_lag_ms", "n_runs", "fast_z", "median_contrast_z",
        "pre_level", "post_level", "shape", "context_count", "context_fraction",
        "context_components", "shared",
    ])
    pd.DataFrame(events, columns=event_columns).to_csv(cache_dir / "events.csv", index=False)
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    (cache_dir / "review_metadata.json").write_text(payload_json, encoding="utf-8")
    template = (Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8")
    # JSON resides in a script element: quoted recording paths must not close it.
    embedded_json = payload_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (cache_dir / "index.html").write_text(
        template.replace("__PAYLOAD_JSON__", embedded_json), encoding="utf-8"
    )
    return payload
