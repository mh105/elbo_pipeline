"""Portable numerical outputs and plots of the known signals and their measured changes."""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def _fingerprint(capture):
    """Identify the supplied in-memory transform without duplicating its archive."""

    def encode(x):
        if isinstance(x, np.ndarray):
            digest = hashlib.sha256(np.ascontiguousarray(x)).hexdigest()
            return {"dtype": x.dtype.str, "shape": x.shape, "sha256": digest}
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(f"Unsupported capture value {type(x).__name__}")

    payload = json.dumps(vars(capture), sort_keys=True, separators=(",", ":"), default=encode)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _report_values(metrics, name):
    """Keep raw peak shifts in the archive; rank and aggregate interpretable shifts only."""
    if name in ("peak_shift_ms", "legacy_peak_shift_ms"):
        flag = name.replace("shift_ms", "interpretation_flag")
        return np.where(metrics[flag], np.nan, metrics[name])
    return metrics[name]


def _save_case(path, model, result, full_events):
    """Numeric arrays in a pickle-free NPZ, nested structure in its JSON manifest."""
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"a{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, (tuple, list)):
            return [encode(x) for x in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    records = []
    for original in result["records"]:
        r = original.copy()
        if "event_before" in r:
            metrics = r["metrics"]
            selections = {}
            for label, key, rule in (
                ("largest_peak_change", "peak_change_uv", "magnitude"),
                ("largest_peak_shift", "peak_shift_ms", "magnitude"),
                ("largest_legacy_peak_attenuation", "legacy_peak_amplitude_ratio", "minimum"),
                ("largest_legacy_peak_shift", "legacy_peak_shift_ms", "magnitude"),
            ):
                values = _report_values(metrics, key)
                ok = np.flatnonzero(np.isfinite(values))
                if ok.size:
                    i = ok[
                        np.argmin(values[ok]) if rule == "minimum" else np.argmax(abs(values[ok]))
                    ]
                    selections[label] = {
                        "index": int(i),
                        "sample": int(r["used"][i]),
                        "before": r["event_before"][i],
                        "after": r["event_after"][i],
                    }
            r["examples"] = selections
            if not full_events:
                r.pop("event_before")
                r.pop("event_after")
        records.append(r)
    data = dict(result, records=records, model=asdict(model))
    manifest = json.dumps(encode(data), allow_nan=False)
    np.savez_compressed(path, manifest=manifest, **arrays)


def load_case(path):
    """Load a case, its exact model inputs and its measurements."""
    with np.load(path, allow_pickle=False) as archive:

        def decode(node):
            return archive[node["array"]] if set(node) == {"array"} else node

        return json.loads(archive["manifest"].item(), object_hook=decode)


def _distribution(values):
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    result = {"count": int(data.size), "undefined": int(data.size - finite.size)}
    if finite.size:
        result.update(
            zip(
                ("minimum", "p05", "median", "p95", "maximum"),
                np.percentile(finite, (0, 5, 50, 95, 100)).tolist(),
            )
        )
    return result


def _summary_plot(summary):
    """Compare event-level median and 95th-percentile peak loss for each condition."""
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in summary
        if r["convention"] == "baseline_corrected"
        and r["stage"] in ("asr_increment", "whole_step2", "ica")
    ]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(rows) + 1.6)), layout="constrained")
    for i, r in enumerate(rows):
        values = r["events"]["peak_attenuation_percent"]
        if "median" in values:
            ax.plot([values["median"], values["p95"]], [i, i], color=".7")
            ax.scatter(values["median"], i, color="#158080", label="Median" if i == 0 else None)
            ax.scatter(
                values["p95"], i, color="#ce702d", label="95th percentile" if i == 0 else None
            )
    ax.set_yticks(
        range(len(rows)), [" / ".join(r[k] for k in ("family", "stage", "condition")) for r in rows]
    )
    ax.invert_yaxis()
    ax.axvline(0, color=".7", lw=0.7)
    ax.set(xlabel="Oriented peak loss at individual events (%)", title="Synthetic ERP preservation")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig


def write_assessment(directory, assay, models, *, full_events=False, plot=True):
    """Run all supplied models and write a summary and one numerical archive per model."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"Assessment destination already exists: {directory}")
    # A stopped run cannot occupy the final path. A concurrent completed directory
    # is nonempty, so renaming into place also protects the winning result.
    with TemporaryDirectory(dir=directory.parent, prefix=f".{directory.name}.incomplete-") as tmp:
        staging = Path(tmp)
        _write_report(staging, assay, models, full_events=full_events, plot=plot)
        staging.rename(directory)
    return directory


def _write_report(directory, assay, models, *, full_events, plot):
    cap = assay.capture
    if assay.step == 2:
        observed = cap.stage_availability
        status = {key: "available" if value else "unavailable" for key, value in observed.items()}
        status["asr_increment"] = cap.asr_status
        provenance = {
            "input_reference": cap.input_reference,
            "output_reference": cap.output_reference,
            "bad_channels": cap.bad_channels,
            "asr_reason": cap.asr_reason,
            "repairs_observed": cap.repairs_observed,
            "reference_check_observed": cap.reference_check_observed,
            "asr_probe_input": ("after_observed_filter_and_repairs" if observed["whole_step2"]
                                else "standalone_in_effective_reference") if observed["asr_increment"] else None,
            "versions": cap.versions,
        }
    else:
        status = {"ica": "available"}
        provenance = cap.metadata
    context = {
        "version": 1,
        "step": assay.step,
        "stages": status,
        "capture": provenance,
        "capture_fingerprint": _fingerprint(cap),
        "upstream_step2_fingerprint": _fingerprint(assay.upstream_step2)
        if assay.upstream_step2 is not None
        else None,
        "sfreq": assay.sf,
        "first_samp": assay.first,
        "n_times": assay.n,
        "channels": assay.names,
        "measurement_channels": [assay.names[p] for p in assay.picks],
        "windows": asdict(assay.windows),
        "contrast": assay.contrast,
        "coverage": assay.coverage,
        "full_event_waveforms": full_events,
    }

    def write(name, value):
        (directory / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    write("context.json", context)
    groups = defaultdict(lambda: defaultdict(list))
    index = []
    unavailable = []
    for i, model in enumerate(models):
        result = assay.assess(model)
        filename = f"case_{i:04d}.npz"
        _save_case(directory / filename, model, result, full_events)
        index.append({"name": model.name, "family": model.family, "file": filename})
        for r in result["records"]:
            if r["status"] != "measured":
                unavailable.append(
                    {
                        "model": model.name,
                        "stage": r["stage"],
                        "condition": r["condition"],
                        "reason": r["status"],
                    }
                )
                continue
            key = (model.family, r["stage"], r["condition"], r["convention"])
            for field, metrics in (("events", r["metrics"]), ("averages", r["average_metrics"])):
                for name in metrics:
                    groups[key][field, name].append(np.atleast_1d(_report_values(metrics, name)))
    summaries = []
    for key, items in groups.items():
        row = dict(zip(("family", "stage", "condition", "convention"), key), events={}, averages={})
        for (field, name), values in items.items():
            row[field][name] = _distribution(np.concatenate(values))
        summaries.append(row)
    write("summary.json", {"cases": index, "unavailable": unavailable, "distributions": summaries})
    if plot:
        import warnings

        import matplotlib.pyplot as plt

        try:
            fig = _summary_plot(summaries)
            try:
                fig.savefig(directory / "preservation.pdf")
            finally:
                plt.close(fig)
        except Exception as exc:  # noqa: BLE001 -- the numerical report is complete
            message = f"Preservation figure unavailable: {type(exc).__name__}: {exc}"
            (directory / "figure_warning.txt").write_text(message + "\n", encoding="utf-8")
            warnings.warn(message, RuntimeWarning, stacklevel=2)


def plot_waveforms(series, *, condition, convention="baseline_corrected", event_sample=None):
    """Overlay an average or a single stimulus from (label, case result, stage) tuples."""
    import matplotlib.pyplot as plt

    curves = []
    for label, result, stage in series:
        r = next(
            (
                r
                for r in result["records"]
                if r["status"] == "measured" and r.get("convention") == convention
                and r["stage"] == stage
                and r["condition"] == condition
            ),
            None,
        )
        if r is None:
            raise ValueError(
                f"No measured waveform for {label}, {stage}, {condition}, {convention}."
            )
        if event_sample is None:
            before, after = r["mean_before"], r["mean_after"]
        else:
            ix = np.flatnonzero(r["used"] == event_sample)
            if not ix.size:
                raise ValueError("The requested event has no complete epoch.")
            if "event_before" in r:
                before, after = r["event_before"][ix[0]], r["event_after"][ix[0]]
            else:
                example = next(
                    (e for e in r.get("examples", {}).values() if e["sample"] == event_sample), None
                )
                if example is None:
                    raise ValueError(
                        "Save full_events=True to redraw arbitrary individual waveforms."
                    )
                before, after = example["before"], example["after"]
        curves.append((label, result["time_s"], before, after, result["representative"]))
    if not curves:
        raise ValueError("Supply at least one measured waveform.")
    fig, ax = plt.subplots(layout="constrained")
    _, time, known, _, electrode = curves[0]
    shared = all(
        site == electrode and np.array_equal(t, time) and np.array_equal(b, known)
        for _, t, b, _, site in curves
    )
    for label, t, before, after, site in curves:
        (line,) = ax.plot(t, after * 1e6, label=f"{label} · after")
        if not shared:
            ax.plot(t, before * 1e6, ls="--", color=line.get_color(), label=f"{label} · before")
    if shared:
        ax.plot(
            time,
            known * 1e6,
            ls="--",
            lw=1.2,
            color="#303846",
            label="Known signal before correction",
        )
    ax.axhline(0, color=".7", lw=0.7)
    ax.axvline(0, color=".7", lw=0.7)
    sites = ", ".join(dict.fromkeys(c[-1] for c in curves))
    ax.set(
        xlabel="Time from event (s)", ylabel="Synthetic ERP (µV)", title=f"{condition} · {sites}"
    )
    ax.legend()
    return fig


def plot_model(model, assay, locations, *, condition):
    """Show the source pulses and their signed scalp weights in the comparison reference."""
    import matplotlib.pyplot as plt

    xyz = np.asarray(locations, dtype=float)
    terms = [t for t in model.terms if condition in t.amplitudes_uv]
    if not terms or xyz.shape != (len(assay.names), 3):
        raise ValueError(
            "Supply XYZ positions in capture order and a condition present in the model."
        )
    maps = [assay._in_reference(np.asarray(t.weights)) * t.amplitudes_uv[condition] for t in terms]
    picks = [p for p in assay.eeg if np.isfinite(xyz[p]).all()]
    if not picks:
        raise ValueError("No EEG coordinates available for plotting.")
    fig, axes = plt.subplots(
        2, len(terms), squeeze=False, figsize=(max(4, 2.7 * len(terms)), 5), layout="constrained"
    )
    limit = max(np.max(np.abs(m[picks])) for m in maps) or 1.0
    for i, (term, weights) in enumerate(zip(terms, maps)):
        ax = axes[0, i]
        ax.plot(
            (np.arange(len(term.pulse)) + term.offset_samples) / assay.sf,
            term.pulse * term.amplitudes_uv[condition],
        )
        ax.set(xlabel="Time (s)", ylabel="Pulse (µV)")
        points = axes[1, i].scatter(
            xyz[picks, 0], xyz[picks, 1], c=weights[picks], cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axes[1, i].set_aspect("equal")
        axes[1, i].axis("off")
    fig.colorbar(points, ax=axes[1, :].tolist(), label="Signed model weight (µV)")
    fig.suptitle(f"{model.name} · {condition}")
    return fig
