"""Replay synthetic event trains on the recorded grid using previously fitted operations."""

from dataclasses import dataclass, field

import numpy as np
from step2_capture import Step2Capture
from step3_capture import Step3Capture

from .metrics import Windows, measure


def _train(pulse, onsets, n, offset=0):
    data = np.zeros(n)
    for i in onsets + offset:
        start, stop = max(0, i), min(n, i + len(pulse))
        if stop > start:
            data[start:stop] += pulse[start - i : stop - i]
    return data


@dataclass
class Assessment:
    """events maps condition labels to *absolute* sample numbers (MNE convention)."""

    capture: object
    events: dict
    windows: Windows = field(default_factory=Windows)
    bad_intervals: object = None
    contrast: tuple = None
    upstream_step2: Step2Capture = None

    def __post_init__(self):
        cap = self.capture
        if isinstance(cap, Step2Capture):
            self.step = 2
            self.names, self.sf, self.n, self.first = (
                cap.channels,
                cap.sfreq,
                cap.n_times,
                cap.first_samp,
            )
            self.eeg = list(cap.eeg_picks)
            self.ref = (
                self.names.index(cap.output_reference) if cap.output_reference is not None else None
            )
            self.picks = [
                i for i in self.eeg if self.names[i] not in cap.bad_channels and i != self.ref
            ]
            self.asr_status = cap.asr_status
        elif isinstance(cap, Step3Capture):
            self.step = 3
            meta = cap.metadata
            self.names, self.sf = meta["channels"], meta["sfreq"]
            self.n, self.first = meta["n_times"], meta["first_samp"]
            self.picks = [self.names.index(n) for n in meta["fitted_channels"]]
            self.ref = (
                self.names.index(meta["reference_channel"]) if meta["reference_channel"] else None
            )
            self.eeg = self.picks + ([] if self.ref is None else [self.ref])
            self.asr_status = None
        else:
            raise TypeError("Supply a saved ELBO Step 2 or Step 3 capture.")
        self.sf, self.n, self.first = float(self.sf), int(self.n), int(self.first)
        if not self.picks:
            raise ValueError("No eligible EEG electrodes for the assay.")
        if self.upstream_step2 is not None:
            up = self.upstream_step2
            if (
                self.step != 3
                or up.channels != self.names
                or up.first_samp != self.first
                or up.sfreq != self.sf
                or up.n_times != self.n
                or up.output_reference != self.capture.metadata["reference_channel"]
            ):
                raise ValueError(
                    "Upstream Step 2 must match this ICA recording, grid and reference."
                )
            self.eeg = list(up.eeg_picks)
        self.onsets = {}
        for label, samples in self.events.items():
            v = np.asarray(samples)
            if (
                not isinstance(label, str)
                or v.ndim != 1
                or (
                    v.size
                    and (
                        v.dtype.kind not in "iu"
                        or np.any(v < self.first)
                        or np.any(v >= self.first + self.n)
                    )
                )
            ):
                raise ValueError(
                    "Events must use named conditions and absolute integer samples on the capture grid."
                )
            self.onsets[label] = v.astype(int) - self.first
        if not self.onsets:
            raise ValueError("Supply at least one event condition.")
        if self.contrast is not None and (
            len(self.contrast) != 2
            or len(set(self.contrast)) != 2
            or not set(self.contrast) <= self.onsets.keys()
        ):
            raise ValueError("contrast must be two distinct condition labels in subtraction order.")
        self.start, self.stop = self.windows.bounds(self.sf)
        self.coverage = self._coverage()

    def auditory_models(self, locations, *, conditions, model_reference=None, eeg_picks=None):
        """Build the default catalog on this grid; preserve the original map normalization."""
        from .models import auditory_models

        source = self.capture if self.step == 2 else self.upstream_step2
        if model_reference is None:
            model_reference = (
                source.input_reference
                if source is not None
                else self.capture.metadata["reference_channel"]
            )
        index = None if model_reference is None else self.names.index(model_reference)
        picks = self.eeg if eeg_picks is None else eeg_picks
        return auditory_models(
            self.sf, locations, conditions=conditions, model_reference=index, eeg_picks=picks
        )

    def _coverage(self):
        source = self.capture if self.step == 2 else self.upstream_step2

        def intervals(kind):
            return [(op["start"], op["stop"]) for op in source.operations if op["kind"] == kind]

        spans = {"bad": self.bad_intervals, "repair": None, "asr": None}
        if source is not None:
            spans["repair"] = intervals("repair")
            if source.asr_status in ("applied", "skipped"):
                spans["asr"] = intervals("asr")
        for name, bounds in spans.items():
            if bounds is not None:
                spans[name] = list(bounds)
            if spans[name] is not None and any(
                not 0 <= lo <= hi <= self.n for lo, hi in spans[name]
            ):
                raise ValueError(
                    f"Invalid {name} sample intervals (expected relative half-open bounds)."
                )
        coverage = {}
        for label, onsets in self.onsets.items():
            rows = []
            for onset in onsets:
                lo, hi = onset + self.start, onset + self.stop
                row = {
                    "sample": int(onset + self.first),
                    "complete": bool(lo >= 0 and hi <= self.n),
                }
                for name, bounds in spans.items():
                    row[name] = (
                        None
                        if bounds is None
                        else any(a < b and lo < b and a < hi for a, b in bounds)
                    )
                rows.append(row)
            coverage[label] = rows
        return coverage

    def _check_model(self, model):
        maps = [term.weights for term in model.terms] + [model.focus]
        if not model.terms or model.polarity not in (-1, 1):
            raise ValueError("Model needs terms and a focus polarity of -1 or +1.")
        for weights in maps:
            if np.shape(weights) != (len(self.names),) or not np.isfinite(weights).all():
                raise ValueError("Every scalp map must be finite and in capture channel order.")
        for term in model.terms:
            if (
                np.ndim(term.pulse) != 1
                or not len(term.pulse)
                or not np.isfinite(term.pulse).all()
                or not term.amplitudes_uv
                or not term.amplitudes_uv.keys() <= self.onsets.keys()
                or not np.isfinite(list(term.amplitudes_uv.values())).all()
            ):
                raise ValueError("Invalid pulse, amplitude or condition label.")
        for window in (
            model.peak_window,
            model.mean_window,
            self.windows.wide_mean,
            self.windows.baseline,
        ):
            self.windows.mask(self.sf, window, require_samples=False)

    def _in_reference(self, signal):
        if self.ref is None:
            return signal
        out = signal.copy()
        out[self.eeg] -= signal[self.ref].copy()
        return out

    def _input_terms(self, model):
        for term in model.terms:
            trains = [
                _train(term.pulse, self.onsets[label], self.n, term.offset_samples)
                for label in term.amplitudes_uv
            ]
            amps = list(term.amplitudes_uv.values())
            weights = term.weights.copy()
            if self.capture.input_reference is not None:
                weights[self.eeg] -= weights[self.names.index(self.capture.input_reference)]
            if len(amps) == 1:
                yield (1e-6 * amps[0]) * weights[:, None] * trains[0]
            else:
                yield 1e-6 * weights[:, None] * sum(a * t for a, t in zip(amps, trains))

    def _step2_pairs(self, model):
        probe = sum(self._input_terms(model))
        stages = self.capture.replay_checkpoints(probe)
        expected = self._in_reference(probe)
        observed = self.capture.stage_availability
        pairs = {}
        if "filtered" in stages:
            filtered = self._in_reference(stages["filtered"])
            pairs["filter_only"] = expected, filtered
        if "pre_asr" in stages:
            pairs["repair_increment"] = filtered, stages["pre_asr"]
        elif observed["repair_increment"]:
            pairs["repair_increment"] = expected, self.capture.replay_local(probe)
        if observed["asr_increment"]:
            if "post_asr" in stages:
                pairs["asr_increment"] = stages["pre_asr"], stages["post_asr"]
            else:
                pairs["asr_increment"] = expected, self.capture.replay(expected, stage="asr_only")
        if observed["whole_step2"]:
            pairs["whole_step2"] = expected, stages["post_asr"]
        return pairs

    def _step3_terms(self, model):
        maps = self._in_reference(np.column_stack([t.weights for t in model.terms]))
        after = self.capture.replay(maps, ch_names=self.names)
        temporal = np.array(
            [
                1e-6
                * sum(
                    amp * _train(t.pulse, self.onsets[label], self.n, t.offset_samples)
                    for label, amp in t.amplitudes_uv.items()
                )
                for t in model.terms
            ]
        )
        return maps[self.picks], after[self.picks], temporal

    def _epoch_blocks(self, pair, used, temporal):
        for start in range(0, len(used), 4):
            samples = used[start : start + 4]
            if temporal is None:
                yield tuple(
                    np.array([x[self.picks, n + self.start : n + self.stop] for n in samples])
                    for x in pair
                )
            else:
                epochs = np.array([temporal[:, n + self.start : n + self.stop] for n in samples])
                yield tuple(np.einsum("ck,ekt->ect", m, epochs, optimize=True) for m in pair)

    def assess(self, model):
        """Measure one model, keeping other models and biological samples out of memory."""
        self._check_model(model)
        focus = self._in_reference(np.asarray(model.focus))[self.picks]
        rep = int(np.argmax(abs(focus)))
        sign = float(model.polarity * np.sign(focus[rep]))
        records = []
        if self.step == 2:
            temporal = None
            pairs = self._step2_pairs(model)
        else:
            before_maps, after_maps, temporal = self._step3_terms(model)
            pairs = {"ica": (before_maps, after_maps)}
        for label in self.onsets:
            if not any(label in t.amplitudes_uv for t in model.terms):
                continue
            onsets = self.onsets[label]
            used = onsets[(onsets + self.start >= 0) & (onsets + self.stop <= self.n)]
            for stage, pair in pairs.items():
                record = {
                    "stage": stage,
                    "condition": label,
                    "requested": len(onsets),
                    "used": used + self.first,
                }
                if not len(used):
                    records.append(dict(record, status="no_complete_epochs"))
                    continue

                blocks = self._epoch_blocks(pair, used, temporal)
                records.extend(self._records(record, blocks, model, rep, sign))
        return {
            "name": model.name,
            "family": model.family,
            "info": model.info,
            "representative": self.names[self.picks[rep]],
            "orientation": sign,
            "focus_available": bool(focus[rep]),
            "time_s": np.arange(self.start, self.stop) / self.sf,
            "measurement_windows": {
                name: self.windows.coverage(self.sf, window)
                for name, window in (("peak", model.peak_window), ("mean", model.mean_window),
                                     ("wide_mean", self.windows.wide_mean), ("baseline", self.windows.baseline))
            },
            "records": records,
            "contrasts": self._contrasts(records, model),
        }

    def _records(self, record, blocks, model, rep, sign):
        shape = (len(self.picks), self.stop - self.start)
        sum_before, sum_after = np.zeros(shape), np.zeros(shape)
        baseline_present = self.windows.mask(self.sf, self.windows.baseline, require_samples=False).any()
        stats = {False: []}
        if baseline_present:
            stats[True] = []
        report_gain = baseline_present and model.family == "legacy" and record["stage"] == "whole_step2"
        if report_gain:
            power, projection = np.zeros(len(self.picks)), np.zeros(len(self.picks))
            idx = np.flatnonzero(self.windows.mask(self.sf, self.windows.baseline))
            baseline = slice(idx[0], idx[-1] + 1)
        args = (self.sf, rep, sign, model.peak_window, model.mean_window, self.windows)
        for before, after in blocks:
            for b, a in zip(before, after):
                sum_before += b
                sum_after += a
            for corrected, chunks in stats.items():
                chunks.append(measure(before, after, *args, corrected))
            if report_gain:
                x = before - before[:, :, baseline].mean(2, keepdims=True)
                y = after - after[:, :, baseline].mean(2, keepdims=True)
                x, y = x[:, :, -self.start :], y[:, :, -self.start :]
                power += (x * x).sum(axis=(0, 2))
                projection += (x * y).sum(axis=(0, 2))
        count = len(record["used"])
        for corrected, chunks in stats.items():
            metrics = {k: np.concatenate([chunk[0][k] for chunk in chunks]) for k in chunks[0][0]}
            bt, at = (np.concatenate([chunk[i] for chunk in chunks]) for i in (1, 2))
            avg, _, _ = measure(sum_before[None] / count, sum_after[None] / count, *args, corrected)
            item = dict(
                record,
                status="measured",
                convention="baseline_corrected" if corrected else "unbaselined",
                metrics=metrics,
                average_metrics={k: v[0] for k, v in avg.items()},
                mean_before=bt.mean(0),
                mean_after=at.mean(0),
            )
            if corrected:
                item.update(event_before=bt, event_after=at)
            if corrected and report_gain:
                item["channel_gain"] = projection / np.maximum(power, 1e-30)
                focus = self._in_reference(model.focus)[self.picks]
                item["channel_gain_reported"] = abs(focus) >= 0.2 * np.max(abs(focus))
            yield item

        if not baseline_present:
            yield dict(record, status="no_baseline_samples", convention="baseline_corrected")

    def _contrasts(self, records, model):
        if self.contrast is None:
            return []
        lookup = {
            (r["stage"], r["condition"]): r
            for r in records
            if r.get("convention") == "baseline_corrected" and r["status"] == "measured"
        }
        results = []
        for stage in dict.fromkeys(r["stage"] for r in records):
            pos, neg = (lookup.get((stage, label)) for label in self.contrast)
            if pos is None or neg is None:
                continue
            before = pos["mean_before"] - neg["mean_before"]
            after = pos["mean_after"] - neg["mean_after"]
            result = {
                "stage": stage,
                "labels": self.contrast,
                "before": before,
                "after": after,
                "windows": {},
            }
            for key, win in (("mean", model.mean_window), ("wide_mean", self.windows.wide_mean)):
                mask = self.windows.mask(self.sf, win, require_samples=False)
                if not mask.any():
                    result["windows"][key] = dict(expected_uv=None, output_uv=None, change_uv=None)
                    continue
                result["windows"][key] = {
                    "expected_uv": float(before[mask].mean() * 1e6),
                    "output_uv": float(after[mask].mean() * 1e6),
                    "change_uv": float((after - before)[mask].mean() * 1e6),
                }
            results.append(result)
        return results
