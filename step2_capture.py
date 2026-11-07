"""Capture and replay the linear operations actually applied during ELBO Step 2."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import mne
from mne.annotations import _annotations_starts_stops
from mne.filter import _filt_check_picks, _overlap_add_filter
import mne_denoise
import numpy as np


def filter_step2_inplace(raw, *, capture=None):
    """Apply ELBO's existing 0.3-Hz FIR, optionally recording its replay settings."""
    if capture is not None:
        capture.check_raw(raw)
        _, picks = _filt_check_picks(raw.info, None, 0.3, None)
        starts, stops = _annotations_starts_stops(raw, ("edge", "bad_acq_skip"), invert=True)
        taps = mne.filter.create_filter(None, raw.info["sfreq"], 0.3, None, verbose=False)
    raw.filter(l_freq=0.3, h_freq=None, phase="zero", pad="reflect_limited", verbose=False)
    if capture is not None:
        capture.operations.append(dict(kind="filter", picks=picks, taps=taps, phase="zero",
                                       pad="reflect_limited", bounds=np.column_stack((starts, stops))))
    return raw


def repair_piece(data, start, stop, targets, donors, weights, common, other_donors,
                 other_weights, fade):
    """Apply one spherical-spline projection and its donor/target crossfades."""
    for first in range(start, stop, 8192):
        last = min(first + 8192, stop)
        projected = weights @ data[donors, first:last]
        if fade is not None:
            previous = data[targets, first:last]
            if len(common):
                previous[common] = other_weights @ data[other_donors, first:last]
            part = fade[first - start:last - start]
            projected = part * projected + (1 - part) * previous
        data[targets, first:last] = projected


def asr_transitions(data, sfreq, state, diagnostics, process):
    """Recover exact original-grid transitions for mne-denoise's standard ASR.

    Its public diagnostics give current matrices only for *visible* windows.
    If the first visible crossfade needs an unpublished predecessor, expose
    the full update grid with the same calibrated state and statistics input.
    """
    n_times = data.shape[1]
    mask = diagnostics["sample_mask"]
    if not mask.any():
        return []
    ahead = diagnostics["lookahead_samples"]
    step = diagnostics["stepsize_samples"]
    total = n_times + ahead
    ends = np.unique(np.r_[1, np.arange(step, total, step), total])
    begins = np.r_[0, ends[:-1]]
    visible = ends > ahead
    start, stop = np.maximum(begins[visible] - ahead, 0), ends[visible] - ahead
    if not (np.array_equal(start, diagnostics["window_starts"])
            and np.array_equal(stop, diagnostics["window_stops"])):
        raise ValueError("ASR diagnostic windows do not match the reconstruction grid.")

    matrices = diagnostics["reconstruction_matrices"]
    previous = np.eye(data.shape[0])
    if begins[visible][0] and mask[0]:
        # With zero lookahead, mne-denoise exposes every matrix. Supply the
        # reflected tail explicitly so the causal statistics, number of updates
        # and covariance memory mode are identical to the production call.
        tail = 2 * data[:, [-1]] - data[:, -2:-ahead-2:-1]
        _, full = process(np.concatenate((data, tail), axis=1), sfreq, state,
                          lookahead=0., store_reconstruction_matrices=True)
        if not (np.array_equal(full["window_stops"], ends) and
                np.array_equal(full["reconstruction_matrices"][visible], matrices)):
            raise ValueError("Could not verify the ASR matrix preceding the first visible crossfade.")
        previous = full["reconstruction_matrices"][np.flatnonzero(visible)[0] - 1]

    transitions = []
    for lo, hi, left, right, current in zip(start, stop, begins[visible], ends[visible],
                                           matrices, strict=True):
        if np.any(mask[lo:hi] != mask[lo]):
            raise ValueError("ASR correction support changed inside a diagnostic window.")
        if mask[lo]:
            width = right - left
            ramp = (1 - np.cos(np.pi * np.arange(1, width + 1) / width)) / 2
            offset = lo + ahead - left
            transitions.append(dict(kind="asr", start=int(lo), stop=int(hi),
                                    previous=previous.copy(), current=current.copy(),
                                    blend=ramp[offset:offset + hi-lo].copy()))
        previous = current
    return transitions


class Step2Capture:
    """Operators on a fixed channel order and sample grid; contains no EEG traces."""

    def __init__(self, raw, reference_channel):
        self.channels = list(raw.ch_names)
        self.sfreq = float(raw.info["sfreq"])
        self.first_samp = int(raw.first_samp)
        self.n_times = raw.n_times
        self.input_reference = reference_channel
        self.output_reference = reference_channel
        self.eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[]).tolist()
        if reference_channel is not None and reference_channel not in [self.channels[i] for i in self.eeg_picks]:
            raise ValueError("Input reference must be an EEG channel in this recording.")
        self.bad_channels = list(raw.info["bads"])
        self.asr_picks = []
        self.asr_status = "not_run"
        self.asr_reason = None
        self.repairs_observed = False
        self.reference_check_observed = False
        self.operations = []
        self.versions = {"mne": mne.__version__, "mne_denoise": mne_denoise.__version__}

    @property
    def stage_availability(self):
        """Which comparisons have evidence, including confirmed no-op repairs."""
        filtered = any(op["kind"] == "filter" for op in self.operations)
        local = self.repairs_observed and self.reference_check_observed
        return {"filter_only": filtered, "repair_increment": local,
                "asr_increment": self.asr_status == "applied",
                "whole_step2": filtered and local and self.asr_status in ("applied", "skipped")}

    def check_raw(self, raw):
        if (raw.ch_names != self.channels or raw.n_times != self.n_times
                or raw.first_samp != self.first_samp or raw.info["sfreq"] != self.sfreq):
            raise ValueError("Capture and Raw must share channel order, sample bounds and rate.")

    def skip_asr(self, reason):
        """Record an explicit omission; earlier operations remain available for replay."""
        if self.asr_status not in ("not_run", "skipped") or not reason:
            raise ValueError("An ASR omission requires a reason and no previous application.")
        self.asr_status, self.asr_reason = "skipped", str(reason)

    def record_reference(self, name):
        self.operations.append(dict(kind="reference", pick=self.channels.index(name)))
        self.output_reference = name

    def record_asr(self, data, corrected, picks, state, diagnostics, process):
        transitions = asr_transitions(data, self.sfreq, state, diagnostics, process)
        # Verify that the stored transitions reproduce the actual unmodified ASR
        # output, including any region marked uncorrected.
        pos = 0
        for item in transitions:
            start, stop = item["start"], item["stop"]
            if not np.array_equal(data[:, pos:start], corrected[:, pos:start]):
                raise ValueError("ASR changed samples outside its recorded transitions.")
            replayed = self._asr_segment(data[:, start:stop], item)
            if not np.allclose(replayed, corrected[:, start:stop], atol=1e-12, rtol=1e-10):
                raise ValueError(f"ASR replay disagrees with the applied correction at samples {start}:{stop}.")
            pos = stop
        if not np.array_equal(data[:, pos:], corrected[:, pos:]):
            raise ValueError("ASR changed samples beyond its last recorded transition.")
        self.asr_picks = np.asarray(picks, dtype=int).tolist()
        self.operations.extend(transitions)
        self.asr_status, self.asr_reason = "applied", None

    @staticmethod
    def _asr_segment(segment, op):
        blend = op["blend"]
        return (op["current"] @ segment) * blend + (op["previous"] @ segment) * (1 - blend)

    def replay(self, data, *, stage="post_asr"):
        """Replay on volts in the saved channel order, with the original sample zero.

        Replay stored operations up to the requested stage. For complete-chain
        comparisons use stage_availability and replay_checkpoints: an archive
        can start partway through cleaning. asr_only takes an input in the
        pre-ASR reference.
        """
        if stage not in ("filtered", "pre_asr", "post_asr", "asr_only"):
            raise ValueError(f"Unknown replay stage: {stage}")
        if stage in ("post_asr", "asr_only") and self.asr_status == "unavailable":
            raise ValueError(f"ASR capture unavailable: {self.asr_reason}")
        if stage == "asr_only" and self.asr_status != "applied":
            raise ValueError(f"ASR {self.asr_status}: {self.asr_reason or 'no captured operation'}")
        if stage == "post_asr" and self.asr_status == "not_run":
            raise ValueError("ASR status has not been recorded.")
        result = np.array(data, dtype=float, copy=True)
        if result.shape != (len(self.channels), self.n_times):
            raise ValueError("Expected a channels-by-samples array on the recorded grid.")
        for op in self.operations:
            kind = op["kind"]
            if kind == "asr":
                if stage in ("asr_only", "post_asr"):
                    span = slice(op["start"], op["stop"])
                    result[self.asr_picks, span] = self._asr_segment(result[self.asr_picks, span], op)
            elif stage == "asr_only":
                continue
            elif kind == "filter":
                for start, stop in op["bounds"]:
                    if start < stop:
                        block = result[:, start:stop]
                        _overlap_add_filter(block, op["taps"], phase=op["phase"], picks=op["picks"],
                                            copy=False, pad=op["pad"])
            elif stage == "filtered":
                continue
            else:
                self._apply_local_operation(result, op)
        return result

    def _apply_local_operation(self, result, op):
        if op["kind"] == "repair":
            repair_piece(result, **{k: v for k, v in op.items() if k != "kind"})
        elif op["kind"] == "reference":
            result[self.eeg_picks] -= result[op["pick"]].copy()
        else:
            raise ValueError(f"Unknown captured operation: {op['kind']}")

    def replay_local(self, signal):
        """Apply observed local repairs and reference changes to a signal at their input."""
        if not self.stage_availability["repair_increment"]:
            raise ValueError("Local repairs and the reference check were not both captured.")
        result = np.array(signal, dtype=float, copy=True)
        if result.shape != (len(self.channels), self.n_times):
            raise ValueError("Expected a channels-by-samples array on the recorded grid.")
        for op in self.operations:
            if op["kind"] in ("repair", "reference"):
                self._apply_local_operation(result, op)
        return result

    def replay_checkpoints(self, data):
        """Return only endpoints for which the chain from the Step 2 input was observed."""
        phases = {"filter": 0, "repair": 1, "reference": 1, "asr": 2}
        order = [phases[op["kind"]] for op in self.operations]
        if order != sorted(order):
            raise ValueError("Checkpoint replay requires filtering, then local operations, then ASR.")
        observed = self.stage_availability
        if not observed["filter_only"]:
            return {}
        filtered = self.replay(data, stage="filtered")
        result = {"filtered": filtered}
        if observed["repair_increment"]:
            pre = self.replay_local(filtered)
            result["pre_asr"] = pre
            if self.asr_status == "applied":
                result["post_asr"] = self.replay(pre, stage="asr_only")
            elif self.asr_status == "skipped":
                result["post_asr"] = pre
        return result

    def save(self, filename):
        """Write a single versioned NPZ atomically; arrays never require pickle."""
        arrays = {}

        def store(value):
            if isinstance(value, np.ndarray):
                key = f"a{len(arrays)}"
                arrays[key] = value
                return {"array": key}
            if isinstance(value, np.integer):
                return int(value)
            raise TypeError(type(value).__name__)

        manifest = json.dumps({"version": 2, **vars(self)}, default=store)
        path = Path(filename)
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("wb") as stream:
                np.savez_compressed(stream, manifest=manifest, **arrays)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, filename):
        with np.load(filename, allow_pickle=False) as archive:
            def restore(item):
                return archive[item["array"]] if item.keys() == {"array"} else item
            manifest = json.loads(archive["manifest"].item(), object_hook=restore)
        version = manifest.pop("version")
        if version not in (1, 2):
            raise ValueError("Unsupported Step 2 capture format.")
        if version == 1:
            # Old archives contain applied operations but do not certify that both local
            # procedures completed. Individual stored operators can still be replayed.
            manifest.update(repairs_observed=False, reference_check_observed=False)
        capture = cls.__new__(cls)
        vars(capture).update(manifest)
        return capture
