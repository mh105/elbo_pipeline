"""Capture the affine reconstruction from a reviewed ELBO ICA without refitting."""
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
import shutil
import warnings

import mne
import numpy as np

from step3_ica_viewer.storage import (_indices, load_review_choice, model_fingerprint,
                                      recording_sample_digest)


@dataclass
class Step3Capture:
    """Spatial matrix L and offset b for F(x) = Lx + b, in original channel order."""

    metadata: dict
    matrix: np.ndarray
    offset_v: np.ndarray

    def __post_init__(self):
        names = self.metadata["channels"]
        fit = self.metadata["fitted_channels"]
        n = len(names)
        if (len(set(names)) != n or not fit or len(set(fit)) != len(fit)
                or [name for name in names if name in fit] != fit):
            raise ValueError("Invalid channel order in the ICA capture.")
        if (self.matrix.shape != (n, n) or self.offset_v.shape != (n,)
                or not np.isfinite(self.matrix).all() or not np.isfinite(self.offset_v).all()):
            raise ValueError("Invalid ICA matrix or offset.")
        other = [i for i, name in enumerate(names) if name not in fit]
        identity = np.eye(n)
        if (not np.array_equal(self.matrix[other], identity[other])
                or not np.array_equal(self.matrix[:, other], identity[:, other])
                or np.any(self.offset_v[other])):
            raise ValueError("Non-fitted channels must pass through unchanged.")

    def replay(self, signal, *, ch_names):
        """Return F(signal) - F(0) for a channels-by-time signal in volts."""
        x = np.asarray(signal, dtype=float)
        if list(ch_names) != self.metadata["channels"] or x.ndim != 2 or x.shape[0] != len(ch_names):
            raise ValueError("Signal must use the captured channel names and order.")
        picks = [self.metadata["channels"].index(name) for name in self.metadata["fitted_channels"]]
        out = x.copy()
        # Non-fitted channels may contain NaNs; they never enter the fitted block.
        out[picks] = self.matrix[np.ix_(picks, picks)] @ x[picks]
        return out

    def verify_recording(self, before, after):
        """Check the affine map against a reconstructed recording, in small blocks."""
        meta = self.metadata
        for raw in (before, after):
            if (raw.ch_names != meta["channels"] or raw.n_times != meta["n_times"]
                    or raw.first_samp != meta["first_samp"] or raw.info["sfreq"] != meta["sfreq"]
                    or raw.info["bads"] != meta["bad_channels"]):
                raise ValueError("Recording and capture have different channels, bads or sample grids.")
        if before.annotations != after.annotations:
            raise ValueError("The before and after recordings have different annotations.")
        picks = [meta["channels"].index(name) for name in meta["fitted_channels"]]
        other = [i for i in range(len(meta["channels"])) if i not in picks]
        worst = 0.
        for start in range(0, before.n_times, 8192):
            stop = min(start + 8192, before.n_times)
            x = before.get_data(start=start, stop=stop)
            y = after.get_data(start=start, stop=stop)
            if not np.array_equal(x[other], y[other], equal_nan=True):
                raise ValueError("Reconstruction modified non-fitted channels.")
            if not np.isfinite(x[picks]).all() or not np.isfinite(y[picks]).all():
                raise ValueError("Non-finite fitted-channel data in ICA reconstruction.")
            replay = self.replay(x, ch_names=before.ch_names)[picks] + self.offset_v[picks, None]
            if not np.allclose(y[picks], replay, atol=1e-15, rtol=1e-10):
                raise ValueError("Saved ICA transformation does not reproduce the supplied reconstruction.")
            worst = max(worst, float(np.max(np.abs(y[picks] - replay))))
        return worst

    def save(self, path):
        """Serialize only provenance, matrix and offset, without pickling or EEG samples."""
        path = Path(path)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
        try:
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, metadata=json.dumps(self.metadata, allow_nan=False),
                                    matrix=self.matrix, offset_v=self.offset_v)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            if metadata.get("schema_version") != 1:
                raise ValueError("Unsupported ICA capture version.")
            return cls(metadata, archive["matrix"], archive["offset_v"])


def reconstruct_ica(raw, ica, *, reference_channel, exclude=None, viewer_dir=None,
                    version=None, n_pca_components=None, allow_other_recording=False):
    """Return (MNE-reconstructed Raw copy, Step3Capture) for one exact choice.

    With a viewer directory, use its saved labeling (the designated final version
    by default). An explicit list, including [], replaces all existing excludes.
    A viewer directory can also identify the fit of an explicit comparison list.
    """
    after, capture = _reconstruct_ica(raw, ica, reference_channel=reference_channel,
                                      exclude=exclude, viewer_dir=viewer_dir, version=version,
                                      n_pca_components=n_pca_components,
                                      allow_other_recording=allow_other_recording)
    capture.metadata["reconstruction_max_abs_error_v"] = capture.verify_recording(raw, after)
    return after, capture


def _reconstruct_ica(raw, ica, *, reference_channel, exclude=None, viewer_dir=None,
                     version=None, n_pca_components=None, allow_other_recording=False):
    if version is not None and (viewer_dir is None or exclude is not None):
        raise ValueError("A labeling version requires a viewer directory and its saved exclusions.")
    if exclude is None and viewer_dir is None:
        raise ValueError("Supply a viewer directory or an explicit exclusion list.")
    if [ch for ch in raw.ch_names if ch in ica.ch_names] != list(ica.ch_names):
        raise ValueError("Recording must contain the fitted channels in ICA fit order.")
    if reference_channel is not None:
        if (reference_channel not in raw.ch_names or reference_channel in ica.ch_names
                or np.any(raw.get_data(picks=[reference_channel]))):
            raise ValueError("The named reference must be present, zero, and outside the ICA fit.")
    transform_id = model_fingerprint(ica, reference_channel)
    fit_id = transform_id
    choice = {"source": "explicit"}
    recording_check = {"status": "no_viewer_identity"}
    if viewer_dir is not None:
        saved = load_review_choice(viewer_dir, ica=ica, version=version)
        grid = saved["original_raw"]
        if saved["reference_channel"] != reference_channel:
            raise ValueError("The recording reference differs from the saved viewer fit.")
        if grid is None:
            recording_check = {"status": "no_saved_grid"}
        else:
            matches_grid = ((grid["first_samp"], grid["n_times"], grid["sfreq"]) ==
                            (raw.first_samp, raw.n_times, raw.info["sfreq"]))
            if not matches_grid and not allow_other_recording:
                raise ValueError("The recording sample grid differs from the saved viewer fit.")
            expected = grid.get("samples_sha256")
            if expected is None:
                recording_check = {"status": "grid_only" if matches_grid else "different_grid"}
            else:
                actual = recording_sample_digest(raw)
                matches = matches_grid and actual == expected
                if not matches and not allow_other_recording:
                    raise ValueError("The processed recording differs from the saved viewer input. "
                                     "Use allow_other_recording=True for deliberate transfer.")
                recording_check = dict(status="verified_samples" if matches else "different_recording",
                                       expected_sha256=expected, input_sha256=actual)
        fit_id = saved["fit_id"]
        if exclude is None:
            exclude = saved["exclude"]
            choice = {key: saved[key] for key in ("version", "revision", "updated_at_utc")}
            choice["source"] = "viewer"
    model = ica.copy()
    model.exclude = _indices(exclude, model.n_components_)
    # Passing exclude to ICA.apply would union it with model.exclude. Assign it
    # on the copy so a previously rejected component can be kept in this branch.
    apply_settings = dict(n_pca_components=n_pca_components, verbose=False)
    after = raw.copy().load_data(verbose=False)
    model.apply(after, **apply_settings)
    n = len(raw.ch_names)
    basis_v = 1e-6
    def probe(data):
        with mne.io.RawArray(data, raw.info.copy(), verbose=False) as trial:
            model.apply(trial, **apply_settings)
            return trial.get_data()
    origin = probe(np.zeros((n, n)))
    if not np.allclose(origin, origin[:, :1], atol=1e-15, rtol=1e-10):
        raise ValueError("ICA offset unexpectedly varies across samples.")
    matrix = (probe(basis_v * np.eye(n)) - origin) / basis_v
    setting = model.n_pca_components if n_pca_components is None else n_pca_components
    # Ask the same MNE resolver used by ICA.apply (including rank and float settings).
    resolved = int(model._check_n_pca_components(setting, verbose=False))
    metadata = dict(schema_version=1, fit_id=fit_id, model_fingerprint=transform_id,
                    fit_id_kind="viewer" if viewer_dir is not None else "model_fingerprint",
                    choice=choice, recording_check=recording_check, exclude=model.exclude, n_components=int(model.n_components_),
                    channels=list(raw.ch_names), fitted_channels=list(model.ch_names),
                    bad_channels=list(raw.info["bads"]), reference_channel=reference_channel,
                    custom_ref_applied=int(raw.info["custom_ref_applied"]),
                    first_samp=int(raw.first_samp), n_times=int(raw.n_times), sfreq=float(raw.info["sfreq"]),
                    n_pca_components=setting.item() if isinstance(setting, np.generic) else setting,
                    resolved_pca_components=resolved, mne_version=mne.__version__)
    return after, Step3Capture(metadata, matrix, origin[:, 0])


def recorded_ica_change(before, after, capture, *, bad_samples="include", window_seconds=1.0):
    """Measure RMS voltage changes on fitted channels over the original time grid."""
    from mne.annotations import _annotations_starts_stops

    meta = capture.metadata
    max_error = capture.verify_recording(before, after)
    if bad_samples not in ("include", "omit"):
        raise ValueError("bad_samples must be 'include' or 'omit'.")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite.")
    width = int(round(window_seconds * meta["sfreq"]))
    if width < 1:
        raise ValueError("The RMS window must contain at least one sample.")
    selected = np.ones(before.n_times, dtype=bool)
    starts, stops = _annotations_starts_stops(before, ("BAD",))
    if bad_samples == "omit":
        for start, stop in zip(starts, stops):
            selected[start:stop] = False
    names = meta["fitted_channels"]
    sumsq = np.zeros((3, len(names)))
    windows = []
    for lo in range(0, before.n_times, width):
        hi = min(lo + width, meta["n_times"])
        mask = selected[lo:hi]
        a = before.get_data(picks=names, start=lo, stop=hi) * 1e6
        b = after.get_data(picks=names, start=lo, stop=hi) * 1e6
        if not mask.all():
            a, b = a[:, mask], b[:, mask]
        row = dict(start=lo, stop=hi, n_used=int(mask.sum()), time_s=(lo + hi)/(2 * meta["sfreq"]))
        for idx, (key, data) in enumerate(zip(("before", "after", "change"), (a, b, a - b))):
            square = data * data
            sumsq[idx] += square.sum(axis=1)
            row[key + "_rms_uv"] = float(np.sqrt(square.mean())) if data.size else None
        windows.append(row)
    count = int(selected.sum())
    rms = np.sqrt(sumsq / count) if count else np.full_like(sumsq, np.nan)
    channels = []
    for name, (initial, final, change) in zip(names, rms.T):
        channels.append(dict(name=name, before_rms_uv=float(initial) if count else None,
                             after_rms_uv=float(final) if count else None,
                             change_rms_uv=float(change) if count else None,
                             relative_change_percent=float(100 * change / initial)
                             if count and initial > 0 else None))
    return dict(fit_id=meta["fit_id"], choice=meta["choice"], recording_check=meta.get("recording_check", {"status": "not_recorded"}),
                reconstruction_max_abs_error_v=max_error, exclude=meta["exclude"],
                reference_channel=meta["reference_channel"], n_components=meta["n_components"],
                first_samp=meta["first_samp"], sfreq=meta["sfreq"], n_total=meta["n_times"], n_used=count,
                bad_samples=bad_samples, bad_intervals=np.column_stack((starts, stops)).tolist(),
                window_samples=width, windows=windows, channels=channels)


def plot_ica_change(change):
    """Plot pooled RMS over time and the cumulative distribution across fitted channels."""
    import matplotlib.pyplot as plt

    fig, (time_ax, channel_ax) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    times = [row["time_s"] for row in change["windows"]]
    for key, label in (("before", "Before ICA"), ("after", "After ICA"), ("change", "Difference")):
        time_ax.plot(times, [row[key + "_rms_uv"] for row in change["windows"]], label=label)
    time_ax.set(xlabel="Seconds from recording start", ylabel="RMS across fitted electrodes (µV)")
    time_ax.legend()
    values = sorted(row["relative_change_percent"] for row in change["channels"]
                    if row["relative_change_percent"] is not None)
    if values:
        x = [values[0], *values]
        channel_ax.step(x, 100*np.arange(len(x))/len(values), where="post")
    channel_ax.set(xlabel="RMS difference / before-ICA RMS (%)",
                   ylabel="Electrodes at or below this change (%)", ylim=(0, 100))
    fig.suptitle(f"ICA reconstruction · {len(change['exclude'])} components excluded · "
                 f"{change['n_used']} / {change['n_total']} samples")
    return fig


def save_ica_report(parent, before, after, capture, *, bad_samples="include", window_seconds=1.0):
    """Publish a new report directory. Keep completed choices when the viewer is revised."""
    import matplotlib.pyplot as plt

    change = recorded_ica_change(before, after, capture, bad_samples=bad_samples,
                                 window_seconds=window_seconds)
    capture.metadata["reconstruction_max_abs_error_v"] = change["reconstruction_max_abs_error_v"]
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix="ica-", dir=parent))
    try:
        capture.save(folder / "ica_transform.npz")
        try:
            fig = plot_ica_change(change)
            try:
                fig.savefig(folder / "eeg_change.png", dpi=150)
            finally:
                plt.close(fig)
        except Exception as exc:
            change["figure_warning"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"Could not draw EEG change figure: {exc}", RuntimeWarning, stacklevel=2)
        (folder / "eeg_change.json").write_text(
            json.dumps(change, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(folder)
        raise
    return folder


def reconstruct_ica_report(parent, before, ica, **settings):
    """Reconstruct, verify and write the EEG-change report in one pass through verification."""
    after, capture = _reconstruct_ica(before, ica, **settings)
    folder = save_ica_report(parent, before, after, capture)
    return after, capture, folder
