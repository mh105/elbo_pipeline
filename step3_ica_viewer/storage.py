"""Recording-specific ICA snapshots, immutable identities, and review decisions."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import mne
import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise",
           "channel noise", "other")


class ReviewConflict(ValueError):
    """The caller attempted to replace a newer saved revision."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot fingerprint {type(value).__name__}.")


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _fingerprint(fields, arrays=()):
    digest = hashlib.sha256(_canonical(fields))
    for name, values in arrays:
        values = np.asarray(values)
        dtype = values.dtype.newbyteorder("<")
        values = np.ascontiguousarray(values.astype(dtype, copy=False))
        digest.update(_canonical({"name": name, "shape": values.shape, "dtype": dtype.str}))
        digest.update(memoryview(values).cast("B"))
    return "sha256:" + digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(ica, reference_channel):
    """Hash transform semantics, excluding mutable ICA exclusions and paths."""
    fields = dict(version=1, reference_channel=reference_channel,
                  n_components=int(ica.n_components_), ch_names=list(ica.ch_names),
                  channel_types=ica.info.get_channel_types(),
                  custom_ref_applied=int(ica.info["custom_ref_applied"]),
                  projs=[dict(active=bool(p["active"]), kind=int(p["kind"]),
                              data=_jsonable(p["data"])) for p in ica.info["projs"]],
                  n_pca_components=ica.n_pca_components,
                  noise_cov=None)
    if ica.noise_cov is not None:
        raise ValueError("The viewer supports standard ICA whitening (noise_cov=None).")
    arrays = [(name, getattr(ica, name)) for name in
              ("pre_whitener_", "pca_components_", "pca_mean_", "mixing_matrix_",
               "unmixing_matrix_")]
    return _fingerprint(fields, arrays)


@contextmanager
def _locked(path):
    with Path(path).open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _indices(exclude, n_components):
    if not isinstance(exclude, (list, tuple)):
        raise ValueError("exclude must be a list of zero-based component integers.")
    if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
           or not 0 <= index < n_components for index in exclude):
        raise ValueError("exclude contains a noninteger or out-of-range component.")
    values = [int(index) for index in exclude]
    if len(values) != len(set(values)):
        raise ValueError("exclude contains duplicate components.")
    return sorted(values)


def _manifest(viewer_dir):
    manifest = _read_json(Path(viewer_dir) / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("The ICA viewer manifest must be a JSON object.")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported ICA viewer manifest schema.")
    if type(manifest.get("n_components")) is not int or manifest["n_components"] < 1:
        raise ValueError("Invalid ICA component count in manifest.")
    expected = _fingerprint({key: manifest[key] for key in
                             ("recording_identity", "model_identity", "training_identity")})
    if manifest.get("fit_id") != expected:
        raise ValueError("The viewer manifest fit identity is inconsistent.")
    decision = manifest["files"]["decision"]
    if Path(decision).name != decision:
        raise ValueError("Invalid decision filename in manifest.")
    return manifest


def _validate_review(review, manifest):
    if not isinstance(review, dict):
        raise ValueError("Saved ICA exclusions must be a JSON object.")
    for field in ("schema_version", "recording_filename", "fit_id", "n_components"):
        if review.get(field) != manifest[field]:
            raise ValueError(f"Saved ICA exclusions have mismatched {field}.")
    if type(review.get("revision")) is not int or review["revision"] < 0:
        raise ValueError("Saved ICA exclusions have an invalid revision.")
    if _indices(review.get("initial_auto_exclude"), manifest["n_components"]) != manifest["initial_auto_exclude"]:
        raise ValueError("Saved ICA exclusions have a mismatched automatic baseline.")
    return _indices(review.get("exclude"), manifest["n_components"])


def read_review(viewer_dir):
    """Read a validated authoritative decision record; never reset corrupt data."""
    manifest = _manifest(viewer_dir)
    review = _read_json(Path(viewer_dir) / manifest["files"]["decision"])
    _validate_review(review, manifest)
    return review


def _initialize_review(viewer_dir, manifest):
    """Initialize only an absent file; never rewrite an existing saved record."""
    with _locked(Path(viewer_dir) / ".review.lock"):
        path = Path(viewer_dir) / manifest["files"]["decision"]
        if not path.exists():
            review = {key: manifest[key] for key in
                      ("schema_version", "recording_filename", "fit_id", "n_components", "initial_auto_exclude")}
            review.update(revision=0, updated_at_utc=_now(), exclude=manifest["initial_exclude"])
            _atomic_json(path, review)
        read_review(viewer_dir)


def save_review(viewer_dir, exclude, expected_revision):
    """Atomically replace the complete exclusion list using optimistic revisioning."""
    viewer_dir = Path(viewer_dir)
    with _locked(viewer_dir / ".review.lock"):
        manifest = _manifest(viewer_dir)
        review = read_review(viewer_dir)
        if type(expected_revision) is not int or expected_revision != review["revision"]:
            raise ReviewConflict("Saved decisions changed; reload before saving again.")
        review["exclude"] = _indices(exclude, manifest["n_components"])
        review["revision"] += 1
        review["updated_at_utc"] = _now()
        _atomic_json(viewer_dir / manifest["files"]["decision"], review)
        return review


def load_exclusions(viewer_dir, ica=None):
    """Return the full final list, optionally checking a target ICA transform."""
    manifest = _manifest(viewer_dir)
    review = read_review(viewer_dir)
    if ica is not None:
        if (ica.n_components_ != manifest["n_components"] or
                model_fingerprint(ica, manifest["reference_channel"]) != manifest["model_identity"]):
            raise ValueError("The target ICA decomposition does not match this saved review.")
    return _indices(review["exclude"], manifest["n_components"])


def _epoch_table(fit_epochs, raw_bads):
    sfreq = float(fit_epochs.info["sfreq"])
    original_sfreq = float(raw_bads.info["sfreq"])
    raw_sfreq = getattr(fit_epochs, "_raw_sfreq", original_sfreq)
    if raw_sfreq != original_sfreq or original_sfreq != 1000:
        raise ValueError("Original epoch event timing must refer to the supplied 1000 Hz Raw.")
    samples = len(fit_epochs.times)
    starts = fit_epochs.events[:, 0] + np.rint(fit_epochs.times[0] * original_sfreq).astype(int)
    duration = samples / sfreq
    ends = starts + int(round(duration * original_sfreq))
    if (np.any(starts < raw_bads.first_samp) or
            np.any(ends > raw_bads.first_samp + raw_bads.n_times) or
            np.any(starts[1:] < ends[:-1])):
        raise ValueError("Fitting epochs have overlapping or incompatible original Raw timing.")
    contiguous = np.r_[False, starts[1:] == ends[:-1]]
    return pd.DataFrame(dict(
        retained_epoch=np.arange(len(fit_epochs)),
        selection_index=np.asarray(fit_epochs.selection, dtype=int),
        event_code=fit_epochs.events[:, 2], original_event_sample=fit_epochs.events[:, 0],
        retained_start_sample=np.arange(len(fit_epochs)) * samples,
        retained_end_sample=(np.arange(len(fit_epochs)) + 1) * samples,
        retained_start_s=np.arange(len(fit_epochs)) * duration,
        retained_end_s=(np.arange(len(fit_epochs)) + 1) * duration,
        original_start_sample=starts, original_end_sample=ends,
        original_start_s=(starts - raw_bads.first_samp) / original_sfreq,
        original_end_s=(ends - raw_bads.first_samp) / original_sfreq,
        contiguous_with_previous=contiguous,
    ))


def _validate_inputs(ica, epochs, raw, reference_channel, labels):
    if (epochs.info["sfreq"] != 250 or len(epochs.times) != 500 or
            not np.isclose(epochs.times[0], 0) or
            not np.isclose(epochs.info["highpass"], 1) or
            not np.isclose(epochs.info["lowpass"], 100)):
        raise ValueError("The viewer requires exact 1–100 Hz, 250 Hz, 2-second fitting epochs.")
    if not len(epochs) or epochs.ch_names != list(ica.ch_names):
        raise ValueError("Fitting epochs must retain the fitted ICA channel order.")
    if set(epochs.get_channel_types()) != {"eeg"} or ica.noise_cov is not None:
        raise ValueError("The viewer supports EEG-only fits with noise_cov=None.")
    if reference_channel not in raw.ch_names or reference_channel in epochs.ch_names:
        raise ValueError("Provide the effective zero reference omitted from ICA fitting.")
    if not np.all(raw.get_data(picks=[reference_channel]) == 0):
        raise ValueError("The effective reference channel must be entirely zero.")
    if any(not np.all(np.isfinite(ch["loc"][:3])) or np.linalg.norm(ch["loc"][:3]) == 0
           for ch in epochs.info["chs"]):
        raise ValueError("All fitted electrodes need finite nonzero positions for topomaps.")
    data = epochs.get_data(copy=False)
    if not np.all(np.isfinite(data)):
        raise ValueError("Fitting epochs contain nonfinite samples.")
    if getattr(ica, "n_samples_", data.shape[0] * data.shape[2]) != data.shape[0] * data.shape[2]:
        raise ValueError("The epoch sample count differs from ICA's fitting support.")
    required = {"component", "winning_label", "winning_probability", "iclabel_auto_exclude", *CLASSES}
    if not isinstance(labels, pd.DataFrame) or not required.issubset(labels.columns):
        raise ValueError("Provide fit_ica's unrounded ICLabel component table.")
    labels = labels.copy().sort_values("component").reset_index(drop=True)
    if labels.component.tolist() != list(range(ica.n_components_)):
        raise ValueError("The ICLabel table must contain each zero-based component exactly once.")
    probabilities = labels[list(CLASSES)].to_numpy(dtype=float)
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or
            np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid ICLabel class probabilities.")
    winners = probabilities.argmax(axis=1)
    if (labels.winning_label.tolist() != [CLASSES[index] for index in winners] or
            not np.allclose(labels.winning_probability, probabilities[np.arange(len(labels)), winners],
                            rtol=1e-12, atol=1e-12)):
        raise ValueError("ICLabel winning labels/probabilities do not match the class table.")
    if not all(isinstance(value, (bool, np.bool_)) for value in labels.iclabel_auto_exclude):
        raise ValueError("ICLabel automatic exclusions must be booleans.")
    return labels, data


def _source_files(eeg_path, raw):
    paths = [eeg_path]
    if eeg_path.suffix.lower() == ".set":
        # MNE retains the actual external data filename, including renamed .fdt files.
        companions = [Path(name).resolve() for name in raw.filenames
                      if name and Path(name).suffix.lower() == ".fdt"]
        paths += companions
        companion = eeg_path.with_suffix(".fdt")
        if not companions and not any(raw.filenames) and companion.is_file():
            paths.append(companion)
    return [dict(name=path.name, sha256=_file_hash(path), size=path.stat().st_size)
            for path in dict.fromkeys(paths)]


def _verify_archived(viewer_dir, manifest):
    for entry in manifest["input_files"]:
        path = Path(viewer_dir) / entry["path"]
        if not path.is_file() or _file_hash(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed archived ICA viewer input: {entry['path']}")


def _cache_id(manifest):
    from .build import BUILD_VERSION
    folder = Path(__file__).parent
    versions = {name: _file_hash(folder / name) for name in
                ("build.py", "viewer_template.html")}
    return _fingerprint(dict(fit_id=manifest["fit_id"], build_version=BUILD_VERSION,
                             source_files=versions)).split(":", 1)[1]


def cache_directory(viewer_dir):
    """Return this code version's cache location; call ensure_cache to build it."""
    manifest = _manifest(viewer_dir)
    return Path(viewer_dir) / "cache" / _cache_id(manifest)


def ensure_cache(viewer_dir):
    """Rebuild missing derived assets from immutable archived fit inputs."""
    from .build import build_viewer
    viewer_dir = Path(viewer_dir)
    manifest = _manifest(viewer_dir)
    _initialize_review(viewer_dir, manifest)
    cache = cache_directory(viewer_dir)
    with _locked(viewer_dir / ".cache.lock"):
        if (cache / ".complete.json").is_file():
            try:
                complete = _read_json(cache / ".complete.json")
            except (ValueError, OSError):
                complete = {}
            files = complete.get("files", []) if isinstance(complete, dict) else []
            if (files and complete.get("fit_id") == manifest["fit_id"] and
                    {"index.html", "review_metadata.json", "summary.csv", "events.csv"}.issubset(files) and
                    all((cache / item).is_file() for item in files)):
                return cache
        _verify_archived(viewer_dir, manifest)
        ica = mne.preprocessing.read_ica(viewer_dir / "inputs/fitted-ica.fif", verbose=False)
        epochs = mne.read_epochs(viewer_dir / "inputs/training-epo.fif", preload=True, verbose=False)
        exact_data = manifest["files"].get("exact_data")
        if exact_data is not None:
            samples = np.load(viewer_dir / exact_data, allow_pickle=False)
            if samples.shape != epochs.get_data(copy=False).shape:
                raise ValueError("The exact fitting-sample archive has an incompatible shape.")
            epochs._data = samples
        sample_identity = manifest.get("training_samples_identity")
        if sample_identity is not None and _fingerprint(
                {}, (("samples", epochs.get_data(copy=False)),)) != sample_identity:
            raise ValueError("Archived fitting samples do not match the exact training identity.")
        # FIF channel locations use single precision; retain captured map geometry.
        for info in (ica.info, epochs.info):
            for channel, xyz in zip(info["chs"], manifest["geometry"]):
                channel["loc"][:3] = xyz
        labels = pd.read_csv(viewer_dir / "inputs/iclabel.csv", float_precision="round_trip")
        labels.attrs.update(manifest["iclabel_metadata"])
        epoch_table = pd.read_csv(viewer_dir / "inputs/epochs.csv")
        cache.parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=cache.parent))
        try:
            build_viewer(temporary, ica, epochs, labels,
                         recording_filename=manifest["recording_filename"],
                         reference_channel=manifest["reference_channel"], epoch_table=epoch_table,
                         review_id=manifest["fit_id"], initial_excluded=manifest["initial_exclude"])
            files = [str(path.relative_to(temporary)) for path in temporary.rglob("*") if path.is_file()]
            _atomic_json(temporary / ".complete.json", dict(fit_id=manifest["fit_id"], files=files))
            if cache.exists():
                shutil.rmtree(cache)
            os.replace(temporary, cache)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return cache


def prepare_bundle(ica, fit_epochs, raw_bads, *, eeg_path, reference_channel, iclabel_table):
    """Archive one exact ICA fit and build its viewer without overwriting reviews."""
    eeg_path = Path(eeg_path).expanduser().resolve(strict=True)
    labels, samples = _validate_inputs(ica, fit_epochs, raw_bads, reference_channel, iclabel_table)
    epoch_table = _epoch_table(fit_epochs, raw_bads)
    initial_exclude = _indices(ica.exclude, ica.n_components_)
    initial_auto = labels.loc[labels.iclabel_auto_exclude, "component"].astype(int).tolist()
    model_id = model_fingerprint(ica, reference_channel)
    geometry = np.array([ch["loc"] for ch in fit_epochs.info["chs"]])
    raw_metadata = dict(sfreq=float(raw_bads.info["sfreq"]), first_samp=int(raw_bads.first_samp),
                        n_times=int(raw_bads.n_times), crop_origin_s=float(raw_bads.first_time))
    source_files = _source_files(eeg_path, raw_bads)
    recording_id = _fingerprint(dict(filename=eeg_path.name, files=source_files,
                                     reference_channel=reference_channel, raw=raw_metadata))
    training_id = _fingerprint(dict(ch_names=fit_epochs.ch_names,
                                    channel_types=fit_epochs.get_channel_types(), sfreq=250,
                                    highpass=1, lowpass=100, epoch_tmin=float(fit_epochs.times[0]),
                                    mapping=epoch_table.to_dict("list")),
                               (("samples", samples), ("geometry", geometry)))
    training_samples_id = _fingerprint({}, (("samples", samples),))
    identities = dict(recording_identity=recording_id, model_identity=model_id,
                      training_identity=training_id)
    fit_id = _fingerprint(identities)
    parent = eeg_path.parent / ".ica_viewer" / eeg_path.name
    parent.mkdir(parents=True, exist_ok=True)
    viewer_dir = parent / fit_id.split(":", 1)[1]
    with _locked(parent / ".prepare.lock"):
        if viewer_dir.exists():
            manifest = _manifest(viewer_dir)
            if any(manifest.get(key) != value for key, value in identities.items()):
                raise ValueError("ICA viewer directory identity collision.")
            _initialize_review(viewer_dir, manifest)
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".preparing-", dir=parent))
            try:
                inputs = temporary / "inputs"
                inputs.mkdir()
                ica.save(inputs / "fitted-ica.fif", overwrite=False, verbose=False)
                # MNE temporarily rescales its preloaded buffer while writing;
                # calibration round-off must not affect the caller's fit data.
                epoch_paths = fit_epochs.copy().save(inputs / "training-epo.fif", fmt="double",
                                                      overwrite=False, verbose=False)
                restored = mne.read_epochs(inputs / "training-epo.fif", preload=True, verbose=False)
                exact_data = None
                if not np.array_equal(restored.get_data(copy=False), samples):
                    # FIF divides/multiplies channel calibration even in double
                    # precision. Preserve the original bits when that rounds.
                    np.save(inputs / "training-data.npy", samples, allow_pickle=False)
                    exact_data = "inputs/training-data.npy"
                    restored._data = np.load(temporary / exact_data, allow_pickle=False)
                if _fingerprint({}, (("samples", restored.get_data(copy=False)),)) != training_samples_id:
                    raise ValueError("Exact fitting samples did not round-trip correctly.")
                restored_ica = mne.preprocessing.read_ica(inputs / "fitted-ica.fif", verbose=False)
                if model_fingerprint(restored_ica, reference_channel) != model_id:
                    raise ValueError("The archived ICA transform did not round-trip exactly.")
                labels.to_csv(inputs / "iclabel.csv", index=False)
                epoch_table.to_csv(inputs / "epochs.csv", index=False)
                input_files = [dict(path=str(path.relative_to(temporary)), sha256=_file_hash(path))
                               for path in sorted(inputs.iterdir()) if path.is_file()]
                versions = {name: importlib.metadata.version(name) for name in
                            ("mne", "mne-icalabel", "numpy", "pandas")}
                decision_name = eeg_path.stem + "_ica_exclude.json"
                manifest = dict(schema_version=SCHEMA_VERSION, recording_filename=eeg_path.name,
                                source_path=str(eeg_path), fit_id=fit_id, **identities,
                                n_components=int(ica.n_components_), reference_channel=reference_channel,
                                original_raw=raw_metadata, geometry=geometry[:, :3].tolist(),
                                source_files=source_files,
                                training_samples_identity=training_samples_id,
                                n_epochs=len(fit_epochs), samples_per_epoch=500, sfreq=250,
                                initial_auto_exclude=initial_auto, initial_exclude=initial_exclude,
                                iclabel_metadata=_jsonable(labels.attrs), software_versions=versions,
                                input_files=input_files, created_at_utc=_now(),
                                files=dict(decision=decision_name, epochs=[str(Path(p).relative_to(temporary))
                                                                          for p in epoch_paths]))
                if exact_data is not None:
                    manifest["files"]["exact_data"] = exact_data
                review = {key: manifest[key] for key in
                          ("schema_version", "recording_filename", "fit_id", "n_components", "initial_auto_exclude")}
                review.update(revision=0, updated_at_utc=_now(), exclude=initial_exclude)
                _atomic_json(temporary / "manifest.json", manifest)
                _atomic_json(temporary / decision_name, review)
                os.replace(temporary, viewer_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    ensure_cache(viewer_dir)
    return viewer_dir
