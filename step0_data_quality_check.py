"""Check ELBO input data and inspect it before line-noise removal.

Filter/reference checks validate MNE metadata, not processing before import.
EEGLAB import can supply default filter metadata and synthesized fiducials:
https://github.com/mne-tools/mne-python/blob/maint/1.13/mne/io/eeglab/eeglab.py
"""

from collections import Counter
from pathlib import Path
import warnings

from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mne
from mne._fiff.constants import FIFF
import numpy as np
import pandas as pd
from somata.multitaper.multitaper_spectrogram_python import multitaper_spectrogram


def load_raw_data(subject_id):
    """Load a subject's preloaded ColorPrecMemStudy run-1 EEG and crop to the task.

    Require 1000 Hz sampling and record the export's 100 Hz amplifier low-pass
    in MNE metadata. Crop from 3 seconds before the first enci annotation to
    2.5 seconds after the last encm annotation, preserving the native reference.
    """
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
    return raw


def check_raw_quality(raw):
    """Validate raw without modifying it; return reference and fitted sphere.

    Collect structural failures in one ValueError. Check every sample, including
    auxiliary/bad channels and annotated spans. Permit finite voltage outliers
    and flat regions, but require exactly one entirely zero EEG channel as the
    reference candidate. This cannot distinguish a reference from a dead sensor.

    Invalid spans use zero-based, half-open sample bounds relative to this Raw.
    All BAD-prefixed annotations are rejected, regardless of onset or duration.
    Geometry warnings are inspection heuristics, not anatomical validation.
    """
    errors = []
    sfreq = float(raw.info["sfreq"])
    if not np.isfinite(sfreq) or sfreq < 100:
        errors.append(f"Sampling rate must be at least 100 Hz; got {sfreq:g} Hz.")
    if not np.isclose(raw.info["highpass"], 0, rtol=0, atol=1e-8):
        errors.append(f"Expected highpass=0 Hz; got {raw.info['highpass']} Hz.")
    if not np.isclose(raw.info["lowpass"], 100, rtol=1e-6):
        errors.append("Expected lowpass=100 Hz (applied by the amplifier export; set "
                      f"raw.info['lowpass'] after loading); got {raw.info['lowpass']} Hz.")
    if raw.info["projs"]:
        errors.append("Projectors must be empty, including inactive projectors.")
    if raw.info["custom_ref_applied"] != FIFF.FIFFV_MNE_CUSTOM_REF_OFF:
        errors.append("MNE metadata indicates that a custom reference was applied.")
    if raw.n_times < 2 * sfreq:
        errors.append("At least two seconds of data are needed for the spectrograms.")

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if not len(eeg_picks):
        errors.append("No EEG channels are present; check channel types.")
    elif any(raw.info["chs"][i]["unit"] != FIFF.FIFF_UNIT_V for i in eeg_picks):
        errors.append("EEG channel units must be volts in MNE.")

    # Read one channel at a time, avoiding a second full-recording data copy.
    reference_candidates = []
    invalid_channels = []
    invalid_times = np.zeros(raw.n_times, dtype=bool)
    for i, name in enumerate(raw.ch_names):
        data = raw.get_data(picks=[i], reject_by_annotation=None)[0]
        invalid = ~np.isfinite(data)
        if invalid.any():
            invalid_channels.append(f"{name} ({invalid.sum()} samples)")
            invalid_times |= invalid
        if i in eeg_picks and data.size and np.all(data == 0):
            reference_candidates.append(name)
    if invalid_channels:
        edges = np.flatnonzero(np.diff(np.r_[False, invalid_times, False]))
        spans = edges.reshape(-1, 2)
        examples = ", ".join(
            f"[{start}, {stop}) samples / [{start / sfreq:.6f}, "
            f"{stop / sfreq:.6f}) s" for start, stop in spans[:10]
        )
        errors.append(
            f"Nonfinite values in {', '.join(invalid_channels)}. "
            f"Invalid spans across channels ({len(spans)} total; first 10 shown): "
            f"{examples}. Split into finite continuous chunks before step 1."
        )
    if len(reference_candidates) != 1:
        errors.append("Expected exactly one entirely zero EEG reference channel; "
                      f"found {reference_candidates}.")

    # Reject all BAD* annotations. Other zero-duration endpoint boundaries are OK.
    gap_names = {"boundary", "edge boundary", "edge"}
    flagged_annotations = []
    last_time = (raw.n_times - 1) / sfreq
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        start = onset - raw.first_time
        label = description.lower()
        if label.startswith("bad") or (label in gap_names and (
            0 < start < last_time or (duration > 0 and start <= last_time
                                      and start + duration > 0)
        )):
            flagged_annotations.append(f"{description} at {start:.6f} s ({duration:g} s)")
    if flagged_annotations:
        errors.append("Disallowed BAD annotations or recording discontinuities: "
                      + "; ".join(flagged_annotations)
                      + ". Resolve flagged annotations or split into valid continuous "
                      "chunks before step 1.")

    # MNE head coordinates: +x right, +y anterior, +z superior; units are meters.
    # https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
    geometry_errors = []
    xyz = np.array([raw.info["chs"][i]["loc"][:3] for i in eeg_picks])
    dig = raw.info["dig"]
    if not dig or not any(d["kind"] == FIFF.FIFFV_POINT_EEG for d in dig):
        geometry_errors.append("EEG sensor digitization is missing.")
    else:
        if any(d["coord_frame"] != FIFF.FIFFV_COORD_HEAD for d in dig):
            geometry_errors.append("Digitization must be in MNE head coordinates.")
        if not all(np.isfinite(d["r"]).all() for d in dig):
            geometry_errors.append("Digitization contains nonfinite coordinates.")
        fids = {d["ident"]: d["r"] for d in dig
                if d["kind"] == FIFF.FIFFV_POINT_CARDINAL}
        fid_ids = (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION,
                   FIFF.FIFFV_POINT_RPA)
        if not all(i in fids for i in fid_ids):
            geometry_errors.append("LPA, nasion, and RPA landmarks are required.")
        else:
            lpa, nasion, rpa = (np.asarray(fids[i]) for i in fid_ids)
            if not (lpa[0] < 0 < rpa[0] and nasion[1] > 0
                    and np.linalg.norm(np.cross(nasion - lpa, rpa - lpa)) > 0):
                geometry_errors.append("Head landmarks have invalid orientation/geometry.")
    if len(eeg_picks):
        missing = ~np.isfinite(xyz).all(axis=1) | (np.linalg.norm(xyz, axis=1) == 0)
        if missing.any():
            names = [raw.ch_names[i] for i in eeg_picks[missing]]
            geometry_errors.append(f"Missing/invalid EEG channel locations: {names}.")
        if len(np.unique(xyz, axis=0)) != len(xyz):
            geometry_errors.append("EEG channels have duplicate locations.")
        if any(raw.info["chs"][i]["coord_frame"] != FIFF.FIFFV_COORD_HEAD
               for i in eeg_picks):
            geometry_errors.append("EEG channel locations must be in head coordinates.")
    errors.extend(geometry_errors)

    sphere = None
    if len(eeg_picks) and not geometry_errors:
        # MNE warns about implausible radius/center; actual heads are not spheres.
        # https://mne.tools/stable/generated/mne.bem.fit_sphere_to_headshape.html
        try:
            radius, center, _ = mne.bem.fit_sphere_to_headshape(
                raw.info, dig_kinds="eeg", units="m", verbose=False
            )
            sphere = np.r_[center, radius]
            residual = np.sqrt(np.mean((np.linalg.norm(xyz - center, axis=1)
                                       - radius) ** 2)) / radius
            print(f"EEG sphere: radius {radius * 1000:.1f} mm; "
                  f"relative radial RMS residual {residual:.1%}.")
            if residual > 0.15:
                warnings.warn("EEG sphere residual exceeds 15%; inspect sensor geometry. "
                              "This is a heuristic, not an anatomical rejection rule.",
                              stacklevel=2)
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"Could not fit an EEG sphere: {exc}", stacklevel=2)

    print(f"Raw: {len(raw.ch_names)} channels {dict(Counter(raw.get_channel_types()))}; "
          f"{raw.n_times / sfreq:.3f} s at {sfreq:g} Hz.")
    print(f"Annotations: {dict(Counter(raw.annotations.description))}")
    if raw.info["bads"]:
        warnings.warn(f"Existing bad-channel labels: {raw.info['bads']}. "
                      "These channels remain included in step 0.", stacklevel=2)
    if errors:
        raise ValueError("Raw quality check failed:\n- " + "\n- ".join(errors))
    print(f"Raw quality checks passed. Zero reference candidate: {reference_candidates[0]}.")
    return {"reference_channel": reference_candidates[0], "sphere": sphere}


def select_spectrogram_channels(raw, reference_channel, sphere):
    """Return a 4 x 2 array of names: frontal-to-occipital rows, left/right columns."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[reference_channel])
    xyz = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    center = np.zeros(3) if sphere is None else sphere[:3]
    radius = np.median(np.linalg.norm(xyz, axis=1)) if sphere is None else sphere[3]
    directions = xyz - center
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    selected = []
    for side in (-1, 1):
        available = np.flatnonzero(
            (side * xyz[:, 0] > 0.1 * radius) & (directions[:, 2] > 0)
        ).tolist()
        if len(available) < 4:
            raise ValueError("The eight-channel view needs four scalp EEG sensors per "
                             "hemisphere, above the sphere equator and off the midline.")
        side_picks = []
        # Moderate lateral targets avoid selecting face/ear electrodes at extremes.
        for y in (0.8, 0.35, -0.35, -0.8):
            target = [side * 0.5, y, np.sqrt(1 - 0.5 ** 2 - y ** 2)]
            pick = available[int(np.argmax(directions[available] @ target))]
            available.remove(pick)
            side_picks.append(pick)
        side_picks.sort(key=lambda pick: -xyz[pick, 1])
        selected.append([raw.ch_names[picks[i]] for i in side_picks])
    return np.array(selected).T


def plot_sensor_locations(raw, sphere, selected=()):
    """Plot EEG sensor names and 1-based Raw channel numbers; optionally highlight."""
    info = raw.info.copy()
    labels = {name: f"{name}\n({i + 1})" for i, name in enumerate(raw.ch_names)}
    info.rename_channels(labels)
    info["bads"] = []  # Red denotes selected sensors in this plotting-only copy.
    eeg_picks = mne.pick_types(info, eeg=True, exclude=[])
    chosen = [raw.ch_names.index(name) for name in selected]
    other = [i for i in eeg_picks if i not in chosen]
    fig, ax = plt.subplots(figsize=(11, 11), layout="constrained")
    mne.viz.plot_sensors(
        info, ch_type="eeg", sphere=sphere, axes=ax, show=False,
        show_names=[labels[name] for name in selected] if chosen else True,
        ch_groups=[other, chosen], cmap=ListedColormap(["0.65", "red"]),
        pointsize=35 if chosen else 18, linewidth=0.4,
    )
    fig.suptitle(("Spectrogram channels (red)" if chosen else "EEG sensor locations")
                 + " — labels: name (1-based Raw index)")
    for label in ax.texts:
        label.set_fontsize(8 if not chosen else 11)
    return fig


if __name__ == "__main__":
    # %% Load example data
    subject_id = "sub-03"
    raw = load_raw_data(subject_id)

    # %% Inspect all annotations/events and counts before the quality check
    # regexp=None retains BAD*/EDGE* annotations that MNE normally excludes.
    # MNE event IDs are assigned from descriptions, not acquired trigger values.
    events, event_id = mne.events_from_annotations(raw, event_id=None, regexp=None)
    annotations_table = pd.DataFrame({
        "description": raw.annotations.description,
        "onset_s": raw.annotations.onset - raw.first_time,
        "duration_s": raw.annotations.duration,
        "sample_in_raw": events[:, 0] - raw.first_samp,
        "mne_event_id": events[:, 2],
    })
    event_counts = (
        annotations_table.groupby(["description", "mne_event_id"])
        .size().reset_index(name="count")
    )
    with pd.option_context("display.max_rows", None):
        print("All annotations/events (onsets and zero-based samples relative to cropped raw):")
        display(annotations_table)
        print("Counts by annotation description and assigned MNE event ID:")
        display(event_counts)

    # %% Check the original data before filtering or plotting
    quality = check_raw_quality(raw)
    reference_channel = quality["reference_channel"]
    sphere = quality["sphere"]
    selected_channels = select_spectrogram_channels(raw, reference_channel, sphere)
    print("Spectrogram channels (left, right; frontal to occipital):\n", selected_channels)
    plot_sensor_locations(raw, sphere)

    # %% PSD of a temporary 1-Hz high-pass copy, through Nyquist
    raw_prep = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    sfreq = float(raw_prep.info["sfreq"])
    eeg_picks = mne.pick_types(raw_prep.info, eeg=True, exclude=[reference_channel])
    psd_before = raw_prep.compute_psd(
        fmax=sfreq / 2, picks=eeg_picks, exclude=[reference_channel],
        reject_by_annotation=False,
    )
    psd_fig = psd_before.plot(
        picks="data", exclude=[reference_channel], amplitude=False, show=False
    )
    psd_fig.suptitle("Before line-noise removal (1 Hz high-pass copy)")

    # %% Eight-channel spectrograms to check data quality
    plot_sensor_locations(raw, sphere, selected_channels.ravel())
    fmax = min(150.0, sfreq / 2)
    spectrogram_params = dict(
        frequency_range=[1, fmax],
        time_bandwidth=2,
        num_tapers=3,
        window_params=[2, 1],
        min_nfft=int(2 ** np.ceil(np.log2(sfreq / 0.05))),
        detrend_opt="constant",
        plot_on=False,
        verbose=False,
    )
    power_db = []
    for channel in selected_channels.ravel():
        power, times, freqs = multitaper_spectrogram(
            raw_prep.get_data(picks=[channel], units="uV")[0],
            sfreq, **spectrogram_params,
        )
        power_db.append(10 * np.log10(np.where(power > 0, power, np.nan)))
    power_db = np.array(power_db)
    if np.isfinite(power_db).any():
        vmin, vmax = np.nanpercentile(power_db, [5, 98])
    else:
        vmin, vmax = 0, 1
        warnings.warn("Selected channels have no positive spectral power; panels are blank.")
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True, sharey=True,
                             layout="constrained")
    for i, (ax, channel, spectrum) in enumerate(
        zip(axes.flat, selected_channels.ravel(), power_db)
    ):
        mesh = ax.pcolormesh(times, freqs, spectrum, shading="auto",
                             vmin=vmin, vmax=vmax, cmap="cet_rainbow4", rasterized=True)
        ax.set_title(f"{'Left' if i % 2 == 0 else 'Right'}: {channel}")
        ax.set_ylabel("Frequency (Hz)")
    for ax in axes[-1]:
        ax.set_xlabel("Time from start of cropped recording (s)")
    fig.suptitle("Before line-noise removal (1 Hz high-pass copy) — frontal to occipital")
    fig.colorbar(mesh, ax=axes, label=r"PSD (dB re 1 $\mu V^2$/Hz)")
    plt.show()
