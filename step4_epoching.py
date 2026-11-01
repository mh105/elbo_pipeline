"""ELBO step 4: epoching scaffold.

Run Steps 0-3 to prepare ICA-cleaned continuous data for epoching.
Epoch definitions and rejection settings remain to be specified.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from autoreject import AutoReject

from step0_data_quality_check import load_raw_data
from step1_line_noise_removal import (
    ZapLine_notch,
    Interpolate_fft,
    remove_line_noise_inplace,
)
from step2_bad_ch_time_detection import (
    bad_ch_by_nan,
    bad_ch_by_flatline,
    detect_flatline_inplace,
    mask_segment_by_flatline,
    bad_ch_time_detection_inplace,
)
from step3_ica_cleaning import (
    fit_ica,
    prepare_ica_viewer,
    load_ica_exclusions,
    summarize_ica_exclusions,
)


def prepare_epochs(raw, behavior):
    """Filter raw in place and return baseline-corrected 100 Hz encm epochs.

    Require one behavior row per encm event and 1000 Hz input data.
    """
    skip_annotations = ("edge", "bad_acq_skip", "BAD")
    raw.filter(
        l_freq=None,
        h_freq=45,
        h_trans_bandwidth=5,
        skip_by_annotation=skip_annotations,
        verbose=False,
    )

    events_from_annot, event_dict = mne.events_from_annotations(raw)
    stim_events = mne.pick_events(events_from_annot, include=event_dict["encm"])
    if len(behavior) != len(stim_events):
        raise ValueError(
            f"Behavior rows ({len(behavior)}) do not match encm events "
            f"({len(stim_events)})."
        )
    epochs = mne.Epochs(
        raw,
        stim_events,
        tmin=-0.5,
        tmax=2.5,
        preload=True,
        baseline=(None, 0),
    )
    if epochs.info["sfreq"] != 1000:
        raise ValueError("Expected 1000 Hz sampling rate in raw data.")
    with warnings.catch_warnings():
        # The low-pass stopband starts at the new Nyquist frequency (50 Hz).
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning,
            message=r"The measurement information indicates a low-pass frequency "
                    r"of 45\.0 Hz\. The decim=10 parameter will result in a sampling "
                    r"frequency of 100\.0 Hz, which can cause aliasing artifacts\.$",
        )
        epochs.decimate(decim=10)
    if epochs.info["sfreq"] != 100:
        raise ValueError("Expected 100 Hz sampling rate after decimation.")
    return epochs


def plot_epoch_rejection(epochs, epochs_clean, candidate_channels, n_epochs=20):
    """Show static panels of pre-AutoReject traces; return their figures.

    Stack available, non-bad candidate channels and show n_epochs per panel.
    Red traces mark AutoReject rejections; black traces mark retained epochs.
    All traces precede AutoReject interpolation. Trial labels are one-based
    original event indices, preserving gaps from earlier annotation rejection.
    Automatically use +/-3 times the pooled interquartile range of the plotted
    channels in retained trials, using their original pre-AutoReject traces.
    This widens MNE's scalings="auto" range for readable static traces. Share
    the scale across panels; use 1 microvolt if the range is zero or unavailable.
    Both Epochs objects must come from the same Step 4 run.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be positive.")
    available_channels = set(epochs.ch_names) - set(epochs.info["bads"])
    channels = [c for c in candidate_channels if c in available_channels]
    if not channels:
        raise ValueError("No candidate channels are available and non-bad.")

    selection = epochs.selection
    if not len(selection):
        raise ValueError("No epochs are available to plot.")
    expected_selection = np.array([
        trial for trial, reasons in enumerate(epochs_clean.drop_log)
        if not reasons or "AUTOREJECT" in reasons
    ])
    if (len(epochs.drop_log) != len(epochs_clean.drop_log)
            or not np.array_equal(selection, expected_selection)):
        raise ValueError("Pre-AutoReject and cleaned epochs do not match.")
    rejected = np.array([
        "AUTOREJECT" in epochs_clean.drop_log[trial] for trial in selection
    ])
    if not np.array_equal(epochs.events[~rejected], epochs_clean.events):
        raise ValueError("Pre-AutoReject and cleaned epoch events do not match.")
    data = epochs.get_data(picks=channels, units="uV")
    scale_data = data[~rejected]
    finite_data = scale_data[np.isfinite(scale_data)]
    scaling_uv = (3 * np.diff(np.percentile(finite_data, [25, 75]))[0]
                  if finite_data.size else 0.0)
    if scaling_uv == 0:
        scaling_uv = 1.0
    spacing_uv = 2 * scaling_uv
    offsets = np.arange(len(channels))[::-1] * spacing_uv
    times = epochs.times
    duration = times[-1] - times[0] + 1 / epochs.info["sfreq"]
    relative_times = times - times[0]
    figures = []
    for start in range(0, len(data), n_epochs):
        stop = min(start + n_epochs, len(data))
        fig, ax = plt.subplots(figsize=(14, max(6, 0.85 * len(channels))),
                               layout="constrained")
        for slot, epoch_index in enumerate(range(start, stop)):
            x = relative_times + slot * duration
            color = "red" if rejected[epoch_index] else "black"
            ax.plot(x, (data[epoch_index] + offsets[:, None]).T,
                    color=color, linewidth=0.6)
            ax.axvline(slot * duration, color="0.4", linewidth=0.7,
                       linestyle="--")
            ax.axvline(slot * duration - times[0], color="0.6",
                       linewidth=0.5, linestyle=":")
        ax.set(
            xticks=(np.arange(stop - start) + 0.5) * duration,
            xticklabels=selection[start:stop] + 1,
            yticks=offsets,
            yticklabels=channels,
            xlim=(0, n_epochs * duration),
            ylim=(-spacing_uv / 2, offsets[0] + spacing_uv / 2),
            xlabel=(f"Original trial (each epoch: {times[0]:g} to {times[-1]:g} s; "
                    "dotted line: event onset)"),
            title=(f"Pre-AutoReject epochs {start + 1}–{stop} of {len(data)} "
                   "— red: rejected; black: retained"),
        )
        ax.plot([-0.025, -0.025], [offsets[0] - scaling_uv,
                                   offsets[0] + scaling_uv],
                transform=ax.get_yaxis_transform(), color="C5",
                linewidth=2, clip_on=False)
        ax.text(-0.035, offsets[0], f"{spacing_uv:.3g} µV", rotation=90,
                transform=ax.get_yaxis_transform(), color="C5",
                ha="right", va="center")
        figures.append(fig)
    plt.show(block=False)
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ELBO preprocessing and save epochs.")
    parser.add_argument("subject_id", nargs="?", default="sub-06")
    args = parser.parse_args()

    # %% Load example data
    subject_id = args.subject_id
    set_file = (Path(__file__).resolve().parents[3] / subject_id / "eeg"
                / f"{subject_id}_task-ColorPrecMemStudy_run-1_eeg.set")
    epochs_file = set_file.with_name(f"{set_file.stem}-epo.fif")
    epochs_clean_file = set_file.with_name(f"{set_file.stem}-epo_clean.fif")
    if epochs_clean_file.exists():
        epochs = mne.read_epochs(epochs_file, preload=True)
        print(f"Loaded pre-AutoReject epochs: {epochs_file}")
    else:
        raw, behavior = load_raw_data(subject_id, return_behav=True)

        # %% Detect nonfinite data and sparse flat spans before preprocessing
        bad_ch_report = {}
        # bad_ch_report.update(bad_ch_by_impedance(raw, initial_impedance))
        bad_ch_report.update(bad_ch_by_nan(raw))
        flatlines = detect_flatline_inplace(raw)
        mask_segment_by_flatline(raw, flatlines)
        bad_ch_report.update(bad_ch_by_flatline(raw, flatlines=flatlines))

        # %% Line noise removal
        raw_prep = raw.copy()
        raw_prep.filter(l_freq=1.0, h_freq=None, verbose=False)
        if subject_id == "sub-02":
            clean = Interpolate_fft(raw_prep, line_freq=[30.0, 90.0])
            clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=4.0)
        elif subject_id == "sub-26":
            clean = ZapLine_notch(raw_prep, line_freq=30.0)
            clean = Interpolate_fft(clean, line_freq=90.0)
            clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=1.5)
        else:
            clean = ZapLine_notch(raw_prep, line_freq=30.0)
            clean = ZapLine_notch(clean, line_freq=90.0)
            clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=1.5)
        remove_line_noise_inplace(raw, raw_prep, clean)  # without high-pass
        del raw_prep, clean  # Free two full-recording copies.

        # %% High-pass filter at 0.3 Hz for bad-channel detection
        raw_bads = raw.copy()
        raw_bads.filter(l_freq=0.3, h_freq=None, verbose=False)

        # Bad-channel detection and ASR correction in validated native reference
        asr_details, reference_channel = bad_ch_time_detection_inplace(
            raw_bads, bad_ch_report, flatlines,
        )

        # %% Fit ICA and select artifact components
        ica, fit_epochs, iclabel_table = fit_ica(
            raw_bads, reference_channel, return_details=True,
        )
        ica.plot_components()

        # Prepare and open the interactive viewer; each choice autosaves to JSON
        viewer_dir = prepare_ica_viewer(
            ica, fit_epochs, raw_bads,
            eeg_path=set_file,
            reference_channel=reference_channel,
            iclabel_table=iclabel_table,
            check_identity=True,
        )

        # Run these after reviewing; loading replaces, rather than unions, exclusions
        ica.exclude = load_ica_exclusions(viewer_dir, ica=ica)
        exclusion_figure = summarize_ica_exclusions(ica, fit_epochs, raw_bads)
        ica.apply(raw_bads)

        # %% Epoching
        epochs = prepare_epochs(raw_bads, behavior)
        epochs.save(epochs_file)
        print(f"Saved pre-AutoReject epochs: {epochs_file}")

    # Reject bad epochs
    ar = AutoReject(random_state=999)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    # Inspect the original traces, including epochs rejected by AutoReject
    frontal_cluster = [
        "E3", "E4", "E5", "E10", "E11", "E12", "E16", "E18", "E19", "E23",
    ]
    epoch_figures = plot_epoch_rejection(epochs, epochs_clean, frontal_cluster)

    # Optional manual rejection of additional bad epochs
    available_channels = set(epochs_clean.ch_names) - set(epochs_clean.info["bads"])
    valid_frontal_cluster = [c for c in frontal_cluster if c in available_channels]
    # epochs_clean.plot(
    #     picks=valid_frontal_cluster,
    #     n_epochs=10,
    #     block=True,
    # )

    # Save AutoReject and any additional manual rejections for Step 5
    epochs_clean.save(epochs_clean_file)
    print(f"Saved cleaned epochs: {epochs_clean_file}")
