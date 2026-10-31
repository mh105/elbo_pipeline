"""ELBO step 4: epoching scaffold.

Run Steps 0-3 to prepare ICA-cleaned continuous data for epoching.
Epoch definitions and rejection settings remain to be specified.
"""

import warnings
from pathlib import Path

import mne
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


if __name__ == "__main__":
    # %% Load example data
    subject_id = "sub-26"
    set_file = (Path(__file__).resolve().parents[3] / subject_id / "eeg"
                / f"{subject_id}_task-ColorPrecMemStudy_run-1_eeg.set")
    epochs_file = set_file.with_name(f"{set_file.stem}-epo.fif")
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

    # %% Bad-channel detection and ASR correction in validated native reference
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
    skip_annotations = ("edge", "bad_acq_skip", "BAD")
    raw_bads.filter(
        l_freq=None,
        h_freq=45,
        h_trans_bandwidth=5,
        skip_by_annotation=skip_annotations,
    )

    events_from_annot, event_dict = mne.events_from_annotations(raw_bads)
    stim_events = mne.pick_events(events_from_annot, include=event_dict["encm"])
    if len(behavior) != len(stim_events):
        raise ValueError(
            f"Behavior rows ({len(behavior)}) do not match encm events "
            f"({len(stim_events)})."
        )
    epochs = mne.Epochs(
        raw_bads,
        stim_events,
        tmin=-0.5,
        tmax=2.5,
        preload=True,
        baseline=(None, 0),
    )
    if epochs.info["sfreq"] != 1000:
        raise ValueError("Expected 1000 Hz sampling rate.")
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
        raise ValueError("Expected 100 Hz sampling rate.")

    # %% Reject bad epochs
    ar = AutoReject(random_state=42)
    epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    # Manual review of epochs
    frontal_cluster = [
        "E3", "E4", "E5", "E10", "E11", "E12", "E16", "E18", "E19", "E23",
    ]
    available_channels = set(epochs.ch_names) - set(epochs.info["bads"])
    valid_frontal_cluster = [c for c in frontal_cluster if c in available_channels]
    epochs.plot(
        picks=valid_frontal_cluster,
        n_epochs=10,
        block=True,
    )

    # Save the epoch file after manual review
    epochs.save(epochs_file)
    print(f"Saved reviewed epochs: {epochs_file}")
