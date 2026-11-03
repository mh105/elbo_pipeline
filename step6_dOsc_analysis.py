"""ELBO step 6: dOsc state-space modeling analysis."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from somata.oscillator_search import DecomposedOscillatorModel as DecOsc


if __name__ == "__main__":
    # %% Load example data
    subject_id = "sub-02"
    project_dir = Path(__file__).resolve().parents[3]
    epochs_file = (project_dir / subject_id / "eeg"
                   / f"{subject_id}_task-ColorPrecMemStudy_run-1_eeg-clean-epo.fif")
    if not epochs_file.exists():
        raise FileNotFoundError(f"Epochs file not found: {epochs_file}")

    behavior_file = (project_dir / subject_id / "beh"
                     / f"{subject_id}_task-ColorPrecMemStudy_labelled_beh.csv")
    behavior = pd.read_csv(behavior_file, keep_default_na=False)

    epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
    if len(behavior) != len(epochs.drop_log):
        raise ValueError(
            f"Behavior rows ({len(behavior)}) do not match cached epoch trials "
            f"({len(epochs.drop_log)})."
        )

    # %% Extract HIT and MISS frontal signals and check trial counts
    frontal_cluster = [
        "E3", "E4", "E5", "E10", "E11", "E12", "E16", "E18", "E19", "E23",
    ]
    available_channels = set(epochs.ch_names) - set(epochs.info["bads"])
    valid_frontal_cluster = [c for c in frontal_cluster if c in available_channels]
    frontal_epoch_signals = [
        epoch.mean(axis=0)
        for epoch in epochs.get_data(picks=valid_frontal_cluster, units="uV")
    ]
    retained_response_types = behavior.iloc[epochs.selection]["response_type"].to_numpy()
    hit_epoch_signals = [
        signal
        for signal, response_type in zip(frontal_epoch_signals, retained_response_types)
        if response_type == "HIT"
    ]
    miss_epoch_signals = [
        signal
        for signal, response_type in zip(frontal_epoch_signals, retained_response_types)
        if response_type == "MISS"
    ]
    n_hits = len(hit_epoch_signals)
    n_misses = len(miss_epoch_signals)
    if n_hits == 0 or n_misses == 0:
        raise ValueError("At least one valid HIT and MISS epoch is required.")

    # %% Fit dOsc model to HIT and MISS frontal signals
    Fs = epochs.info["sfreq"]
    stimulus_onset_idx = epochs.time_as_index(0, use_rounding=True)[0]

    hit_oscillators = []
    for signal in hit_epoch_signals:
        y = signal[stimulus_onset_idx:]
        do1 = DecOsc(y, Fs, noise_start=None, osc_range=7, iterate=True)
        if do1.knee_index == 0:
            do1.knee_index = np.argmax(do1.ll)
        hit_oscillators.append(do1)

    miss_oscillators = []
    for signal in miss_epoch_signals:
        y = signal[stimulus_onset_idx:]
        do1 = DecOsc(y, Fs, noise_start=None, osc_range=7, iterate=True)
        if do1.knee_index == 0:
            do1.knee_index = np.argmax(do1.ll)
        miss_oscillators.append(do1)

    # %% Collect HIT and MISS frequencies within 3.5-7.5 Hz
    hit_frequencies = np.empty(len(hit_oscillators), dtype=object)
    for trial_idx, hit_oscillator in enumerate(hit_oscillators):
        freqs = hit_oscillator.get_knee_osc().freq
        hit_frequencies[trial_idx] = freqs[(freqs >= 3.5) & (freqs <= 7.5)]

    miss_frequencies = np.empty(len(miss_oscillators), dtype=object)
    for trial_idx, miss_oscillator in enumerate(miss_oscillators):
        freqs = miss_oscillator.get_knee_osc().freq
        miss_frequencies[trial_idx] = freqs[(freqs >= 3.5) & (freqs <= 7.5)]

    n_hits_with_band_oscillator = sum(freqs.size > 0 for freqs in hit_frequencies)
    n_hits_without_band_oscillator = len(hit_frequencies) - n_hits_with_band_oscillator
    print(
        f"HIT trials with a 3.5-7.5 Hz oscillator: "
        f"{n_hits_with_band_oscillator / len(hit_frequencies):.1%} "
        f"({n_hits_with_band_oscillator}/{n_hits_without_band_oscillator} with/without)"
    )

    n_misses_with_band_oscillator = sum(freqs.size > 0 for freqs in miss_frequencies)
    n_misses_without_band_oscillator = len(miss_frequencies) - n_misses_with_band_oscillator
    print(
        f"MISS trials with a 3.5-7.5 Hz oscillator: "
        f"{n_misses_with_band_oscillator / len(miss_frequencies):.1%} "
        f"({n_misses_with_band_oscillator}/{n_misses_without_band_oscillator} with/without)"
    )
