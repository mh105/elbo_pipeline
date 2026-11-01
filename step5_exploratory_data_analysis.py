"""ELBO step 5: exploratory analysis of cached epochs."""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from somata.multitaper.multitaper_spectrogram_python import multitaper_spectrogram


def average_mtm(signals):
    epoch_spectrograms = []
    for signal in signals:
        mt_spectrogram, stimes, sfreqs = multitaper_spectrogram(
            signal,
            Fs,
            frequency_range=[0, 30],
            time_bandwidth=2,
            num_tapers=3,
            window_params=[1, 0.05],
            min_nfft=min_nfft,
            detrend_opt="constant",
            plot_on=False,
            verbose=False,
        )
        epoch_spectrograms.append(mt_spectrogram)
    return np.mean(epoch_spectrograms, axis=0), stimes, sfreqs


if __name__ == "__main__":
    # %% Load example data
    subject_id = "sub-05"
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

    # %% Compare HIT and MISS multitaper spectrograms
    Fs = epochs.info["sfreq"]
    df = 0.05
    min_nfft = int(2 ** np.ceil(np.log2(Fs / df)))

    hit_spectrogram, stimes, sfreqs = average_mtm(hit_epoch_signals)
    miss_spectrogram, _, _ = average_mtm(miss_epoch_signals)
    spectrogram_times = stimes + epochs.times[0]

    # Plot HIT and MISS multitaper spectrograms side by side
    spectrograms = np.array([hit_spectrogram, miss_spectrogram])
    spectrograms_db = 10 * np.log10(spectrograms)
    vmin, vmax = np.nanpercentile(spectrograms_db, [5, 98])
    fig_spectrogram, axes_spectrogram = plt.subplots(
        1, 2, figsize=(12, 4), sharex=True, sharey=True, layout="constrained"
    )
    for ax_spectrogram, response_type, spectrogram_db, n_trials in zip(
        axes_spectrogram, ("HIT", "MISS"), spectrograms_db, (n_hits, n_misses)
    ):
        pm = ax_spectrogram.pcolormesh(
            spectrogram_times,
            sfreqs,
            spectrogram_db,
            shading="auto",
            cmap="cet_rainbow4",
            vmin=vmin,
            vmax=vmax,
        )
        ax_spectrogram.set(
            xlabel="Time (Second)",
            ylabel="Frequency (Hz)",
            ylim=(0, 30),
            title=f"{response_type} Multitaper Spectrogram (n={n_trials})",
        )
        for theta_edge in (4, 8):
            ax_spectrogram.axhline(theta_edge, color="black", linestyle=":")
    fig_spectrogram.colorbar(
        pm, ax=axes_spectrogram, label=r"PSD (dB re 1 $\mu V^2$/Hz)", shrink=0.8
    )

    # Plot the HIT minus MISS spectrogram beside the median spectra
    hit_spectrum = np.median(hit_spectrogram, axis=1)
    miss_spectrum = np.median(miss_spectrogram, axis=1)
    hit_spectrum_db = 10 * np.log10(hit_spectrum)
    miss_spectrum_db = 10 * np.log10(miss_spectrum)

    difference_spectrogram_db = spectrograms_db[0] - spectrograms_db[1]
    difference_limit = np.nanmax(np.abs(difference_spectrogram_db))
    fig_spectrum, (ax_difference, ax_spectrum) = plt.subplots(
        1, 2, figsize=(12, 4), layout="constrained"
    )
    pm_difference = ax_difference.pcolormesh(
        spectrogram_times,
        sfreqs,
        difference_spectrogram_db,
        shading="auto",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    fig_spectrum.colorbar(
        pm_difference, ax=ax_difference, label="HIT - MISS (dB)", shrink=0.8
    )
    ax_difference.set(
        xlabel="Time (Second)",
        ylabel="Frequency (Hz)",
        ylim=(0, 30),
        title="HIT - MISS Multitaper Spectrogram",
    )
    for theta_edge in (4, 8):
        ax_difference.axhline(theta_edge, color="black", linestyle=":")

    ax_spectrum.plot(sfreqs, hit_spectrum_db, label=f"HIT (n={n_hits})")
    ax_spectrum.plot(sfreqs, miss_spectrum_db, label=f"MISS (n={n_misses})")
    ax_spectrum.set(
        xlabel="Frequency (Hz)",
        ylabel=r"PSD (dB re 1 $\mu V^2$/Hz)",
        xlim=(0, 30),
        title="HIT vs MISS Median Spectrum",
    )

    for theta_edge in (4, 8):
        ax_spectrum.axvline(theta_edge, color="black", linestyle=":")
    ax_spectrum.legend()

    plt.show()
