import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_denoise.zapline import ZapLine
from mne_denoise.spectrum_interpolation import SpectrumInterpolation
from somata.multitaper.multitaper_spectrogram_python import multitaper_spectrogram
from step0_data_quality_check import load_raw_data


def ZapLine_notch(raw_prep, line_freq):
    """Return a Raw copy cleaned at line_freq (Hz) on good EEG channels.

    Fit standard ZapLine on its usual moving-average residual, with its default
    harmonic bias. Apply the learned artifact projection only to the difference
    between the input and a copy notched at line_freq. Uses mne-denoise 0.0.3's fitted
    channel names and selected-component filters/patterns.
    """
    sfreq = float(raw_prep.info["sfreq"])
    model = ZapLine(sfreq=sfreq, line_freq=line_freq, n_select="auto")
    model.fit(raw_prep.copy().pick("eeg", exclude="bads"))

    clean = raw_prep.copy().load_data()
    if model.n_removed_ == 0:
        return clean

    # Match the exact channel set and order used during fitting.
    picks = mne.pick_channels(
        raw_prep.ch_names, include=model._mne_ch_names_, ordered=True
    )
    raw_notched = raw_prep.copy().load_data()
    raw_notched.notch_filter(
        freqs=[line_freq], picks=picks, method="fir", phase="zero", verbose=False,
    )
    notched = raw_notched.get_data(picks=picks)
    residual = raw_prep.get_data(picks=picks) - notched

    # Do not call transform(): that would create the moving-average split again.
    sources = model.filters_ @ residual
    artifact = model.patterns_ @ sources
    clean._data[picks] = notched + (residual - artifact)
    return clean


def Interpolate_fft(raw_prep, line_freq, n_harmonics=1, bandwidth=1.0):
    """Return a Raw copy cleaned by amplitude-spectrum interpolation.

    line_freq is the fundamental frequency in Hz, or a list of explicit targets.
    For a fundamental, n_harmonics counts multiples starting with the fundamental:
    1 targets only line_freq; None targets all multiples below Nyquist.
    bandwidth is the half-width in Hz around each target, so bandwidth=1.5
    replaces the band from target - 1.5 Hz to target + 1.5 Hz.

    Replace Fourier magnitudes in each target band with the mean magnitude of
    the adjacent reference bands (each 2 Hz wide by default), retaining phase.
    Sampling frequency is inferred from raw_prep; the input is not modified.
    """
    model = SpectrumInterpolation(
        line_freq=line_freq,
        n_harmonics=n_harmonics,
        bandwidth=bandwidth,
    )
    clean = model.fit_transform(raw_prep)
    return clean


def remove_line_noise_inplace(raw, raw_prep, clean):
    """Subtract estimated line noise from raw in place; return None.

    raw must be preloaded; raw_prep is the filtered copy used to produce clean.
    All three inputs must have matching channels, order, and samples.
    Estimate the removed noise as raw_prep - clean.
    Only this difference is subtracted; raw retains its original filter metadata.
    """
    line_noise = raw_prep.get_data() - clean.get_data()
    raw._data -= line_noise


if __name__ == "__main__":
    # %% Load example data
    raw = load_raw_data("sub-02")

    # %% Line noise removal

    # For line noise removal, first highpass filter at 1.0 Hz to remove slow drifts
    raw_prep = raw.copy()
    raw_prep.filter(l_freq=1.0, h_freq=None, verbose=False)

    # %% Some manual inspection of the data before line-noise removal
    # Inspect the spectra across channels
    fmax = 150  # <-- adjust fmax as needed
    psd_before = raw_prep.compute_psd(fmax=fmax)
    psd_before.plot(picks="data", exclude=['Cz'], amplitude=False)

    # Inspect the posterior channel before any line-noise removal.
    posterior_channel = "E75"  # Posterior midline electrode in this EGI montage.
    sfreq = float(raw_prep.info["sfreq"])
    spectrogram_params = dict(
        frequency_range=[1, fmax],
        time_bandwidth=2,
        num_tapers=3,
        window_params=[2, 1],  # Two-second windows, one-second step.
        min_nfft=int(2 ** np.ceil(np.log2(sfreq / 0.05))),
        detrend_opt="constant",
        plot_on=False,
        verbose=False,
    )
    power_before, times, freqs = multitaper_spectrogram(
        raw_prep.get_data(picks=[posterior_channel], units="uV")[0],
        sfreq, **spectrogram_params,
    )
    power_before_db = 10 * np.log10(
        np.where(power_before > 0, power_before, np.nan)
    )
    vmin, vmax = np.nanpercentile(power_before_db, [5, 98])
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    mesh = ax.pcolormesh(times, freqs, power_before_db, shading="auto",
                         vmin=vmin, vmax=vmax, cmap="cet_rainbow4")
    ax.set_title(f"{posterior_channel}: Before line-noise removal (1 Hz high-pass)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time from start of cropped recording (s)")
    fig.colorbar(mesh, ax=ax, label=r"PSD (dB re 1 $\mu V^2$/Hz)")
    plt.show()

    # %% Apply line-noise removal
    # clean = ZapLine_notch(raw_prep, line_freq=30.0)
    # clean = ZapLine_notch(clean, line_freq=90.0)
    # clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=1.5)

    # sub-02 - when ZapLine fails, use SpectrumInterpolation for 30 Hz and 90 Hz as well
    clean = Interpolate_fft(raw_prep, line_freq=[30.0, 90.0])
    clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=4.0)

    # Remove the estimated noise from the original data without high-pass filtering it
    remove_line_noise_inplace(raw, raw_prep, clean)

    # %% Some more manual inspection of the data after line-noise removal
    # Need to re-apply the 1 Hz high-pass filter to the modified raw data without
    # line noise for comparison, since the original raw was not high-pass filtered.
    raw_prep = raw.copy()
    raw_prep.filter(l_freq=1.0, h_freq=None, verbose=False)

    # Inspect the spectra across channels
    psd_after = raw_prep.compute_psd(fmax=fmax)
    psd_after.plot(picks="data", exclude=['Cz'], amplitude=False)

    # Inspect a posterior channel's spectrogram before and after cleaning
    power_after, _, _ = multitaper_spectrogram(
        raw_prep.get_data(picks=[posterior_channel], units="uV")[0],
        sfreq, **spectrogram_params,
    )
    power_after_db = 10 * np.log10(np.where(power_after > 0, power_after, np.nan))
    power_db = np.stack([power_before_db, power_after_db])
    vmin, vmax = np.nanpercentile(power_db, [5, 98])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True,
                             layout="constrained")
    for ax, spectrum, title in zip(axes, power_db, ("Before cleaning", "After cleaning")):
        mesh = ax.pcolormesh(times, freqs, spectrum, shading="auto",
                             vmin=vmin, vmax=vmax, cmap="cet_rainbow4")
        ax.set_title(f"{posterior_channel}: {title}")
        ax.set_ylabel("Frequency (Hz)")
    axes[-1].set_xlabel("Time from start of cropped recording (s)")
    fig.colorbar(mesh, ax=axes, label=r"PSD (dB re 1 $\mu V^2$/Hz)")
    plt.show()

    def mean_psd_db(psd):
        """Convert each channel's power to dB re 1 uV^2/Hz, then average channels."""
        data, freqs = psd.get_data(
            picks="data", exclude=["Cz"], return_freqs=True
        )  # (n_channels, n_freqs), V^2/Hz
        return freqs, (10 * np.log10(data * 1e12)).mean(axis=0)

    # Channel-averaged PSD, before vs after line-noise removal
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(*mean_psd_db(psd_before), label="before cleaning")
    ax.plot(*mean_psd_db(psd_after), label="after cleaning")
    ax.set(
        xlabel="Frequency (Hz)",
        ylabel=r"PSD (dB re 1 $\mu$V$^2$/Hz)",
        title="Channel-averaged PSD",
    )
    ax.legend()

    # Per-channel attenuation (before - after), overlaid across channels
    psd_power_before, psd_freqs = psd_before.get_data(
        picks="data", exclude=["Cz"], return_freqs=True
    )  # (n_channels, n_freqs), V^2/Hz
    psd_power_after = psd_after.get_data(picks="data", exclude=["Cz"])

    # Per channel: 10*log10(before) - 10*log10(after) == 10*log10(before / after)
    atten_per_channel = 10 * np.log10(psd_power_before / psd_power_after)
    # Mean: average the per-channel dB changes, matching mean_psd_db.
    atten_mean = atten_per_channel.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(psd_freqs, atten_per_channel.T, color="C0", alpha=0.5, lw=0.8)
    ax.plot(psd_freqs, atten_mean, color="k", lw=2.5, label="mean across channels")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set(
        xlabel="Frequency (Hz)",
        ylabel="Power removed (dB)",
        title="Line-noise removal attenuation, per channel",
    )
    ax.legend()

    # Distribution of the line-noise removal impact across channels
    # Both PSDs use a 1 Hz high-pass filter; sub-1 Hz bins lie in its transition band.
    BAND = (0.5, 20.0)

    # 0 dB == channel untouched, negative == power removed
    delta_db = 10 * np.log10(psd_power_after / psd_power_before)
    pct = np.percentile(delta_db, [5, 25, 50, 75, 95], axis=0)

    df = psd_freqs[1] - psd_freqs[0]
    band_mask = (psd_freqs >= BAND[0]) & (psd_freqs <= BAND[1])
    band_before = psd_power_before[:, band_mask].sum(axis=1) * df  # V^2
    band_after = psd_power_after[:, band_mask].sum(axis=1) * df
    band_delta_db = 10 * np.log10(band_after / band_before)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

    ax = axes[0]
    ax.fill_between(psd_freqs, pct[0], pct[4], color="C0", alpha=0.2, lw=0, label="5-95%")
    ax.fill_between(psd_freqs, pct[1], pct[3], color="C0", alpha=0.4, lw=0, label="25-75%")
    ax.plot(psd_freqs, pct[2], color="k", lw=2, label="median")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.axvspan(*BAND, color="C1", alpha=0.12, zorder=0)
    ax.set(
        xlabel="Frequency (Hz)",
        ylabel="Power change (dB)",
        title="Per-channel change, distribution across channels",
    )
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.scatter(10 * np.log10(band_before * 1e12), band_delta_db, s=18, alpha=0.7)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set(
        xlabel=f"Channel power, {BAND[0]}-{BAND[1]} Hz (dB re 1 $\\mu$V$^2$)",
        ylabel="Band power change (dB)",
        title=f"{BAND[0]}-{BAND[1]} Hz impact vs channel amplitude",
    )

    psd_ch_names = psd_before.copy().pick("data", exclude=["Cz"]).ch_names
    worst = int(np.argmin(band_delta_db))
    print(
        f"{BAND[0]}-{BAND[1]} Hz change (dB): "
        f"median {np.median(band_delta_db):+.3f}, "
        f"range [{band_delta_db.min():+.3f}, {band_delta_db.max():+.3f}], "
        f"most affected channel {psd_ch_names[worst]}"
    )

    # Worst-case single channels at 20 Hz and 80 Hz, before vs after
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True,
                             layout="constrained")
    for ax, f_probe in zip(axes, (20.0, 80.0)):
        i_freq = int(np.argmin(np.abs(psd_freqs - f_probe)))
        i_ch = int(np.argmin(delta_db[:, i_freq]))  # most negative == most power lost
        ax.plot(psd_freqs, 10 * np.log10(psd_power_before[i_ch] * 1e12),
                label="before cleaning", lw=2.5)
        ax.plot(psd_freqs, 10 * np.log10(psd_power_after[i_ch] * 1e12),
                label="after cleaning", lw=2.5, alpha=0.5)
        ax.axvline(psd_freqs[i_freq], color="0.6", lw=0.8, ls=":")
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel=r"PSD (dB re 1 $\mu$V$^2$/Hz)",
            title=(
                f"{psd_ch_names[i_ch]}: largest loss at {psd_freqs[i_freq]:.1f} Hz "
                f"({delta_db[i_ch, i_freq]:+.2f} dB)"
            ),
        )
        ax.legend()
    plt.show()
