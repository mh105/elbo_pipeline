"""ELBO step 3: native-reference ICA fitting and automatic artifact selection.

Fit on an internal 1-100 Hz copy, using annotation-clean 2-second epochs at
250 Hz. Locally interpolated and ASR-corrected data remain eligible. The zero
reference is omitted from fitting and restored only in ICLabel's mixing maps,
which are average-referenced. ICLabel's private activations use EEGLAB's
RMS-microvolt scaling without changing the fitted ICA or native source traces.

The input Raw is not modified. Apply the returned ICA directly to the original
0.3 Hz, native-rate data, preserving its reference, bad channels and annotations::

    ica, fit_epochs = fit_ica(raw_bads, reference_channel)
    ica.apply(raw_bads)
"""

import warnings
from pathlib import Path

from IPython.display import display
import mne
import numpy as np
import pandas as pd
from mne_icalabel.iclabel import run_iclabel
from mne_icalabel.iclabel.features import (
    _compute_ica_activations,
    _eeg_autocorr_fftw,
    _eeg_rpsd,
    _eeg_topoplot,
)
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


def prepare_ica_epochs(raw, reference_channel=None):
    """Return (filtered Raw copy, fit Epochs, effective reference name).

    Require preloaded 1000 Hz data. Resolve the reference as the unique good
    entirely-zero EEG channel, or validate the supplied effective reference.
    Keep good EEG and the reference in the private Raw; omit the reference from
    fitting. Reject every BAD-prefixed annotation, retaining repaired local spans.
    """
    if not raw.preload:
        raise ValueError("raw must be preloaded.")
    if raw.info["sfreq"] != 1000:
        raise ValueError("Expected 1000 Hz sampling rate for ICA epoch decimation.")
    eeg = mne.pick_types(
        raw.info,
        eeg=True,
        exclude="bads",
    )
    names = [raw.ch_names[pick] for pick in eeg]
    if reference_channel is None:
        zeros = [raw.ch_names[pick] for pick in eeg if np.all(raw._data[pick] == 0)]
        if len(zeros) != 1:
            raise ValueError(f"Expected one entirely-zero good EEG reference; found {zeros}.")
        reference_channel = zeros[0]
    elif (reference_channel not in names
          or not np.all(raw._data[raw.ch_names.index(reference_channel)] == 0)):
        raise ValueError("reference_channel must be a good entirely-zero EEG channel.")

    raw_ica = raw.copy().pick(names)

    skip_annotations = ("edge", "bad_acq_skip", "BAD")
    raw_ica.filter(
        l_freq=1.0,
        h_freq=None,
        skip_by_annotation=skip_annotations,
        verbose=False,
    )
    raw_ica.filter(
        l_freq=None,
        h_freq=100.0,
        h_trans_bandwidth=25.0,
        skip_by_annotation=skip_annotations,
        verbose=False,
    )
    fit_epochs = mne.make_fixed_length_epochs(
        raw_ica,
        duration=2.0,
        preload=True,
        reject_by_annotation=True,
        proj=False,
        overlap=0.0,
        verbose=False,
    )
    if len(fit_epochs) == 0:
        raise ValueError("No clean 2-second epochs remain for ICA fitting.")
    fit_epochs.drop_channels([reference_channel])
    with warnings.catch_warnings():
        # The low-pass stopband starts at the new Nyquist frequency (125 Hz).
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning,
            message=r"The measurement information indicates a low-pass frequency "
                    r"of 100\.0 Hz\. The decim=4 parameter will result in a sampling "
                    r"frequency of 250\.0 Hz, which can cause aliasing artifacts\.$",
        )
        fit_epochs.decimate(decim=4)
    if fit_epochs.info["sfreq"] != 250:
        raise ValueError("Expected 250 Hz sampling rate for ICA epochs.")
    return raw_ica, fit_epochs, reference_channel


def label_ica_components(raw_ica, fit_epochs, ica, reference_channel):
    """Return seven ICLabel probabilities using CAR maps and EEGLAB RMS scaling.

    Restore the effective reference's zero map row before centering electrode
    values. For the EEG-only fit, scale private activations by each CAR map's
    spatial RMS before PSD and autocorrelation extraction. Use ICLabel's private
    feature routines (mne-icalabel 0.9.0); leave the fitted ICA unchanged.
    """
    maps = np.vstack([ica.get_components(), np.zeros((1, ica.n_components_))])
    maps -= maps.mean(axis=0, keepdims=True)
    map_rms = np.sqrt(np.mean(maps ** 2, axis=0))
    activations = _compute_ica_activations(fit_epochs, ica)
    activations *= map_rms[:, None, None]
    psd = _eeg_rpsd(fit_epochs, ica, activations) * 0.99
    autocorr = _eeg_autocorr_fftw(fit_epochs, ica, activations) * 0.99
    map_names = ica.ch_names + [reference_channel]
    # Only positions are needed; avoid the topoplot routine copying a full Raw.
    map_info = mne.pick_info(raw_ica.info, [raw_ica.ch_names.index(name) for name in map_names])
    map_raw = mne.io.RawArray(np.zeros((len(map_names), 1)), map_info, verbose=False)
    topo = _eeg_topoplot(map_raw, maps, map_names) * 0.99
    return run_iclabel(topo, psd, autocorr, backend=None)


def exclude_ica_components(probabilities, iclabel_threshold=0.75):
    """Return exclusion indices and the unrounded table; display a rounded copy.

    Use a strict threshold on each of the five ICLabel artifact probabilities.
    """
    ICLABEL_CLASSES = (
        "brain", "muscle artifact", "eye blink", "heart beat", "line noise",
        "channel noise", "other",
    )
    exclude_indices = np.flatnonzero(
        np.any(probabilities[:, 1:6] > iclabel_threshold, axis=1)
    ).tolist()
    winners = np.argmax(probabilities, axis=1)
    components = np.arange(probabilities.shape[0])
    component_table = pd.DataFrame(probabilities, columns=ICLABEL_CLASSES)
    component_table.insert(0, "component", components)
    component_table.insert(1, "winning_label", [ICLABEL_CLASSES[index] for index in winners])
    component_table.insert(2, "winning_probability",
                           probabilities[components, winners])
    component_table["iclabel_auto_exclude"] = component_table["component"].isin(exclude_indices)
    component_table["excluded"] = component_table["component"].isin(exclude_indices)
    component_table.attrs.update(
        iclabel_threshold=float(iclabel_threshold),
        automatic_exclusion_classes=list(ICLABEL_CLASSES[1:6]),
        automatic_exclusion_comparison=">",
    )
    display(component_table.round(3))
    return exclude_indices, component_table


def fit_ica(
    raw,
    reference_channel=None,
    *,
    iclabel_threshold=0.75,
    return_details=False,
):
    """Return (ICA, fit Epochs) without modifying or applying to raw.

    Exclude components with any of the five ICLabel artifact probabilities
    strictly above iclabel_threshold (default 0.75).
    Always display the table rounded to three decimals;
    return (ICA, fit Epochs, unrounded component table) with return_details=True.
    The returned epochs are the exact data used for fitting and labeling;
    manual review and plotting are deferred.
    """
    raw_ica, fit_epochs, reference_channel = prepare_ica_epochs(raw, reference_channel)
    ica = mne.preprocessing.ICA(
        n_components=None,
        method="infomax",
        fit_params={"extended": True},
        max_iter="auto",
        rng=518,
    )
    ica.fit(fit_epochs, picks="eeg")
    print(f"ICA components retained: {ica.n_components_}")
    print(f"ICA training support: {len(fit_epochs)} epochs with {len(fit_epochs) * 2:.1f} s; "
          f"{len(ica.ch_names)} EEG channels; {ica.n_iter_} iterations.")

    probabilities = label_ica_components(raw_ica, fit_epochs, ica, reference_channel)
    exclude_indices, component_table = exclude_ica_components(
        probabilities,
        iclabel_threshold=iclabel_threshold,
    )
    ica.exclude = exclude_indices
    print(f"Excluded ICA components: {ica.exclude}")
    if return_details:
        return ica, fit_epochs, component_table
    return ica, fit_epochs


def prepare_ica_viewer(
    ica,
    fit_epochs,
    raw_bads,
    *,
    eeg_path,
    reference_channel,
    iclabel_table,
):
    """Save this fit and prepare its subject-local interactive review directory.

    Return a Path under eeg_path.parent / '.ica_viewer'. Repeating preparation
    for the same fit preserves its saved exclusions. The input ICA, fitting
    epochs and Raw are not modified, and ICA is neither refitted nor applied.
    """
    from step3_ica_viewer.storage import prepare_bundle

    return prepare_bundle(
        ica, fit_epochs, raw_bads,
        eeg_path=eeg_path,
        reference_channel=reference_channel,
        iclabel_table=iclabel_table,
    )


def open_ica_viewer(viewer_dir, *, open_browser=True):
    """Serve a prepared viewer with disk autosaving; return its local URL.

    The local service runs in the background while this Python process remains
    alive. Reopening the same directory restores its saved review decisions.
    """
    from step3_ica_viewer.server import open_viewer

    url = open_viewer(viewer_dir, open_browser=open_browser)
    print(f"ICA viewer: {url}")
    return url


def load_ica_exclusions(viewer_dir, *, ica):
    """Return the saved final exclusions after validating the target ICA.

    Assign the result to ica.exclude to replace its previous contents, including
    when the saved list is empty. This helper does not modify or apply ICA.
    """
    from step3_ica_viewer.storage import load_exclusions

    return load_exclusions(viewer_dir, ica=ica)


def summarize_ica_exclusions(ica, fit_epochs, raw_bads):
    """Print the reviewed exclusions and show three topomap/PSD pairs per row.

    Call before applying ICA to raw_bads. Measure the joint variance reduction
    as 1 - sum(channel temporal variances after exclusion) / sum(before), on
    exactly ica.ch_names in the original reference. Omit BAD-annotated samples
    from both variance estimates. This is the actual net reduction, not a sum
    of individual IC variance fractions; it can be negative.

    PSDs use the exact fitting epochs and their sampling rate, with the same
    native-unit median estimator as the viewer. Leave all inputs unchanged.
    Return the displayed figure, or None when no components are excluded.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from step3_ica_viewer.build import _median_psd

    excluded = sorted(set(ica.exclude))
    print(f"Excluded ICA components: {excluded}")
    print(f"Excluded components: {len(excluded)} / {ica.n_components_}")
    if not excluded:
        print("Variance reduction on unannotated original data: 0.000%")
        return None

    comparison = raw_bads.copy().pick(ica.ch_names)
    original = comparison.get_data(reject_by_annotation="omit", verbose=False)
    n_retained = original.shape[1]
    if not n_retained:
        raise ValueError("No unannotated samples remain for variance estimation.")
    before = np.var(original, axis=1).sum()
    del original
    if before <= 0:
        raise ValueError("The fitted channels must have positive temporal variance.")
    ica.apply(comparison, verbose=False)
    after = np.var(comparison.get_data(reject_by_annotation="omit", verbose=False),
                   axis=1).sum()
    reduction = 1 - after / before
    del comparison
    print(f"Variance reduction on unannotated original data: {100 * reduction:.3f}%")
    print(f"Variance support: {len(ica.ch_names)} fitted channels; "
          f"{n_retained} / {raw_bads.n_times} samples at {raw_bads.info['sfreq']:g} Hz, "
          "excluding BAD spans, in the original reference.")

    sources = ica.get_sources(fit_epochs).get_data(picks=excluded)
    sfreq = float(fit_epochs.info["sfreq"])
    frequencies, psd_db = _median_psd(sources, sfreq)
    maps = ica.get_components()
    n_rows = (len(excluded) + 2) // 3
    fig, axes = plt.subplots(
        n_rows, 6, figsize=(18, 2.8 * n_rows), squeeze=False,
        gridspec_kw={"width_ratios": [1, 1.4] * 3}, layout="constrained",
    )
    fig.suptitle(f"Excluded ICA components: {len(excluded)} / {ica.n_components_} · "
                 f"variance reduction {100 * reduction:.3f}%\n"
                 f"Median PSDs from {len(fit_epochs)} fitting epochs at {sfreq:g} Hz")
    for pair, component in enumerate(excluded):
        topo_ax, psd_ax = axes[pair // 3, 2 * (pair % 3):2 * (pair % 3) + 2]
        weights = maps[:, component]
        maximum = np.max(np.abs(weights))
        values = weights / maximum if maximum else np.zeros_like(weights)
        mne.viz.plot_topomap(
            values, ica.info, axes=topo_ax, vlim=(-1, 1), cmap="RdBu_r",
            sensors=True, contours=6, outlines="head", show=False,
        )
        for markers in topo_ax.collections:
            if (isinstance(markers, PathCollection)
                    and markers.get_offsets().shape == (len(ica.ch_names), 2)):
                markers.set_facecolor("#000000")
                markers.set_edgecolor("#39ff14")
                markers.set_sizes([10.0])
                markers.set_linewidths([0.65])
        topo_ax.set_title(f"ICA{component:03d}")
        psd_ax.plot(frequencies, psd_db[pair], color="#2b5371", linewidth=1)
        psd_ax.set(xlim=(0, min(50.0, sfreq / 2)), xlabel="Frequency (Hz)",
                   ylabel="PSD (dB AU²/Hz)", title="Median epoch PSD")
        positive = psd_db[pair, frequencies > 0]
        low, high = np.min(positive), np.max(positive)
        padding = max(1.0, 0.05 * (high - low))
        psd_ax.set_ylim(low - padding, high + padding)
        psd_ax.grid(alpha=0.25)
    for pair in range(len(excluded), 3 * n_rows):
        for ax in axes[pair // 3, 2 * (pair % 3):2 * (pair % 3) + 2]:
            ax.set_visible(False)
    plt.show()
    return fig


if __name__ == "__main__":
    # %% Load example data
    subject_id = "sub-03"
    eeg_path = (Path(__file__).resolve().parents[3] / subject_id / "eeg"
                / f"{subject_id}_task-ColorPrecMemStudy_run-1_eeg.set")
    raw = load_raw_data(subject_id)

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
    clean = ZapLine_notch(raw_prep, line_freq=30.0)
    clean = ZapLine_notch(clean, line_freq=90.0)
    clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=1.5)
    # clean = Interpolate_fft(raw_prep, line_freq=[30.0, 90.0])
    # clean = Interpolate_fft(clean, line_freq=60.0, bandwidth=4.0)
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
        eeg_path=eeg_path,
        reference_channel=reference_channel,
        iclabel_table=iclabel_table,
    )
    open_ica_viewer(viewer_dir)

    # %% Run this cell after reviewing; loading replaces, rather than unions, exclusions
    input("Review ICs in the interactive viewer,"
          "then press [Enter] to load the saved exclusions: ")
    ica.exclude = load_ica_exclusions(viewer_dir, ica=ica)
    exclusion_figure = summarize_ica_exclusions(ica, fit_epochs, raw_bads)

    # Apply reviewed ICA exclusion and view the cleaned data
    ica.apply(raw_bads)
    raw_bads.plot(title="After ICA cleaning")
