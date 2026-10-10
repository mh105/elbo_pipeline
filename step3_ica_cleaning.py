"""ELBO step 3: native-reference ICA fitting and automatic artifact selection.

Fit on an internal 1-100 Hz copy, using annotation-clean 2-second epochs at
250 Hz. Locally interpolated and ASR-corrected data remain eligible. The zero
reference is omitted from fitting and restored only in ICLabel's mixing maps,
which are average-referenced without changing the native component traces.

The input Raw is not modified. Apply the returned ICA directly to the original
0.3 Hz, native-rate data, preserving its reference, bad channels and annotations::

    ica = fit_ica(raw_bads, reference_channel)
    ica.apply(raw_bads)
"""

import warnings

from IPython.display import display
import mne
import numpy as np
import pandas as pd
from mne_icalabel.iclabel import get_iclabel_features, run_iclabel
from mne_icalabel.iclabel.features import _eeg_topoplot
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


def find_eog_components(raw_ica, ica, *, eog_pairs):
    """Add usable virtual EOG pairs to the private Raw; return indices and scores.

    Use MNE's 1-10 Hz correlation scoring and iterative z-score threshold 3.
    Each EOG pair contains (name, anode, cathode).
    Skip a pair when either source is missing or bad. EEG traces are unchanged.
    """
    usable_pairs = []
    for name, anode, cathode in eog_pairs:
        unusable = [channel for channel in (anode, cathode)
                    if channel not in raw_ica.ch_names or channel in raw_ica.info["bads"]]
        if unusable:
            print(f"Skipping {name}; source channel(s) missing or bad: {', '.join(unusable)}")
        else:
            usable_pairs.append((name, anode, cathode))
    if not usable_pairs:
        print("Skipping find_bads_eog(); no usable virtual EOG channels remain.")
        return [], None

    eog_names = [name for name, _, _ in usable_pairs]
    mne.set_bipolar_reference(
        raw_ica, anode=[anode for _, anode, _ in usable_pairs],
        cathode=[cathode for _, _, cathode in usable_pairs], ch_name=eog_names,
        drop_refs=False, copy=False, on_bad="raise",
    )
    raw_ica.set_channel_types({name: "eog" for name in eog_names})
    indices, scores = ica.find_bads_eog(
        raw_ica, ch_name=eog_names, reject_by_annotation=True, verbose=False,
    )
    indices = [int(index) for index in indices]
    print(f"Candidate EOG components: {indices}")
    return indices, scores


def label_ica_components(raw_ica, fit_epochs, ica, reference_channel):
    """Return seven ICLabel probabilities using CAR maps and native time features.

    Restore the effective reference's zero map row before centering electrode
    values. Use ICLabel's private topoplot routine to preserve its interpolation
    and normalization (mne-icalabel 0.9.0); leave the fitted ICA unchanged.
    """
    with warnings.catch_warnings():
        # Our replacement map feature supplies the CAR representation explicitly.
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning,
            message="The provided Epochs instance does not seem to be referenced "
                    "to a common average reference.*",
        )
        _, psd, autocorr = get_iclabel_features(fit_epochs, ica)

    maps = np.vstack([ica.get_components(), np.zeros((1, ica.n_components_))])
    maps -= maps.mean(axis=0, keepdims=True)
    map_names = ica.ch_names + [reference_channel]
    # Only positions are needed; avoid the topoplot routine copying a full Raw.
    map_info = mne.pick_info(raw_ica.info, [raw_ica.ch_names.index(name) for name in map_names])
    map_raw = mne.io.RawArray(np.zeros((len(map_names), 1)), map_info, verbose=False)
    topo = _eeg_topoplot(map_raw, maps, map_names) * 0.99
    return run_iclabel(topo, psd, autocorr, backend=None)


def exclude_ica_components(eog_indices, probabilities, iclabel_threshold):
    """Return exclusion indices and the unrounded table; display a rounded copy.

    Raise if an EOG candidate has brain/other probability above the threshold.
    """
    ICLABEL_CLASSES = (
        "brain", "muscle artifact", "eye blink", "heart beat", "line noise",
        "channel noise", "other",
    )
    conflicts = [index for index in eog_indices
                 if np.any(probabilities[index, [0, 6]] > iclabel_threshold)]
    if conflicts:
        scores = "; ".join(
            f"{index}: brain={probabilities[index, 0]:.6f}, other={probabilities[index, 6]:.6f}"
            for index in conflicts
        )
        raise ValueError(
            f"EOG candidates conflict with ICLabel brain/other >{iclabel_threshold}: {scores}"
        )

    iclabel_indices = np.flatnonzero(
        np.any(probabilities[:, 1:6] > iclabel_threshold, axis=1)
    ).tolist()
    exclude_indices = sorted(set(eog_indices + iclabel_indices))
    winners = np.argmax(probabilities, axis=1)
    components = np.arange(probabilities.shape[0])
    component_table = pd.DataFrame(probabilities, columns=ICLABEL_CLASSES)
    component_table.insert(0, "component", components)
    component_table.insert(1, "winning_label", [ICLABEL_CLASSES[index] for index in winners])
    component_table.insert(2, "winning_probability",
                           probabilities[components, winners])
    component_table["find_bads_eog"] = component_table["component"].isin(eog_indices)
    component_table["iclabel_auto_exclude"] = component_table["component"].isin(iclabel_indices)
    component_table["excluded"] = component_table["component"].isin(exclude_indices)
    display(component_table.round(3))
    return exclude_indices, component_table


def fit_ica(
    raw,
    reference_channel=None,
    *,
    iclabel_threshold=0.75,
    eog_pairs=(("VEOG-R", "E14", "E126"), ("VEOG-L", "E21", "E127")),
    return_details=False,
):
    """Return ICA without modifying or applying to raw.

    Exclude EOG candidates OR any of the five ICLabel artifact probabilities
    strictly above iclabel_threshold (default 0.75). Raise if an EOG candidate
    has brain or other probability strictly above the same threshold. EOG pairs
    contain (name, anode, cathode), defaulting to VEOG-R and VEOG-L.
    Always display the table rounded to three decimals;
    return (ICA, unrounded component table) with return_details=True.
    Manual review and plotting are deferred.
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
    print(f"ICA training support: {len(fit_epochs)} epochs / {len(fit_epochs) * 2:.1f} s; "
          f"{len(ica.ch_names)} EEG channels; {ica.n_iter_} iterations.")

    eog_indices, _ = find_eog_components(raw_ica, ica, eog_pairs=eog_pairs)
    probabilities = label_ica_components(raw_ica, fit_epochs, ica, reference_channel)
    exclude_indices, component_table = exclude_ica_components(
        eog_indices,
        probabilities,
        iclabel_threshold,
    )
    ica.exclude = exclude_indices
    print(f"Excluded ICA components: {ica.exclude}")
    if return_details:
        return ica, component_table
    return ica


if __name__ == "__main__":
    # %% Load example data
    raw = load_raw_data("sub-03")

    # %% Detect nonfinite data and sparse flat spans before preprocessing
    bad_ch_report = {}
    # bad_ch_report.update(bad_ch_by_impedance(raw, initial_impedance))
    bad_ch_report.update(bad_ch_by_nan(raw))
    flatlines = detect_flatline_inplace(raw)  # Also annotates global dropouts.
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
    bad_ch_report, asr_details, reference_channel = bad_ch_time_detection_inplace(
        raw_bads, bad_ch_report=bad_ch_report, flatlines=flatlines,
    )

    # %% Fit ICA and select artifact components
    ica = fit_ica(raw_bads, reference_channel)
