import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


def make_raw(data, sfreq=250, first_samp=0, meas_date=None, average=False):
    """Use a native reference or opposite-polarity pairs with zero channel mean."""
    data = np.asarray(data, dtype=float)
    names = [f"E{i}" for i in range(len(data))]
    if average:
        names += [f"M{i}" for i in range(len(data))]
        data = np.vstack([data, -data])
    else:
        names += ["Cz"]
        data = np.vstack([data, np.zeros(data.shape[1])])
    info = mne.create_info(names, sfreq, "eeg")
    info.set_meas_date(meas_date)
    return mne.io.RawArray(
        data, info,
        first_samp=first_samp, verbose=False,
    )


def std_raw(n_bad=2, sfreq=250, window_length=2):
    samples = int(sfreq * window_length)
    amplitudes = np.array([0.01, 10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7])
    windows = np.tile(amplitudes, (4, 1))
    windows[:n_bad, -1] = 100
    wave = np.tile([-1, 1], samples // 2)
    data = (windows.T[..., None] * wave).reshape(len(amplitudes), -1) * 1e-6
    return make_raw(data, sfreq, average=True)


def hf_raw(n_bad=2, sfreq=250, window_length=2):
    samples = int(sfreq * window_length)
    t = np.arange(4 * samples) / sfreq
    # HF/LF ratios differ slightly across channels, so their robust z has a
    # real spread rather than the machine-epsilon floor.
    data = np.array([
        (10 + i / 10) * np.sin(2 * np.pi * 10 * t)
        + (1 + i / 50) * np.sin(2 * np.pi * 80 * t)
        for i in range(9)
    ]) * 1e-6
    data[0] += 100e-6 * np.sin(2 * np.pi * 10 * t)  # Strong LF alone is not HF noise.
    data[-1, :n_bad * samples] += 20e-6 * np.sin(2 * np.pi * 80 * t[:n_bad * samples])
    return make_raw(data, sfreq)


def bb_raw(n_bad=2, sfreq=250, window_length=2):
    samples = int(sfreq * window_length)
    rng = np.random.default_rng(42)
    data = rng.normal(size=(13, 4 * samples)) * np.linspace(1, 1.3, 13)[:, None] * 1e-6
    t = np.arange(4 * samples) / sfreq
    data[0] += 80e-6 * np.sin(2 * np.pi * 10 * t)  # Narrowband outlier.
    data[-2:, :n_bad * samples] *= 30
    return make_raw(data, sfreq, average=True)


def hf_ratio(data, sfreq):
    """PREP's HF ratio of data low-passed on its own."""
    return step2._prep_hf_ratio(data, step2._lowpass_45_hz(data, sfreq))


WINDOW_DETECTORS = (
    (step2.bad_ch_by_std, std_raw, ["E8", "M8"]),
    (step2.bad_ch_by_HFnoise, hf_raw, ["E8"]),
    (step2.bad_ch_by_BBnoise, bb_raw, ["E11", "E12", "M11", "M12"]),
)


class TestVoltageMask(unittest.TestCase):
    def test_strict_thresholds_merge_partial_tail_and_channel_exclusions(self):
        data = np.ones((6, 45)) * 1e-6
        data[0, 1] = 501e-6  # Exactly one of four eligible channels: do not mask.
        data[1, 2] = -500e-6  # Equality is not an exceedance.
        data[0, 11] = 501e-6
        data[1, 18] = -501e-6  # Exceedances need not be simultaneous.
        data[:2, 21] = 501e-6
        data[:2, 41] = 501e-6
        data[4:] = 1e-3
        raw = make_raw(data, sfreq=100)
        raw.info["bads"] = ["E4"]
        raw.set_channel_types({"E5": "eog"})
        raw.set_annotations(mne.Annotations([0.35], [0], ["event"]))
        original = raw.get_data().copy()
        self.assertIsNone(step2.mask_segment_by_voltage(raw))
        mask = raw.annotations.description == "BAD_voltage_mask"
        np.testing.assert_allclose(raw.annotations.onset[mask], [0.1, 0.4])
        np.testing.assert_allclose(raw.annotations.duration[mask], [0.2, 0.05])
        self.assertIn("event", raw.annotations.description)
        self.assertEqual(raw.info["bads"], ["E4"])
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_cropped_time_base_with_and_without_measurement_date(self):
        for meas_date in (None, datetime(2026, 9, 27, tzinfo=timezone.utc)):
            with self.subTest(meas_date=meas_date):
                data = np.ones((4, 200)) * 1e-6
                data[:2, 71] = 501e-6
                raw = make_raw(data, sfreq=100, first_samp=250, meas_date=meas_date)
                raw.crop(tmin=0.5)
                step2.mask_segment_by_voltage(raw)
                np.testing.assert_allclose(raw.annotations.onset, [3.2])
                np.testing.assert_allclose(raw.annotations.duration, [0.1])
                self.assertEqual(raw.annotations.orig_time, meas_date)
                kept = raw.get_data(picks=[0], reject_by_annotation="omit", verbose=False)
                self.assertEqual(kept.shape[1], 140)

    def test_mask_parameters_and_invalid_data(self):
        raw = make_raw(np.ones((4, 100)) * 1e-6, sfreq=100)
        raw._data[0, 60] = 101e-6
        step2.mask_segment_by_voltage(
            raw, voltage_threshold_uv=100, channel_fraction_cutoff=0.2, window_length=0.5
        )
        np.testing.assert_allclose(raw.annotations.onset, [0.5])
        np.testing.assert_allclose(raw.annotations.duration, [0.5])
        for kwargs in (
            {"voltage_threshold_uv": 0}, {"voltage_threshold_uv": np.inf},
            {"channel_fraction_cutoff": -0.1}, {"channel_fraction_cutoff": 1.1},
            {"window_length": 0.001}, {"window_length": np.nan},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                step2.mask_segment_by_voltage(raw, **kwargs)
        raw._data[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "bad_ch_by_nan"):
            step2.mask_segment_by_voltage(raw)


class TestVoltageDetection(unittest.TestCase):
    def test_sample_occurrence_and_signed_voltage_boundaries(self):
        raw = make_raw(np.ones((5, 100)) * 1e-6, sfreq=100)
        raw._data[0, :26] = 201e-6
        raw._data[1, :25] = -201e-6
        raw._data[2] = np.tile([-200e-6, 200e-6], 50)
        raw._data[4] = np.nan
        raw.info["bads"] = ["E4"]
        original = raw.get_data().copy()
        result = step2.bad_ch_by_voltage(raw)
        self.assertEqual(result, {"voltage": ["E0"]})
        self.assertEqual(step2.bad_ch_by_voltage(raw), {"voltage": []})
        self.assertEqual(raw.info["bads"], ["E4", "E0"])
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertEqual(len(raw.annotations), 0)

    def test_local_flat_samples_leave_the_count_and_denominator(self):
        rng = np.random.default_rng(3)
        raw = make_raw(rng.normal(scale=5e-6, size=(8, 100)), sfreq=100)
        raw._data[0, :20] += 300e-6  # 20 exceeding samples...
        raw._data[0, 40:80] = 1e-6  # ...and a 0.4-s flat span on E0.
        flatlines = step2.detect_flatline_inplace(raw.copy(), downstream_highpass_hz=None)
        self.assertEqual(flatlines["segments"], [[[40, 80]]])
        # Counting the flat span as valid, in-range samples dilutes 20/60 to 20/100.
        report, details = step2.bad_ch_by_voltage(raw.copy(), plot=False, return_details=True)
        self.assertEqual(report, {"voltage": []})
        self.assertAlmostEqual(details["exceed_fraction"][0], 0.2)
        report, details = step2.bad_ch_by_voltage(raw.copy(), plot=False, flatlines=flatlines,
                                                  return_details=True)
        self.assertEqual(report, {"voltage": ["E0"]})
        self.assertAlmostEqual(details["exceed_fraction"][0], 20 / 60)
        # A channel flat throughout has no scorable samples: no fraction, no flag.
        flatlines["segments"] = [[[0, 100]]]
        report, details = step2.bad_ch_by_voltage(raw.copy(), plot=False, flatlines=flatlines,
                                                  return_details=True)
        self.assertEqual(report, {"voltage": []})
        self.assertTrue(np.isnan(details["exceed_fraction"][0]))

    def test_temporary_and_existing_annotations_change_sample_denominator(self):
        raw = make_raw(np.ones((4, 100)) * 1e-6, sfreq=100, first_samp=300)
        raw._data[:4, :20] = 600e-6  # Temporary mask.
        raw._data[0, 40:56] = 201e-6  # 16 of 60 retained samples exceed.
        raw._data[1, 40:55] = -201e-6  # 15 of 60: exactly 25%, accepted.
        raw._data[2, 20:40] = 400e-6
        raw.set_annotations(mne.Annotations([0.2], [0.2], ["bad_existing"]))
        annotations = raw.annotations.copy()
        original = raw.get_data().copy()
        self.assertEqual(step2.bad_ch_by_voltage(raw), {"voltage": ["E0"]})
        self.assertEqual(raw.info["bads"], ["E0"])
        self.assertEqual(raw.annotations, annotations)
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_plot_omits_existing_bads_and_non_eeg_and_keeps_peaks(self):
        raw = make_raw(np.ones((4, 1000)) * 1e-6, sfreq=1000)
        raw._data[0] = 300e-6  # Detected by this call: plotted and labeled.
        raw._data[0, 250] = 800e-6  # A masked peak still selects the lower trace.
        raw._data[1] = 5000e-6  # Existing bad: neither plotted nor used for the y-axis.
        raw._data[2, 501] = 150e-6  # A one-sample peak survives the 20-ms envelope.
        raw._data[3] = 2000e-6 * np.sin(2 * np.pi * raw.times)  # ECG: not plotted.
        raw.set_channel_types({"E3": "ecg"})
        raw.info["bads"] = ["E1"]
        with patch.object(step2.plt, "show"):
            self.assertEqual(step2.bad_ch_by_voltage(raw), {"voltage": ["E0"]})
        fig = plt.gcf()
        try:
            self.assertEqual(len(fig.axes), 2)
            ax, lower = fig.axes
            # 50 bins of 20 ms, each drawn as its minimum and maximum.
            traces = [line for line in ax.lines if len(line.get_ydata()) == 2 * 50]
            self.assertEqual(len(traces), 3)  # E0, E2, and Cz.
            # Sample 501 lies in bin 25, whose maximum is point 51: Cz, E2, E0.
            self.assertEqual(sorted(line.get_ydata()[51] for line in traces), [0, 150, 300])
            self.assertTrue(all(line.get_ydata().max() < 1000 for line in traces))
            self.assertEqual(ax.get_ylim(), (-500, 500))
            self.assertIn("min/max envelope in 20-ms bins", ax.get_title())
            self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["E0"])
            self.assertEqual([text.get_text() for text in lower.get_legend().get_texts()], ["E0"])
            np.testing.assert_array_equal(lower.lines[0].get_xdata(), ax.lines[0].get_xdata())
            np.testing.assert_array_equal(lower.lines[0].get_ydata(), ax.lines[0].get_ydata())
            self.assertEqual(lower.lines[0].get_ydata().max(), 800)
            self.assertIn("min/max envelope in 20-ms bins", lower.get_title())
            self.assertEqual(lower.lines[0].get_color().tolist(), ax.lines[0].get_color().tolist())
            self.assertGreater(lower.get_ylim()[1], 800)
            for panel in fig.axes:
                self.assertEqual(len(panel.patches), 1)
                self.assertAlmostEqual(panel.patches[0].get_x(), 0.2)
                self.assertAlmostEqual(panel.patches[0].get_width(), 0.1)
        finally:
            plt.close(fig)


class TestVoltagePlot(unittest.TestCase):
    def test_pooled_absolute_percentile_envelopes_and_input_unchanged(self):
        raw = make_raw(np.vstack([
            np.arange(100), np.r_[-90, np.ones(99)], np.linspace(1, 2, 100),
        ]) * 1e-6, sfreq=100)
        raw.set_annotations(mne.Annotations([0], [0.01], ["BAD_existing"]))
        original = raw.get_data().copy()
        annotations = raw.annotations.copy()
        with patch.object(step2.plt, "show"):
            fig = step2.plot_voltage_traces(raw)
        try:
            upper, lower = fig.axes
            # A separate percentile per channel would also select low-amplitude E2.
            self.assertEqual([text.get_text() for text in lower.get_legend().get_texts()],
                             ["E0", "E1"])
            for row, line in enumerate(lower.lines[:2]):
                np.testing.assert_array_equal(line.get_xdata(), upper.lines[row].get_xdata())
                np.testing.assert_array_equal(line.get_ydata(), upper.lines[row].get_ydata())
            self.assertEqual(upper.get_ylim(), (-500, 500))
            self.assertEqual(upper.get_xlim(), lower.get_xlim())
            np.testing.assert_array_equal(raw.get_data(), original)
            self.assertEqual(raw.info["bads"], [])
            self.assertEqual(raw.annotations, annotations)
        finally:
            plt.close(fig)

    def test_percentile_ties_do_not_select_channels(self):
        raw = make_raw(np.ones((2, 100)) * 10e-6, sfreq=100)
        with patch.object(step2.plt, "show"):
            fig = step2.plot_voltage_traces(raw)
        try:
            lower = fig.axes[1]
            self.assertEqual(len(lower.lines), 2)  # Only the voltage guide lines.
            self.assertIsNone(lower.get_legend())
            self.assertIn("No channels exceed", lower.texts[0].get_text())
        finally:
            plt.close(fig)

    def test_final_partial_bin_keeps_last_sample_peak(self):
        raw = make_raw(np.vstack([np.ones(1003), np.full(1003, 2)]) * 1e-6,
                       sfreq=1000)
        raw._data[0, -1] = -400e-6
        with patch.object(step2.plt, "show"):
            fig = step2.plot_voltage_traces(raw)
        try:
            upper, lower = fig.axes
            self.assertEqual(len(upper.lines[0].get_ydata()), 102)
            np.testing.assert_array_equal(upper.lines[0].get_ydata()[-2:], [-400, 1])
            self.assertEqual([text.get_text() for text in lower.get_legend().get_texts()],
                             ["E0"])
            self.assertEqual(len(lower.lines[0].get_ydata()), 102)
            np.testing.assert_array_equal(lower.lines[0].get_ydata()[-2:], [-400, 1])
        finally:
            plt.close(fig)


class TestRobustMetrics(unittest.TestCase):
    def test_hf_rank_preserves_candidates_at_or_above_median(self):
        for hf_uv, n_flagged, expected in (
            (1.9, 2, []),
            (2.0, 2, ["E8"]),
            (2.1, 2, ["E8"]),
            (1.9, 4, ["E8"]),  # No non-violating baseline: retain the candidate.
        ):
            with self.subTest(hf_uv=hf_uv, n_flagged=n_flagged):
                raw = hf_raw()
                metrics = []
                for window in range(4):
                    hf = np.r_[np.full(8, 2.0), hf_uv] * 1e-6
                    lf = hf / np.linspace(0.1, 0.3, 9)
                    lf[-1] = hf[-1] / (10 if window < n_flagged else 0.2)
                    metrics.append((hf / lf, hf, lf))
                with patch.object(step2, "_prep_hf_ratio", side_effect=metrics):
                    report, details = step2.bad_ch_by_HFnoise(raw, return_details=True)
                np.testing.assert_array_equal(
                    details["violates"][-1], np.arange(4) < n_flagged
                )
                self.assertEqual(report, {"HFnoise": expected})
                self.assertEqual(raw.info["bads"], expected)

    def test_hf_rank_uses_only_candidate_violating_windows(self):
        metrics = []
        for window in range(4):
            hf = np.r_[np.full(8, 2.0 if window < 2 else 200.0), 3.0] * 1e-6
            lf = hf / np.linspace(0.1, 0.3, 9)
            lf[-1] = hf[-1] / (10 if window < 2 else 0.2)
            metrics.append((hf / lf, hf, lf))
        with patch.object(step2, "_prep_hf_ratio", side_effect=metrics):
            report, details = step2.bad_ch_by_HFnoise(hf_raw(), return_details=True)
        np.testing.assert_array_equal(details["violates"][-1], [1, 1, 0, 0])
        self.assertEqual(report, {"HFnoise": ["E8"]})

    def test_hf_rank_compares_channel_medians(self):
        metrics = []
        for window in range(6):
            # Over the candidate's three violating windows, five channels have
            # median HF=1; pooling channel/window values would give median HF=100.
            hf = np.r_[np.full(5, 1.0 if window < 2 else 100.0),
                       np.full(3, 100.0), 3.0] * 1e-6
            lf = hf / np.linspace(0.1, 0.3, 9)
            lf[-1] = hf[-1] / (10 if window < 3 else 0.2)
            metrics.append((hf / lf, hf, lf))
        with patch.object(step2, "_prep_hf_ratio", side_effect=metrics):
            report, details = step2.bad_ch_by_HFnoise(
                hf_raw(window_length=3), return_details=True
            )
        np.testing.assert_array_equal(details["violates"][-1], [1, 1, 1, 0, 0, 0])
        self.assertEqual(report, {"HFnoise": ["E8"]})

    def test_hf_confirmation_compares_log_hf_increase_with_log_lf_decrease(self):
        for hf_factor, lf_factor, expected in (
            (0.9, 0.5, []),
            (1.0, 0.5, []),
            (1.1, 1.08, ["E8"]),
            (1.1, 0.92, ["E8"]),
            (1.1, 0.89, []),
            (1.1, 0.9, []),
            (1.125, 0.875, []),
            (1.125001, 0.875, []),
            (1.1, 1.2, ["E8"]),
            (1.1, 1 / 1.1, []),
            (1.100001, 1 / 1.1, ["E8"]),
            (2.0, 0.5, []),  # Equal multiplicative changes are a tie.
            (2.000001, 0.5, ["E8"]),
            (1 + 1e-14, 1.0, []),  # A rounding-level HF rise with no LF change is a tie.
            (1.000001, 1.0, ["E8"]),
            (1.1, 0.0, []),
        ):
            with self.subTest(hf_factor=hf_factor, lf_factor=lf_factor):
                raw = hf_raw()
                metrics = []
                for window in range(4):
                    hf = np.r_[np.linspace(1, 3, 8), 2.0] * 1e-6
                    lf = np.full(9, 10e-6)
                    # Keep the candidate below the HF median so confirmation applies.
                    hf[:8] *= 10
                    lf[:8] *= 10
                    if window < 2:
                        # LF elsewhere shifts the cross-channel ratio baseline.
                        lf[:8] *= 10
                        hf[-1] *= hf_factor
                        lf[-1] *= lf_factor
                    metrics.append((hf / np.maximum(lf, np.finfo(float).eps), hf, lf))
                with patch.object(step2, "_prep_hf_ratio", side_effect=metrics):
                    report, details = step2.bad_ch_by_HFnoise(raw, return_details=True)
                np.testing.assert_array_equal(details["violates"][-1], [1, 1, 0, 0])
                self.assertEqual(report, {"HFnoise": expected})
                self.assertEqual(raw.info["bads"], expected)
                with patch.object(step2, "_prep_hf_ratio", side_effect=metrics):
                    self.assertEqual(step2.bad_ch_by_HFnoise(hf_raw()), report)

    def test_hf_confirmation_keeps_candidates_without_passing_windows(self):
        for masked_baseline in (False, True):
            with self.subTest(masked_baseline=masked_baseline):
                raw = hf_raw(n_bad=2 if masked_baseline else 4)
                if masked_baseline:
                    raw.set_annotations(mne.Annotations([4], [4], ["BAD_baseline"]))
                report, details = step2.bad_ch_by_HFnoise(raw, return_details=True)
                self.assertEqual(report, {"HFnoise": ["E8"]})
                flags = details["violates"][-1, details["retained"]]
                np.testing.assert_array_equal(flags, np.ones(2 if masked_baseline else 4))

    def test_hf_confirmation_rejects_lf_driven_ratio_in_filtered_signal(self):
        raw = hf_raw(n_bad=0)
        raw._data[8] -= 0.7e-6 * np.sin(2 * np.pi * 80 * raw.times)
        t = np.arange(1000) / raw.info["sfreq"]
        raw._data[8, :1000] -= 9.8e-6 * np.sin(2 * np.pi * 10 * t)
        report, details = step2.bad_ch_by_HFnoise(raw, return_details=True)
        np.testing.assert_array_equal(details["violates"][8], [1, 1, 0, 0])
        self.assertEqual(report, {"HFnoise": []})
        self.assertEqual(raw.info["bads"], [])

    def test_hf_confirmation_uses_medians_not_means(self):
        raw = hf_raw(window_length=3)  # Six 2-second windows.
        metrics = []
        for window in range(6):
            hf = np.r_[np.linspace(1, 3, 8), 2.0] * 1e-6
            lf = np.full(9, 10e-6)
            if window < 3:
                lf[:8] *= 10
                hf[-1] = (20 if window == 0 else 1.8) * 1e-6
            metrics.append((hf / lf, hf, lf))
        with patch.object(step2, "_prep_hf_ratio", side_effect=metrics):
            report, details = step2.bad_ch_by_HFnoise(raw, return_details=True)
        np.testing.assert_array_equal(details["violates"][-1], [1, 1, 1, 0, 0, 0])
        self.assertEqual(report, {"HFnoise": []})

    def test_hf_details_preserve_masked_times_and_extract_flagged_waveforms(self):
        raw = hf_raw(n_bad=2)
        raw.info["bads"] = ["E1"]
        raw.set_annotations(mne.Annotations([2.25], [0.01], ["BAD_existing"]))
        raw._data[:4, 1000:1005] = 600e-6  # Temporary mask in the third window.
        original = raw.get_data().copy()
        annotations = raw.annotations.copy()
        expected = step2.bad_ch_by_HFnoise(raw.copy())
        report, details = step2.bad_ch_by_HFnoise(raw, return_details=True)
        self.assertEqual(report, expected)
        self.assertEqual(report, {"HFnoise": ["E8"]})
        self.assertEqual(raw.info["bads"], ["E1", "E8"])
        self.assertNotIn("E1", details["ch_names"])
        self.assertNotIn("Cz", details["ch_names"])
        np.testing.assert_array_equal(details["start_s"], [0, 2, 4, 6])
        np.testing.assert_array_equal(details["stop_s"], [2, 4, 6, 8])
        np.testing.assert_array_equal(details["retained"], [True, False, False, True])
        i = details["ch_names"].index("E8")
        np.testing.assert_array_equal(details["violates"][i], [1, np.nan, np.nan, 0])
        for name in details["ch_names"]:
            channel = details["ch_names"].index(name)
            windows = np.flatnonzero(details["violates"][channel] == 1)
            extracted = details["flagged_data_uv"][name]
            self.assertEqual(extracted.shape, (len(windows), 500))
            for waveform, window in zip(extracted, windows):
                np.testing.assert_allclose(
                    waveform, original[raw.ch_names.index(name), window * 500:(window + 1) * 500] * 1e6
                )
        self.assertTrue(np.isnan(details["robust_z"][:, [1, 2]]).all())
        np.testing.assert_allclose(
            details["ratio"][:, details["retained"]],
            (details["hf_mad_uv"] / details["lf_mad_uv"])[:, details["retained"]],
        )
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertEqual(raw.annotations, annotations)

    def test_matlab_iqr_and_distinct_cross_channel_scales(self):
        self.assertEqual(step2._iqr(np.arange(1, 9)), 4)
        values = np.array([0, 1, 2, 3, 7, 8])
        np.testing.assert_allclose(
            step2._robust_channel_z(values, use_iqr=True), (values - 2.5) / (0.7413 * 6)
        )
        np.testing.assert_allclose(
            step2._robust_channel_z(values), (values - 2.5) / (1.4826 * 2)
        )
        np.testing.assert_array_equal(step2._robust_channel_z(np.ones(8)), np.zeros(8))
        np.testing.assert_array_equal(
            step2._robust_channel_z(np.ones(8), use_iqr=True), np.zeros(8)
        )

    def test_std_and_hf_use_strict_positive_z_threshold(self):
        for detector, factory, _ in WINDOW_DETECTORS[:2]:
            with self.subTest(detector=detector.__name__):
                raw = factory()
                z = np.zeros(18 if detector is step2.bad_ch_by_std else 9)
                z[:3] = [-10, 5, 5.001]
                criterion = detector.__name__.removeprefix("bad_ch_by_")
                with patch.object(step2, "_robust_channel_z", return_value=z):
                    self.assertEqual(detector(raw), {criterion: ["E2"]})
                self.assertEqual(raw.info["bads"], ["E2"])

    def test_std_flags_only_by_occurrence_above_its_z_cutoff(self):
        raw = std_raw(n_bad=0)
        z = np.zeros(18)
        z[3:5] = [2.5, 4.999]  # Moderately high in every window, but never above 5.
        with patch.object(step2, "_robust_channel_z", return_value=z):
            report, details = step2.bad_ch_by_std(raw, return_details=True)
        self.assertEqual(report, {"std": []})
        self.assertNotIn("median_robust_z", details)
        with self.assertRaises(TypeError):
            step2.bad_ch_by_std(std_raw(), median_z_cutoff=2.0)

    def test_hf_ratio_tracks_hf_relative_to_lf_and_ignores_dc_offset(self):
        sfreq = 1000
        t = np.arange(2 * sfreq) / sfreq
        low = 10e-6 * np.sin(2 * np.pi * 10 * t)
        high = 2e-6 * np.sin(2 * np.pi * 100 * t)
        ratios = hf_ratio(np.array([low + high, low + 5 * high, 5 * low + high]), sfreq)
        self.assertGreater(ratios[1], 4 * ratios[0])
        self.assertLess(ratios[2], ratios[0] / 4)
        shifted = hf_ratio(np.array([low + high + 100e-6]), sfreq)
        np.testing.assert_allclose(shifted, ratios[:1], rtol=1e-8)
        np.testing.assert_array_equal(hf_ratio(np.zeros((2, len(t))), sfreq), [0, 0])

    def test_hf_ratio_band_edges_stay_at_45_and_50_hz_for_any_sampling_rate(self):
        for sfreq in (250, 1000, 2000):
            with self.subTest(sfreq=sfreq):
                t = np.arange(2 * sfreq) / sfreq
                passband, stopband = np.sin(2 * np.pi * 44 * t), np.sin(2 * np.pi * 51 * t)
                ratios = hf_ratio(np.array([passband, stopband]), sfreq)
                self.assertLess(ratios[0], 0.01)
                self.assertGreater(ratios[1], 100)
        with self.assertRaisesRegex(ValueError, "longer than"):
            hf_ratio(np.zeros((2, 600)), 1000)  # Filter has 661 taps at 1000 Hz.

    def test_hf_lowpass_spans_the_record_and_widens_the_mask_by_its_reach(self):
        raw = hf_raw(n_bad=0)
        raw._data[:4, 500:505] = 600e-6  # Masks samples 500-524 of the second window.
        _, details = step2.bad_ch_by_HFnoise(raw, return_details=True)
        # The 164-sample (0.66-s) reach also removes the first window.
        np.testing.assert_array_equal(details["retained"], [False, False, True, True])
        data = raw.get_data(picks=range(9))
        residual = data - step2._lowpass_45_hz(data, raw.info["sfreq"])
        np.testing.assert_allclose(
            details["hf_mad_uv"][:, 2], step2._mad(residual[:, 1000:1500], axis=1) * 1e6
        )


class TestWindowedNoiseDetection(unittest.TestCase):
    def test_report_keeps_separate_criteria_while_bads_accumulate(self):
        raw = std_raw()
        cz = mne.create_info(["Cz"], raw.info["sfreq"], "eeg")
        raw.add_channels([mne.io.RawArray(np.zeros((1, raw.n_times)), cz, verbose=False)])
        raw._data[1, 5] = np.nan
        report = {}
        report.update(step2.bad_ch_by_nan(raw))  # Needs the native reference, Cz.
        raw.set_eeg_reference("average", projection=False, verbose=False)
        report.update(step2.bad_ch_by_std(raw))
        self.assertEqual(report, {"nan": ["E1"], "std": ["E8", "M8"]})
        self.assertEqual(raw.info["bads"], ["E1", "E8", "M8"])
        # Report lists are independent of the accumulated labels on Raw.
        report["std"].append("E2")
        self.assertEqual(raw.info["bads"], ["E1", "E8", "M8"])

    def test_actual_metrics_strict_occurrence_and_multiple_bb_outliers(self):
        for detector, factory, expected in WINDOW_DETECTORS:
            for n_bad in (1, 2):
                with self.subTest(detector=detector.__name__, n_bad=n_bad):
                    raw = factory(n_bad)
                    original = raw.get_data().copy()
                    criterion = detector.__name__.removeprefix("bad_ch_by_")
                    self.assertEqual(detector(raw), {criterion: expected if n_bad == 2 else []})
                    self.assertEqual(raw.info["bads"], expected if n_bad == 2 else [])
                    np.testing.assert_array_equal(raw.get_data(), original)
                    self.assertEqual(len(raw.annotations), 0)

    def test_any_mask_overlap_removes_whole_window_from_denominator(self):
        for detector, factory, expected in WINDOW_DETECTORS:
            with self.subTest(detector=detector.__name__):
                raw = factory(n_bad=1)
                # A burst starting exactly at the second window masks only that
                # window; HFnoise widens the mask by its 0.66-s low-pass reach, so
                # its burst starts mid-window.
                start = 700 if detector is step2.bad_ch_by_HFnoise else 500
                raw._data[:4, start:start + 5] = 600e-6
                if detector is not step2.bad_ch_by_HFnoise:
                    mirrors = [raw.ch_names.index(f"M{i}") for i in range(4)]
                    raw._data[mirrors, start:start + 5] = -600e-6
                raw.set_annotations(mne.Annotations([7], [0], ["event"]))
                original = raw.get_data().copy()
                annotations = raw.annotations.copy()
                criterion = detector.__name__.removeprefix("bad_ch_by_")
                # One bad window / three retained, not one / four.
                self.assertEqual(detector(raw), {criterion: expected})
                self.assertEqual(raw.info["bads"], expected)
                np.testing.assert_array_equal(raw.get_data(), original)
                self.assertEqual(raw.annotations, annotations)

    def test_existing_bad_annotation_omits_offending_window(self):
        for detector, factory, _ in WINDOW_DETECTORS:
            with self.subTest(detector=detector.__name__):
                raw = factory(n_bad=1)
                raw.set_annotations(mne.Annotations([0.5], [0.01], ["BAD_previous"]))
                annotations = raw.annotations.copy()
                criterion = detector.__name__.removeprefix("bad_ch_by_")
                self.assertEqual(detector(raw), {criterion: []})
                self.assertEqual(raw.info["bads"], [])
                self.assertEqual(raw.annotations, annotations)

    def test_custom_window_and_occurrence_cutoff(self):
        for detector, factory, expected in WINDOW_DETECTORS:
            for cutoff in (0.25, 0.5):
                with self.subTest(detector=detector.__name__, cutoff=cutoff):
                    raw = factory(window_length=1)
                    criterion = detector.__name__.removeprefix("bad_ch_by_")
                    # Two of four windows are bad.
                    self.assertEqual(
                        detector(raw, window_length=1, occurrence_cutoff=cutoff),
                        {criterion: expected if cutoff == 0.25 else []},
                    )
                    self.assertEqual(raw.info["bads"], expected if cutoff == 0.25 else [])

    def test_existing_bads_and_non_eeg_excluded_and_labels_preserved(self):
        for detector, factory, expected in WINDOW_DETECTORS:
            with self.subTest(detector=detector.__name__):
                raw = factory()
                raw.info["bads"] = ["E1"]
                raw._data[1] = np.nan
                raw.set_channel_types({"E2": "eog"})
                raw._data[2] = np.inf
                if detector is not step2.bad_ch_by_HFnoise:
                    raw.set_eeg_reference("average", projection=False, verbose=False)
                criterion = detector.__name__.removeprefix("bad_ch_by_")
                self.assertEqual(detector(raw), {criterion: expected})
                self.assertEqual(raw.info["bads"], ["E1"] + expected)
                self.assertEqual(len(raw.annotations), 0)

    def test_bb_band_requirement_is_configurable(self):
        raw = bb_raw()
        result = step2.bad_ch_by_BBnoise(raw, min_bad_bands=1)
        self.assertIn("E0", result["BBnoise"])
        self.assertEqual(raw.info["bads"], result["BBnoise"])

    def test_fully_masked_recording_raises_without_input_changes(self):
        for detector in (step2.bad_ch_by_voltage,) + tuple(x[0] for x in WINDOW_DETECTORS):
            with self.subTest(detector=detector.__name__):
                average = detector in (step2.bad_ch_by_std, step2.bad_ch_by_BBnoise)
                raw = make_raw(np.full((5, 1000), 600e-6), average=average)
                raw.info["bads"] = ["E0"]
                if average:
                    raw.set_eeg_reference("average", projection=False, verbose=False)
                original = raw.get_data().copy()
                with self.assertRaisesRegex(ValueError, "No .*unmasked"):
                    detector(raw)
                self.assertEqual(raw.info["bads"], ["E0"])
                self.assertEqual(len(raw.annotations), 0)
                np.testing.assert_array_equal(raw.get_data(), original)

    def test_invalid_parameters(self):
        for detector, factory, _ in WINDOW_DETECTORS:
            for kwargs in (
                {"robust_z_cutoff": 0}, {"robust_z_cutoff": np.nan},
                {"occurrence_cutoff": -0.1}, {"occurrence_cutoff": 1.1},
                {"window_length": 0}, {"window_length": 20},
            ):
                with self.subTest(detector=detector.__name__, kwargs=kwargs):
                    raw = factory()
                    with self.assertRaises(ValueError):
                        detector(raw, **kwargs)
                    self.assertEqual(raw.info["bads"], [])
                    self.assertEqual(len(raw.annotations), 0)
        with self.assertRaisesRegex(ValueError, "above 100"):
            step2.bad_ch_by_HFnoise(std_raw(sfreq=100))
        for bands in (0, 6, 3.5):
            with self.subTest(bands=bands), self.assertRaisesRegex(ValueError, "min_bad_bands"):
                step2.bad_ch_by_BBnoise(bb_raw(), min_bad_bands=bands)


if __name__ == "__main__":
    unittest.main()
