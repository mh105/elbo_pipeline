import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


SFREQ = 250
PATTERN = np.linspace(1.0, 2.0, 8)  # Spatial pattern of the shared artifact.
UNIT = PATTERN / np.linalg.norm(PATTERN)


def pca_raw(data):
    """8 EEG channels plus a zero Cz reference."""
    names = [f"E{i}" for i in range(8)] + ["Cz"]
    data = np.vstack([data, np.zeros(data.shape[1])])
    return mne.io.RawArray(data, mne.create_info(names, SFREQ, "eeg"), verbose=False)


def with_positions(raw):
    """Place pca_raw's sensors on a 9-cm sphere; E7 is nearest the vertex after Cz."""
    # Unit directions: +x right, +y anterior, +z superior.
    directions = {
        "E0": [0, 0.8, 0.6], "E1": [-0.6, 0.6, 0.53], "E2": [0.2, 0.1, 0.97],
        "E3": [-0.8, 0, 0.6], "E4": [0.8, 0, 0.6], "E5": [0, -0.8, 0.6],
        "E6": [0.6, -0.6, 0.53], "E7": [0, -0.05, 1], "Cz": [0, 0, 1],
    }
    return raw.set_montage(mne.channels.make_dig_montage(
        {name: 0.09 * np.array(d) / np.linalg.norm(d) for name, d in directions.items()},
        coord_frame="head",
    ))


def noise(seconds=10.5, scale=5e-6, seed=0):
    return np.random.default_rng(seed).normal(0, scale, (8, int(seconds * SFREQ)))


def burst(n_times, windows, amplitude=300e-6):
    """Shared 8-Hz bursts, one Hann-shaped burst filling each listed 1-s window."""
    t = np.arange(n_times) / SFREQ
    course = np.zeros(n_times)
    for window in windows:
        inside = (t >= window) & (t < window + 1)
        course[inside] = amplitude * np.sin(2 * np.pi * 8 * t[inside]) * np.hanning(inside.sum())
    return PATTERN[:, None] * course


def removal_weight(pca_details, run, loading_window, data):
    """Removal weight w(t) over a run's span, from removal = w(t) u (u^T x)."""
    start, stop, removal = pca_details["removed_spans"][run]
    u = pca_details["input"]["pc1_loadings"][loading_window]
    return (u @ removal) / (u @ data[:, start:stop]), start


class TestPCACleaning(unittest.TestCase):
    def test_removes_shared_burst_only_from_non_bad_eeg_channels(self):
        background = noise()
        data = background + burst(background.shape[1], [4])
        raw = pca_raw(np.vstack([data]))
        raw.info["bads"] = ["E7"]
        raw.set_annotations(mne.Annotations([1.0], [0.5], ["BAD_existing"]))
        original = raw.get_data().copy()
        details = step2.pca_cleaning(raw)

        self.assertEqual(details["ch_names"], [f"E{i}" for i in range(7)])
        np.testing.assert_array_equal(details["cleaned"], np.arange(10) == 4)
        self.assertEqual(list(details["removed_pass"]), [""] * 4 + ["input"] + [""] * 5)
        for name in ("input", "highpass"):  # Both passes qualify; input takes priority.
            self.assertTrue(details[name]["qualified"][4])
            self.assertGreater(details[name]["pc1_variance_fraction"][4], 0.99)
            self.assertGreater(details[name]["pc1_peak_uv"][4], 100)
        loadings = details["input"]["pc1_loadings"]
        np.testing.assert_allclose(np.linalg.norm(loadings, axis=1), 1)
        self.assertGreater(abs(loadings[4] @ UNIT[:7]) / np.linalg.norm(UNIT[:7]), 0.999)
        # The trailing half second joins the last window.
        np.testing.assert_allclose(details["start_s"], np.arange(10))
        np.testing.assert_allclose(details["stop_s"], [*range(1, 10), 10.5])

        cleaned = raw.get_data()
        window = slice(4 * SFREQ, 5 * SFREQ)
        self.assertLess(np.abs(cleaned[:7, window] - background[:7, window]).max(), 20e-6)
        span = slice(int(3.75 * SFREQ), int(5.25 * SFREQ))  # Cleaned window plus fades.
        unchanged = np.ones(raw.n_times, dtype=bool)
        unchanged[span] = False
        np.testing.assert_array_equal(cleaned[:, unchanged], original[:, unchanged])
        np.testing.assert_array_equal(cleaned[7:], original[7:])  # Bad channel and Cz.
        self.assertEqual(raw.info["bads"], ["E7"])
        self.assertEqual(list(raw.annotations.description), ["BAD_existing"])

    def test_amplitude_floor_and_inclusive_variance_cutoff(self):
        background = noise(scale=1e-6)
        data = background + burst(background.shape[1], [4], amplitude=30e-6)
        details = step2.pca_cleaning(pca_raw(data))  # Peak ~60 uV: flagged, not qualified.
        self.assertTrue(details["input"]["flagged"][4])
        self.assertFalse(details["cleaned"].any())
        details = step2.pca_cleaning(pca_raw(data), amplitude_floor_uv=20.0)
        self.assertTrue(details["cleaned"][4])

        fraction = details["input"]["pc1_variance_fraction"][4]
        for cutoff, flagged in ((fraction, True), (np.nextafter(fraction, 1), False)):
            with self.subTest(cutoff=cutoff):
                details = step2.pca_cleaning(
                    pca_raw(data), variance_cutoff=cutoff, amplitude_floor_uv=20.0
                )
                self.assertEqual(details["input"]["flagged"][4], flagged)

    def test_highpass_pass_removes_pc1_from_highpassed_data_only(self):
        background = noise(scale=100e-6)
        spike = burst(background.shape[1], [2], amplitude=150e-6)
        data = background + spike
        highpassed = spike + noise(scale=1e-9, seed=1)  # Only the spike survives the filter.
        with patch.object(step2.mne.filter, "filter_data", return_value=highpassed):
            raw = pca_raw(data)
            details = step2.pca_cleaning(raw, return_removed=True)
        self.assertLess(details["input"]["pc1_variance_fraction"][2], 0.9)
        self.assertTrue(details["highpass"]["qualified"][2])
        self.assertEqual(details["removed_pass"][2], "highpass")
        u = details["highpass"]["pc1_loadings"][2]
        self.assertGreater(abs(u @ UNIT), 0.999)
        start, stop, removal = details["removed_spans"][0]
        window = slice(2 * SFREQ - start, 3 * SFREQ - start)
        np.testing.assert_allclose(
            removal[:, window], np.outer(u, u) @ highpassed[:, 2 * SFREQ:3 * SFREQ], atol=1e-15
        )
        # The background the filter rejected is kept, including its part along u.
        np.testing.assert_allclose(raw.get_data()[:8, 2 * SFREQ:3 * SFREQ],
                                   background[:, 2 * SFREQ:3 * SFREQ], rtol=0, atol=1e-7)

    def test_fades_and_hand_overs_keep_total_removal_weight_at_most_one(self):
        background = noise(scale=1e-6)
        n_times = background.shape[1]
        offset = PATTERN[:, None] * np.full(n_times, 50e-6)  # Shared level, never flagged.
        data = background + offset + burst(n_times, [4, 5, 8])
        details = step2.pca_cleaning(pca_raw(data), return_removed=True)
        np.testing.assert_array_equal(details["cleaned"], np.isin(np.arange(10), [4, 5, 8]))
        self.assertEqual(len(details["removed_spans"]), 2)
        fade = int(0.25 * SFREQ)

        weight, start = removal_weight(details, 1, 8, data)  # Single-window run.
        self.assertEqual((start, len(weight)), (8 * SFREQ - fade, SFREQ + 2 * fade))
        rise, core, fall = weight[:fade], weight[fade:-fade], weight[-fade:]
        np.testing.assert_allclose(core, 1, atol=1e-12)
        self.assertTrue((np.diff(rise) > 0).all() and (np.diff(fall) < 0).all())
        self.assertTrue(0 < rise[0] < 0.01 and 0 < fall[-1] < 0.01)
        np.testing.assert_allclose(rise, fall[::-1], atol=1e-12)  # Mirror-image fades.

        weight, start = removal_weight(details, 0, 4, data)  # Windows 4 and 5.
        hand_over = slice(5 * SFREQ - fade - start, 5 * SFREQ + fade - start)
        # Both windows' PC1s match the pattern, so their hand-over weights must sum to one.
        np.testing.assert_allclose(weight[fade:hand_over.start], 1, atol=1e-12)
        np.testing.assert_allclose(weight[hand_over], 1, atol=1e-3)
        self.assertLessEqual(weight.max(), 1 + 1e-3)

    def test_remove_pca_signal_inplace_reproduces_cleaning_and_checks_inputs(self):
        data = noise() + burst(int(10.5 * SFREQ), [3, 7])
        raw = pca_raw(data)
        raw.info["bads"] = ["E7"]
        copy = raw.copy()
        details = step2.pca_cleaning(raw, return_removed=True)
        self.assertIsNone(step2.remove_pca_signal_inplace(copy, details))
        np.testing.assert_allclose(copy.get_data(), raw.get_data(), rtol=0, atol=1e-18)
        np.testing.assert_array_equal(copy.get_data(picks=["E7", "Cz"]),
                                      pca_raw(data).get_data(picks=["E7", "Cz"]))

        with self.assertRaisesRegex(ValueError, "return_removed"):
            step2.remove_pca_signal_inplace(pca_raw(data),
                                            step2.pca_cleaning(pca_raw(data)))
        with self.assertRaisesRegex(ValueError, "same samples"):
            step2.remove_pca_signal_inplace(pca_raw(data).crop(tmax=9.0), details)

    def test_invalid_inputs(self):
        data = noise()
        for kwargs in (
            {"variance_cutoff": 0}, {"variance_cutoff": 1}, {"variance_cutoff": np.nan},
            {"fade_length": -0.1}, {"fade_length": 0.6}, {"highpass_freq": 0},
            {"highpass_freq": SFREQ / 2}, {"amplitude_floor_uv": -1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                step2.pca_cleaning(pca_raw(data), **kwargs)
        average = pca_raw(data).set_eeg_reference("average", projection=False, verbose=False)
        with self.assertRaisesRegex(ValueError, "entirely zero EEG reference"):
            step2.pca_cleaning(average)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test_raw.fif"
            pca_raw(data).save(path, verbose=False)
            with self.assertRaisesRegex(ValueError, "preloaded"):
                step2.pca_cleaning(mne.io.read_raw_fif(path, preload=False, verbose=False))

    def test_plot_pca_cleaning_shows_frontal_middle_posterior_and_one_bad_channel(self):
        raw = with_positions(pca_raw(noise() + burst(int(10.5 * SFREQ), [4])))
        for bads, expected in (
            (["E7", "E3"], ["E0 (frontal)", "E2 (middle)", "E5 (posterior)",
                            "E7 (bad, not cleaned)"]),
            ([], ["E0 (frontal)", "E7 (middle)", "E5 (posterior)"]),
        ):
            with self.subTest(bads=bads):
                raw.info["bads"] = bads
                cleaned = raw.copy()
                details = step2.pca_cleaning(cleaned)
                with patch.object(step2.plt, "show"):
                    overview, traces = step2.plot_pca_cleaning(raw, cleaned, details)
                try:
                    self.assertEqual(len(overview.axes), 2)
                    self.assertEqual([ax.get_ylabel().split("\n")[0] for ax in traces.axes],
                                     expected)
                finally:
                    plt.close(overview)
                    plt.close(traces)

    def test_plot_pca_cleaning_draws_finite_samples_and_flags_nonfinite_spans(self):
        data = noise() + burst(int(10.5 * SFREQ), [4])
        data[7, 2 * SFREQ:3 * SFREQ] = np.nan
        data[7, 6 * SFREQ] = np.inf
        raw = with_positions(pca_raw(data))
        raw.info["bads"] = ["E7"]
        cleaned = raw.copy()
        details = step2.pca_cleaning(cleaned)
        with patch.object(step2.plt, "show"):
            overview, traces = step2.plot_pca_cleaning(raw, cleaned, details)
        try:
            *good_axes, bad_ax = traces.axes
            self.assertTrue(all(ax.get_legend() is None for ax in good_axes))
            self.assertEqual([text.get_text() for text in bad_ax.get_legend().get_texts()],
                             ["NaN/inf samples (not drawn)"])
            gray = [span for span in bad_ax.patches if span.get_facecolor()[:3] == (0.6, 0.6, 0.6)]
            np.testing.assert_allclose([span.get_x() for span in gray], [2.0, 6.0])
            np.testing.assert_allclose([span.get_width() for span in gray], [1.0, 1 / SFREQ])
            drawn = bad_ax.lines[0].get_ydata()
            self.assertEqual(np.isfinite(drawn).sum(), data.shape[1] - SFREQ - 1)
        finally:
            plt.close(overview)
            plt.close(traces)

    def test_plot_pca_cleaning_requires_sensor_positions(self):
        raw = pca_raw(noise())
        cleaned = raw.copy()
        details = step2.pca_cleaning(cleaned)
        with self.assertRaisesRegex(ValueError, "sensor positions"):
            step2.plot_pca_cleaning(raw, cleaned, details)


if __name__ == "__main__":
    unittest.main()
