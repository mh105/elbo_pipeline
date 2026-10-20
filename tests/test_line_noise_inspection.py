"""Check automatic line-noise inspection against known spectral changes."""

from contextlib import redirect_stdout
import io
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step1_line_noise_removal as step1


class TestLineNoiseInspection(unittest.TestCase):
    def setUp(self):
        sfreq = 400.0
        times = np.arange(4000) / sfreq
        amplitudes = np.array([6, 10, 4, 500, 0])[:, None] * 1e-6
        line_amplitudes = np.array([20, 30, 50, 1000, 0])[:, None] * 1e-6
        self.line = line_amplitudes * np.sin(2 * np.pi * 30 * times)
        data = (amplitudes * np.sin(2 * np.pi * 10 * times)
                + self.line
                + 2e-6 * np.sin(2 * np.pi * 80 * times)
                + 20e-6 * np.sin(2 * np.pi * 0.2 * times))
        self.raw = mne.io.RawArray(
            data, mne.create_info(
                ["E75", "E1", "E2", "Cz", "Trigger"], sfreq,
                ["eeg"] * 4 + ["misc"],
            ), first_samp=1200, verbose=False,
        )
        self.raw.info["description"] = "line-noise inspection fixture"
        self.raw.info["bads"] = ["Trigger"]
        self.raw.set_annotations(mne.Annotations([2], [0.25], ["review_marker"]))
        with mne.use_log_level("ERROR"):
            self.raw_prep = self.raw.copy().filter(1.0, None, verbose=False)
        self.clean = self.raw_prep.copy()
        self.clean._data[:3] -= 0.9 * self.line[:3]
        self.addCleanup(plt.close, "all")

    def assert_raw_unchanged(self, actual, original):
        np.testing.assert_array_equal(actual.get_data(), original.get_data())
        self.assert_metadata_unchanged(actual, original)

    def assert_metadata_unchanged(self, actual, original):
        self.assertEqual(actual.ch_names, original.ch_names)
        self.assertEqual(actual.first_samp, original.first_samp)
        for key in ("sfreq", "highpass", "lowpass", "bads", "description"):
            self.assertEqual(actual.info[key], original.info[key])
        np.testing.assert_array_equal(actual.annotations.onset,
                                      original.annotations.onset)
        np.testing.assert_array_equal(actual.annotations.duration,
                                      original.annotations.duration)
        np.testing.assert_array_equal(actual.annotations.description,
                                      original.annotations.description)

    def test_automatic_inspection_receives_already_cleaned_raw(self):
        original = self.raw.copy()
        prep_before, clean_before = self.raw_prep.copy(), self.clean.copy()
        expected = (original.get_data()
                    - (prep_before.get_data() - clean_before.get_data()))

        def inspect(raw, raw_prep):
            self.assertIs(raw, self.raw)
            self.assertIs(raw_prep, self.raw_prep)
            np.testing.assert_array_equal(raw.get_data(), expected)

        with patch.object(step1, "plot_line_noise_removal", side_effect=inspect) as plot:
            result = step1.remove_line_noise_inplace(
                self.raw, self.raw_prep, self.clean
            )

        self.assertIsNone(result)
        plot.assert_called_once_with(self.raw, self.raw_prep)
        np.testing.assert_array_equal(self.raw.get_data(), expected)
        self.assert_metadata_unchanged(self.raw, original)
        self.assert_raw_unchanged(self.raw_prep, prep_before)
        self.assert_raw_unchanged(self.clean, clean_before)

    def test_figures_preserve_inputs_and_show_known_attenuation(self):
        self.raw._data -= self.raw_prep.get_data() - self.clean.get_data()
        original, prep_before = self.raw.copy(), self.raw_prep.copy()
        with mne.use_log_level("ERROR"):
            filtered_after = original.copy().filter(1.0, None, verbose=False)
            before_psd = prep_before.compute_psd(fmax=150, verbose=False)
            after_psd = filtered_after.compute_psd(fmax=150, verbose=False)
        before, frequencies = before_psd.get_data(
            picks="data", exclude=["Cz"], return_freqs=True
        )
        after = after_psd.get_data(picks="data", exclude=["Cz"])
        spectrogram_calls = []
        multitaper = step1.multitaper_spectrogram

        def capture_spectrogram(*args, **kwargs):
            result = multitaper(*args, **kwargs)
            spectrogram_calls.append((args, kwargs, result))
            return result

        with patch.object(plt, "show"), mne.use_log_level("ERROR"), \
                patch.object(step1, "multitaper_spectrogram",
                             side_effect=capture_spectrogram):
            figures = step1.plot_line_noise_removal(self.raw, self.raw_prep)

        self.assertEqual(len(figures), 8)
        self.assertTrue(all(isinstance(figure, Figure) for figure in figures))
        self.assert_raw_unchanged(self.raw, original)
        self.assert_raw_unchanged(self.raw_prep, prep_before)
        axes = {ax.get_title(): ax for fig in figures for ax in fig.axes}
        mean_ax = axes["Channel-averaged PSD"]
        for line, power in zip(mean_ax.lines, (before, after)):
            np.testing.assert_array_equal(line.get_xdata(), frequencies)
            np.testing.assert_allclose(
                line.get_ydata(), np.mean(10 * np.log10(power * 1e12), axis=0)
            )
        attenuation = axes["Line-noise removal attenuation, per channel"]
        np.testing.assert_allclose(
            attenuation.lines[0].get_ydata(), 10 * np.log10(before[0] / after[0])
        )
        i30 = np.argmin(np.abs(frequencies - 30))
        self.assertAlmostEqual(attenuation.lines[0].get_ydata()[i30], 20, delta=0.1)
        i10 = np.argmin(np.abs(frequencies - 10))
        self.assertAlmostEqual(attenuation.lines[0].get_ydata()[i10], 0, delta=0.02)

        self.assertEqual(len(spectrogram_calls), 2)
        for (args, kwargs, _), expected in zip(
                spectrogram_calls, (prep_before, filtered_after)):
            np.testing.assert_array_equal(
                args[0], expected.get_data(picks=["E75"], units="uV")[0]
            )
            self.assertEqual(args[1], 400)
            self.assertEqual(kwargs["frequency_range"], [1, 150])
            self.assertEqual(kwargs["time_bandwidth"], 2)
            self.assertEqual(kwargs["num_tapers"], 3)
            self.assertEqual(kwargs["window_params"], [2, 1])
            self.assertEqual(kwargs["min_nfft"], 8192)
        expected_limits = np.nanpercentile(
            [10 * np.log10(call[2][0]) for call in spectrogram_calls], [5, 98]
        )
        for title in ("E75: Before cleaning", "E75: After cleaning"):
            np.testing.assert_allclose(axes[title].collections[0].get_clim(),
                                       expected_limits)

    def test_inspection_suppresses_routine_output_and_zero_reference_warning(self):
        self.raw._data[3] = 0
        self.raw_prep._data[3] = 0
        montage = mne.channels.make_dig_montage(ch_pos={
            "E75": (-0.04, -0.07, 0.05), "E1": (0.04, 0.04, 0.07),
            "E2": (-0.04, 0.04, 0.07), "Cz": (0.0, 0.0, 0.095),
        }, coord_frame="head")
        for raw in (self.raw, self.raw_prep):
            raw.set_annotations(mne.Annotations([2], [0.25], ["BAD_test"]))
            raw.set_montage(montage)
        output = io.StringIO()

        with patch.object(plt, "show"), redirect_stdout(output), \
                mne.utils.catch_logging(verbose="INFO") as logs, \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figures = step1.plot_line_noise_removal(self.raw, self.raw_prep)

        self.assertEqual(len(figures), 8)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(logs.getvalue(), "")
        self.assertFalse(any("Zero value in spectrum for channel Cz" in str(w.message)
                             for w in caught))


if __name__ == "__main__":
    unittest.main()
