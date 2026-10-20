"""Numerical and display contracts for the reviewed ICA exclusion summary."""

import contextlib
import io
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step3_ica_cleaning as step3
from step3_ica_viewer import build as builder
from test_ica_viewer import make_inputs


class TestICAExclusionSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fitted_ica, cls.epochs, raw, _ = make_inputs(mixed=True)
        rng = np.random.default_rng(619)
        sources = rng.normal(size=(4, raw.n_times))
        sources[1] = 0.9 * sources[0] + 0.2 * sources[1]
        # Put most variance in annotated samples so including them would produce
        # an appreciably different result from annotation-clean original data.
        sources[:2, 2000:3000] *= 20
        maps = cls.fitted_ica.pre_whitener_ * cls.fitted_ica.get_components()
        means = cls.fitted_ica.pre_whitener_ * cls.fitted_ica.pca_mean_[:, None]
        raw._data[:4] = maps @ sources + means
        extra = mne.io.RawArray(
            np.vstack((np.full(raw.n_times, np.nan),
                       rng.normal(scale=1e6, size=raw.n_times))),
            mne.create_info(["BAD_EEG", "AUX"], 1000, ["eeg", "misc"]),
            first_samp=raw.first_samp, verbose=False,
        )
        raw.add_channels([extra], force_update_info=True)
        raw.info["bads"] = ["BAD_EEG"]
        raw.set_annotations(mne.Annotations([2], [1], ["BAD_test"]))
        cls.original_raw = raw

    def setUp(self):
        self.ica = self.fitted_ica.copy()
        self.raw = self.original_raw.copy()
        self.addCleanup(plt.close, "all")

    def summarize(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(plt, "show") as show, mne.use_log_level("ERROR"):
            figure = step3.summarize_ica_exclusions(self.ica, self.epochs, self.raw)
        return figure, output.getvalue(), show

    def reduction(self, exclude):
        """Compute expected sensor-space subtraction independently of ICA.apply."""
        original = self.raw.get_data(picks=self.ica.ch_names)
        sources = self.ica.get_sources(self.raw).get_data()
        maps = self.ica.pre_whitener_ * self.ica.get_components()
        cleaned = original - maps[:, exclude] @ sources[exclude]
        # The known BAD_test interval is 2--3 s relative to the Raw start.
        retained = np.ones(self.raw.n_times, dtype=bool)
        retained[2000:3000] = False
        return 100 * (1 - np.var(cleaned[:, retained], axis=1).sum()
                      / np.var(original[:, retained], axis=1).sum())

    def test_reports_joint_original_rate_variance_without_mutating_inputs(self):
        self.ica.exclude = [0, 1]
        raw_before = self.raw.get_data().copy()
        epochs_before = self.epochs.get_data(copy=True)
        expected = self.reduction([0, 1])
        individual_sum = self.reduction([0]) + self.reduction([1])
        self.assertGreater(abs(expected - individual_sum), 1.0)

        figure, output, show = self.summarize()

        match = re.search(r"variance[^\n]*?([-+]?\d+(?:\.\d+)?)\s*%", output,
                          flags=re.IGNORECASE)
        self.assertIsNotNone(match, output)
        self.assertAlmostEqual(float(match.group(1)), expected, delta=0.005)
        self.assertIn("[0, 1]", output)
        self.assertRegex(output, r"(?i)(?:excluded|components)[^\n]*\b2\b")
        np.testing.assert_array_equal(self.raw.get_data(), raw_before)
        np.testing.assert_array_equal(self.epochs.get_data(), epochs_before)
        self.assertEqual(self.raw.info["bads"], ["BAD_EEG"])
        self.assertEqual(self.ica.exclude, [0, 1])
        self.assertEqual(self.raw.info["sfreq"], 1000)
        self.assertIsNotNone(figure)
        show.assert_called_once()

    def test_four_exclusions_have_three_pairs_per_row_and_hidden_unused_axes(self):
        self.ica.exclude = [0, 1, 2, 3]
        figure, _, _ = self.summarize()
        self.assertEqual(len(figure.axes), 12)
        self.assertEqual(sum(axis.get_visible() for axis in figure.axes), 8)
        gridspec = figure.axes[0].get_subplotspec().get_gridspec()
        self.assertEqual((gridspec.nrows, gridspec.ncols), (2, 6))
        for index in (0, 2, 4, 6):
            self.assertTrue(figure.axes[index].images)
            self.assertTrue(figure.axes[index + 1].lines)

    def test_psd_uses_exact_fitting_epochs_and_their_sampling_rate(self):
        self.ica.exclude = [2]
        expected_sources = self.ica.get_sources(self.epochs).get_data(picks=[2])
        expected_frequencies, expected_power = builder._median_psd(
            expected_sources, self.epochs.info["sfreq"],
        )
        with patch.object(builder, "_median_psd", wraps=builder._median_psd) as psd:
            figure, _, _ = self.summarize()
        psd.assert_called_once()
        self.assertEqual(psd.call_args.args[1], 250)
        np.testing.assert_array_equal(psd.call_args.args[0], expected_sources)
        axis = figure.axes[1]
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), expected_frequencies)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected_power[0])
        self.assertEqual(axis.get_xscale(), "linear")
        self.assertEqual(axis.get_yscale(), "linear")
        self.assertEqual(axis.get_xlim(), (0, 50))

    def test_no_exclusions_prints_zero_and_skips_cleaning_and_plotting(self):
        self.ica.exclude = []
        with patch.object(self.raw, "copy", side_effect=AssertionError("No cleaning needed")), \
                patch.object(self.ica, "get_sources", side_effect=AssertionError("No PSD needed")):
            figure, output, show = self.summarize()
        self.assertIsNone(figure)
        self.assertIn("[]", output)
        self.assertRegex(output, r"(?i)(?:excluded|components)[^\n]*\b0\b")
        self.assertRegex(output, r"(?i)variance[^\n]*\b0(?:\.0+)?\s*%")
        show.assert_not_called()

    def test_changes_only_within_bad_span_do_not_change_variance_reduction(self):
        self.ica.exclude = [0, 1]
        _, original_output, _ = self.summarize()
        self.raw._data[:4, 2000:3000] = np.random.default_rng(925).normal(
            scale=1e6, size=(4, 1000),
        )
        _, changed_output, _ = self.summarize()
        pattern = r"variance[^\n]*?([-+]?\d+(?:\.\d+)?)\s*%"
        original_percent = re.search(pattern, original_output, re.IGNORECASE).group(1)
        changed_percent = re.search(pattern, changed_output, re.IGNORECASE).group(1)
        self.assertEqual(original_percent, changed_percent)

    def test_all_bad_data_has_no_variance_support(self):
        self.ica.exclude = [0]
        self.raw.set_annotations(mne.Annotations([0], [8], ["BAD_entire_recording"]))
        with self.assertRaisesRegex(ValueError, "No unannotated samples remain"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
