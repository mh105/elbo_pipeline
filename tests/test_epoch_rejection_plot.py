"""Check static rejection plots against original trial identities."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import mne
from mne.viz.utils import _compute_scalings
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step4_epoching as step4


class TestEpochRejectionPlot(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(418)
        data = rng.normal(scale=8e-6, size=(44, 3, 31))
        events = np.column_stack((1000 + 100 * np.arange(44),
                                  np.zeros(44, dtype=int),
                                  np.ones(44, dtype=int)))
        self.before = mne.EpochsArray(
            data, mne.create_info(["E1", "E2", "BAD"], 100, "eeg"),
            events=events, tmin=-0.1, baseline=None, verbose=False,
        )
        self.before.info["bads"] = ["BAD"]
        with mne.use_log_level("ERROR"):
            self.before.drop([7], reason="BAD_annotation")
        self.cleaned = self.before.copy()
        # Model interpolation changing accepted traces. The review must show
        # original waveforms for both the accepted and the rejected trials.
        self.cleaned._data[:, :2] = (
            self.cleaned._data[:, :2, ::-1].copy() * 0.25
        )
        self.rejected_trials = set(self.before.selection[[3, 41]])
        with mne.use_log_level("ERROR"):
            self.cleaned.drop([3, 41], reason="AUTOREJECT")
        self.addCleanup(plt.close, "all")

    def plot(self, before=None, cleaned=None, candidates=None, **kwargs):
        with patch.object(plt, "show") as show:
            figures = step4.plot_epoch_rejection(
                self.before if before is None else before,
                self.cleaned if cleaned is None else cleaned,
                ["missing", "E2", "BAD", "E1"]
                if candidates is None else candidates,
                **kwargs,
            )
        return figures, show

    @staticmethod
    def waveform_shape(values):
        differences = np.diff(values)
        return differences / np.linalg.norm(differences)

    def relative_trace_height(self, figure):
        target_shape = self.waveform_shape(self.before.get_data()[0, 1])
        axis = figure.axes[0]
        trace = next(
            line for line in axis.lines
            if len(line.get_ydata()) == len(self.before.times)
            and np.allclose(self.waveform_shape(line.get_ydata()), target_shape)
        )
        spacing = abs(np.diff(axis.get_yticks())[0])
        return np.ptp(trace.get_ydata()) / spacing

    def test_original_epochs_and_saved_rejections_cover_all_trials(self):
        before = self.before
        with tempfile.TemporaryDirectory(prefix="spindrift-epochs-", dir="/tmp") as tmp:
            cleaned_file = Path(tmp) / "cleaned-epo.fif"
            with mne.use_log_level("ERROR"):
                self.cleaned.save(cleaned_file)
                cleaned = mne.read_epochs(cleaned_file, preload=True, verbose=False)
            before_data = before.get_data(copy=True)
            cleaned_data = cleaned.get_data(copy=True)
            figures, show = self.plot(before=before, cleaned=cleaned)

        self.assertEqual(len(figures), 3)
        show.assert_called_once_with(block=False)
        plotted_trials = []
        for page, figure in enumerate(figures):
            axes = [axis for axis in figure.axes if axis.get_visible()]
            self.assertEqual(len(axes), 1)
            axis = axes[0]
            start, stop = page * 20, min((page + 1) * 20, len(before))
            page_trials = before.selection[start:stop]
            self.assertEqual(
                [tick.get_text() for tick in axis.get_xticklabels()],
                [str(trial + 1) for trial in page_trials],
            )
            self.assertEqual(
                [tick.get_text() for tick in axis.get_yticklabels()],
                ["E2", "E1"],
            )
            traces = [line for line in axis.lines
                      if len(line.get_ydata()) == len(before.times)]
            self.assertEqual(len(traces), 2 * len(page_trials))
            expected = {
                (trial, channel): self.waveform_shape(before_data[index, channel])
                for index, trial in enumerate(before.selection)
                if trial in page_trials
                for channel in (1, 0)
            }
            seen = set()
            for line in traces:
                shape = self.waveform_shape(line.get_ydata())
                matches = [key for key, values in expected.items()
                           if np.allclose(shape, values, rtol=1e-5, atol=1e-7)]
                self.assertEqual(len(matches), 1)
                trial, channel = matches[0]
                self.assertNotIn((trial, channel), seen)
                seen.add((trial, channel))
                color = "red" if trial in self.rejected_trials else "black"
                self.assertEqual(to_rgba(line.get_color()), to_rgba(color))
            self.assertEqual(seen, set(expected))
            plotted_trials.extend(page_trials)

        np.testing.assert_array_equal(plotted_trials, before.selection)
        np.testing.assert_array_equal(before.get_data(), before_data)
        np.testing.assert_array_equal(cleaned.get_data(), cleaned_data)

    def test_rejected_trials_do_not_affect_automatic_scale(self):
        figures, _ = self.plot()
        original_spacings = [abs(np.diff(fig.axes[0].get_yticks())[0])
                             for fig in figures]
        original_height = self.relative_trace_height(figures[0])
        with_artifact = self.before.copy()
        rejected = np.isin(with_artifact.selection, list(self.rejected_trials))
        with_artifact._data[rejected] = np.random.default_rng(227).normal(
            loc=1e4, scale=1e3, size=with_artifact._data[rejected].shape,
        )
        figures, _ = self.plot(before=with_artifact)

        for figure, expected_spacing in zip(figures, original_spacings):
            spacing = abs(np.diff(figure.axes[0].get_yticks())[0])
            self.assertEqual(spacing, expected_spacing)
        self.assertAlmostEqual(self.relative_trace_height(figures[0]), original_height)

    def test_automatic_gain_tracks_uniform_amplitude_changes(self):
        figures, _ = self.plot()
        original_spacing = abs(np.diff(figures[0].axes[0].get_yticks())[0])
        original_height = self.relative_trace_height(figures[0])
        scaled = self.before.copy()
        scaled._data *= 10
        figures, _ = self.plot(before=scaled)

        for figure in figures:
            spacing = abs(np.diff(figure.axes[0].get_yticks())[0])
            self.assertAlmostEqual(spacing, 10 * original_spacing)
        self.assertAlmostEqual(self.relative_trace_height(figures[0]), original_height)

    def test_retained_scale_uses_widened_mne_auto_iqr_for_plotted_good_channels(self):
        before = self.before.copy()
        before._data[:, 2] *= 1e6
        retained = before[np.isin(before.selection, self.cleaned.selection)]
        expected_scale_uv = _compute_scalings(
            "auto", retained.pick(["E2", "E1"]),
        )["eeg"] * 1e6
        figures, _ = self.plot(before=before)

        for figure in figures:
            spacing = abs(np.diff(figure.axes[0].get_yticks())[0])
            self.assertAlmostEqual(spacing, 6 * expected_scale_uv)

    def test_mismatched_trial_membership_is_rejected(self):
        mismatched = self.before.copy()
        with mne.use_log_level("ERROR"):
            mismatched.drop([5], reason="BAD_different_run")
        with self.assertRaises(ValueError):
            self.plot(before=mismatched)

    def test_mismatched_retained_event_samples_are_rejected(self):
        mismatched = self.before.copy()
        mismatched.events[0, 0] += 1
        with self.assertRaises(ValueError):
            self.plot(before=mismatched)

    def test_no_usable_candidate_channels_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plot(candidates=["missing", "BAD"])

    def test_nonpositive_page_sizes_are_rejected(self):
        for n_epochs in (0, -1):
            with self.subTest(n_epochs=n_epochs), self.assertRaises(ValueError):
                self.plot(n_epochs=n_epochs)


if __name__ == "__main__":
    unittest.main()
