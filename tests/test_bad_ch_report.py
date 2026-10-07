import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import mne
import numpy as np

from test_robust_average_reference import NAMES, make_native_raw, step2


class TestBadChannelReport(unittest.TestCase):
    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_voltage_mask_selects_a_continuous_window_and_preserves_input(self):
        raw = make_native_raw(seconds=24).crop(tmin=2, tmax=21.996)
        sfreq = int(raw.info["sfreq"])
        raw._data[:12, :2 * sfreq] += 800e-6
        raw._data[:12, 12 * sfreq:] += 800e-6
        raw.info["bads"] = ["O1"]
        raw.set_annotations(mne.Annotations([1], [0], ["event"]))
        report = {"empty": [], "std_high": ["O1", "EOG"],
                  "correlation": ["F3", "O1"]}
        original = raw.copy()
        with patch.object(step2, "_temporary_voltage_mask",
                          wraps=step2._temporary_voltage_mask) as mask:
            fig = step2.plot_bad_ch_topo_traces(raw, report)
        mask.assert_called_once_with(raw)
        axes = {ax.texts[0].get_text(): ax for ax in fig.axes[1:]}
        self.assertEqual(set(axes), set(NAMES))
        for name, ax in axes.items():
            self.assertEqual(ax.get_xlim(), (2, 12))
            self.assertEqual(ax.get_ylim(), (-100, 100))
            self.assertFalse(ax.axison)
            self.assertFalse(ax.patch.get_visible())
            np.testing.assert_array_equal(ax.lines[0].get_xdata(),
                                          np.arange(2 * sfreq, 12 * sfreq) / sfreq)
            np.testing.assert_array_equal(
                ax.lines[0].get_ydata(),
                original.get_data(picks=[name], start=2 * sfreq,
                                  stop=12 * sfreq)[0] * 1e6,
            )
        palette = plt.get_cmap("tab10")
        self.assertEqual(to_rgba(axes["O1"].lines[0].get_color()), palette(1))
        self.assertEqual(to_rgba(axes["F3"].lines[0].get_color()), palette(2))
        self.assertEqual(to_rgba(axes["Cz"].lines[0].get_color()), to_rgba("black"))
        self.assertEqual(axes["O1"].lines[1].get_marker(), "o")
        self.assertEqual(to_rgba(axes["O1"].lines[1].get_color()), palette(2))
        self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()],
                         ["Unflagged", "std_high", "correlation"])
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        self.assertEqual(raw.info["bads"], original.info["bads"])
        self.assertEqual(raw.annotations, original.annotations)
        self.assertEqual(report, {"empty": [], "std_high": ["O1", "EOG"],
                                  "correlation": ["F3", "O1"]})

    def test_random_selection_covers_all_eligible_starts_and_avoids_annotations(self):
        raw = make_native_raw(seconds=32)
        raw.set_annotations(mne.Annotations([10, 24], [2, 1], ["BAD_a", "BAD_b"]))
        # One legal start in 0-10 s, 501 in 12-24 s, none in 25-32 s.
        for draw, expected_start in ((0, 0), (1, 12), (501, 14)):
            with self.subTest(draw=draw):
                rng = Mock()
                rng.integers.return_value = draw
                with patch.object(step2.np.random, "default_rng", return_value=rng):
                    fig = step2.plot_bad_ch_topo_traces(raw, {})
                rng.integers.assert_called_once_with(502)
                for ax in fig.axes[1:]:
                    self.assertEqual(ax.get_xlim(), (expected_start, expected_start + 10))
                plt.close(fig)

    def test_no_unmasked_window_warns_without_creating_a_figure(self):
        raw = make_native_raw(seconds=20)
        raw._data[:12] += 800e-6
        with self.assertWarnsRegex(UserWarning, "No unmasked 10-s window"):
            self.assertIsNone(step2.plot_bad_ch_topo_traces(raw, {}))
        self.assertEqual(plt.get_fignums(), [])

    def test_report_adds_the_companion_and_returns_the_original_topoplot(self):
        raw = make_native_raw(seconds=12)
        report = {"empty": [], "std_high": ["O1"], "correlation": ["F3"]}
        topoplot = step2.plot_bad_ch_report(raw, report)
        figures = [plt.figure(number) for number in plt.get_fignums()]
        self.assertEqual(len(figures), 2)
        self.assertIs(topoplot, figures[0])
        self.assertEqual(len(topoplot.axes), 1)
        companion = figures[1]
        axes = {ax.texts[0].get_text(): ax for ax in companion.axes[1:]}
        for ring, name in zip(topoplot.axes[0].collections[1:], ("O1", "F3")):
            np.testing.assert_array_equal(ring.get_edgecolors()[0],
                                          to_rgba(axes[name].lines[0].get_color()))

    def test_short_metadata_only_raw_keeps_the_report_topoplot_available(self):
        raw = make_native_raw(seconds=12)
        short = mne.io.RawArray(np.zeros((len(raw.ch_names), 2)), raw.info,
                                verbose=False)
        with self.assertWarnsRegex(UserWarning, "No unmasked 10-s window"):
            fig = step2.plot_bad_ch_report(short, {"std_high": ["O1"]})
        self.assertEqual(plt.get_fignums(), [fig.number])


if __name__ == "__main__":
    unittest.main()
