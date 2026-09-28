import sys
import unittest
from pathlib import Path

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from step2_bad_ch_time_detection import bad_ch_by_impedance


class TestBadChByImpedance(unittest.TestCase):
    def setUp(self):
        info = mne.create_info(
            ["C3", "C4", "Cz", "EOG"], 100, ["eeg", "eeg", "eeg", "eog"]
        )
        self.raw = mne.io.RawArray(np.zeros((4, 10)), info, verbose=False)
        self.initial = [501, 100, 500, 600]
        self.end = [100, 501, 500, 700]

    def test_modes_boundary_and_channel_order(self):
        self.assertEqual(
            bad_ch_by_impedance(self.raw, self.initial), {"impedance": ["C3", "EOG"]}
        )
        self.assertEqual(self.raw.info["bads"], ["C3", "EOG"])
        for mode, expected in (
            ("initial", ["C3", "EOG"]),
            ("end", ["C4", "EOG"]),
            ("both", ["C3", "C4", "EOG"]),
        ):
            with self.subTest(mode=mode):
                self.raw.info["bads"] = []
                self.assertEqual(
                    bad_ch_by_impedance(self.raw, self.initial, self.end, mode),
                    {"impedance": expected},
                )
                self.assertEqual(self.raw.info["bads"], expected)

    def test_custom_cutoff_and_no_bad_channels(self):
        self.assertEqual(
            bad_ch_by_impedance(self.raw, self.initial, imp_cutoff=499),
            {"impedance": ["C3", "Cz", "EOG"]},
        )
        self.assertEqual(
            bad_ch_by_impedance(self.raw, [0, 100, 499, 500]), {"impedance": []}
        )
        self.assertEqual(self.raw.info["bads"], ["C3", "Cz", "EOG"])

    def test_returns_only_this_criterion_and_accumulates_bads_without_duplicates(self):
        self.raw.info["bads"] = ["C4", "C3"]
        original = self.raw.get_data().copy()
        result = bad_ch_by_impedance(self.raw, self.initial)
        self.assertEqual(result, {"impedance": ["C3", "EOG"]})
        self.assertEqual(bad_ch_by_impedance(self.raw, self.initial), result)
        self.assertEqual(self.raw.info["bads"], ["C4", "C3", "EOG"])
        np.testing.assert_array_equal(self.raw.get_data(), original)
        self.assertEqual(self.initial, [501, 100, 500, 600])

    def test_nan_readings_are_ignored(self):
        initial = [501, np.nan, np.nan, 500]
        end = [np.nan, 501, np.nan, 500]
        for mode, expected in (
            ("initial", ["C3"]), ("end", ["C4"]), ("both", ["C3", "C4"]),
        ):
            with self.subTest(mode=mode):
                self.raw.info["bads"] = []
                self.assertEqual(
                    bad_ch_by_impedance(self.raw, initial, end, mode),
                    {"impedance": expected},
                )
                self.assertEqual(self.raw.info["bads"], expected)

    def test_all_nan_preserves_existing_bads(self):
        self.raw.info["bads"] = ["C4"]
        self.assertEqual(bad_ch_by_impedance(self.raw, [np.nan] * 4), {"impedance": []})
        self.assertEqual(self.raw.info["bads"], ["C4"])

    def test_invalid_inputs(self):
        for kwargs in (
            {"reject_by": "unknown"}, {"reject_by": "end"}, {"reject_by": "both"},
            {"imp_cutoff": 0}, {"imp_cutoff": -1}, {"imp_cutoff": np.nan},
            {"imp_cutoff": np.inf}, {"end_impedance": [100]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bad_ch_by_impedance(self.raw, self.initial, **kwargs)
        for values in ([100], [[100] * 4], [100, 100, 100, 100, 100],
                       [0, 0, 0, np.inf], [0, 0, 0, -1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                bad_ch_by_impedance(self.raw, values)


if __name__ == "__main__":
    unittest.main()
