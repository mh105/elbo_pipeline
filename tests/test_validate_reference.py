import sys
import unittest
from pathlib import Path

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2
from test_flatline_interpolation import NAMES, flatlines_for, make_raw


def native_with_reference_artifact():
    """Native Cz-referenced data whose reference electrode carries a slow
    artifact, so every other channel holds it with opposite sign; also return
    the data without the artifact and a flatlines dictionary."""
    raw = make_raw()
    clean = raw.get_data()
    flatlines = flatlines_for(raw, {})
    artifact = 200e-6 * np.sin(2 * np.pi * 0.5 * raw.times)
    rows = [row for row, name in enumerate(NAMES) if name != "Cz"]
    raw._data[rows] -= artifact
    return raw, clean, flatlines


class TestValidateReference(unittest.TestCase):
    def test_good_reference_is_kept(self):
        raw = make_raw()
        raw.info["bads"] = ["T7", "EOG"]
        original = raw.get_data()
        self.assertEqual(step2.validate_reference_inplace(raw, flatlines_for(raw, {})), "Cz")
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_bad_reference_moves_to_its_nearest_good_electrode(self):
        raw, clean, flatlines = native_with_reference_artifact()
        eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        nearest = step2._nearest_channels(raw, eeg, n_neighbours=5)[1]["Cz"]
        raw.info["bads"] = ["Cz", nearest[0], "EOG"]
        native = raw.get_data()
        with self.assertWarnsRegex(UserWarning, f"EEG channel to {nearest[1]}, the nearest"):
            reference = step2.validate_reference_inplace(raw, flatlines)
        self.assertEqual(reference, nearest[1])
        row = raw.ch_names.index(reference)
        np.testing.assert_array_equal(raw._data[eeg], native[eeg] - native[row])
        np.testing.assert_array_equal(raw._data[row], 0)
        # The reference's artifact cancels in every channel but the bad Cz.
        others = [i for i in eeg if raw.ch_names[i] != "Cz"]
        np.testing.assert_allclose(raw._data[others], clean[others] - clean[row],
                                   rtol=0, atol=1e-18)
        np.testing.assert_array_equal(raw._data[-1], native[-1])
        self.assertEqual(raw.info["bads"], ["Cz", nearest[0], "EOG"])
        self.assertEqual(flatlines["reference_channel"], "Cz")
        self.assertEqual(step2._reference_channel(raw, "native"), reference)

    def test_raises_without_changes_if_the_five_nearest_are_bad(self):
        raw, _, flatlines = native_with_reference_artifact()
        eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        nearest = step2._nearest_channels(raw, eeg, n_neighbours=5)[1]["Cz"]
        raw.info["bads"] = ["Cz"] + nearest
        original = raw.get_data()
        with self.assertRaisesRegex(ValueError, "No valid native reference"):
            step2.validate_reference_inplace(raw, flatlines)
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_requires_the_native_reference(self):
        raw = make_raw()
        flatlines = flatlines_for(raw, {})
        raw._data[NAMES.index("Cz")] += 1e-6
        with self.assertRaisesRegex(ValueError, "not entirely zero"):
            step2.validate_reference_inplace(raw, flatlines)


if __name__ == "__main__":
    unittest.main()
