import sys
import unittest
from pathlib import Path

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from step1_line_noise_removal import ZapLine_notch


class TestZapLineNotch(unittest.TestCase):
    def setUp(self):
        sfreq = 1000
        times = np.arange(10000) / sfreq
        rng = np.random.default_rng(17)
        data = rng.normal(0, 1e-6, (8, len(times)))
        data[:6] += np.arange(1, 7)[:, None] * np.sin(2 * np.pi * 30 * times) * 5e-6
        data[1] += 300e-6 * np.sin(2 * np.pi * 11 * times)
        data[4] += rng.normal(0, 500e-6, len(times))
        info = mne.create_info(
            ["E1", "E2", "E3", "E4", "E5", "E6", "Trigger", "EOG"],
            sfreq, ["eeg"] * 6 + ["misc", "eog"],
        )
        self.raw = mne.io.RawArray(data, info, verbose=False)
        self.raw.info["bads"] = ["E2", "E5"]

    def test_preserves_bad_and_auxiliary_channels_and_input(self):
        original = self.raw.get_data().copy()
        with mne.use_log_level("ERROR"):
            clean = ZapLine_notch(self.raw, 30.0)
        excluded = ["E2", "E5", "Trigger", "EOG"]
        np.testing.assert_array_equal(
            clean.get_data(picks=excluded), self.raw.get_data(picks=excluded)
        )
        np.testing.assert_array_equal(self.raw.get_data(), original)
        self.assertIsNot(clean, self.raw)
        self.assertEqual(clean.ch_names, self.raw.ch_names)
        self.assertEqual(clean.info["bads"], ["E2", "E5"])

    def test_good_channel_cleaning_matches_input_without_bad_channels(self):
        good = self.raw.copy().pick("eeg", exclude="bads")
        with mne.use_log_level("ERROR"):
            expected = ZapLine_notch(good, 30.0)
            clean = ZapLine_notch(self.raw, 30.0)
        np.testing.assert_allclose(
            clean.get_data(picks=good.ch_names), expected.get_data(),
            rtol=1e-12, atol=1e-15,
        )
        line = np.sin(2 * np.pi * 30 * good.times[2000:-2000])
        self.assertLess(
            np.linalg.norm(expected.get_data()[:, 2000:-2000] @ line),
            0.2 * np.linalg.norm(good.get_data()[:, 2000:-2000] @ line),
        )


if __name__ == "__main__":
    unittest.main()
