import sys
import unittest
from pathlib import Path

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


SFREQ = 1000
N_TIMES = 60 * SFREQ  # One minute, so pop counts equal pops per minute.


def pop_raw(data, montage=True):
    """16 EEG channels on a 4 x 4 grid plus a zero Cz reference at its center."""
    names = [f"E{i}" for i in range(16)] + ["Cz"]
    data = np.vstack([data, np.zeros(data.shape[1])])
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, "eeg"), verbose=False)
    if montage:
        grid = [(i % 4, i // 4) for i in range(16)] + [(1.5, 1.5)]
        raw.set_montage(mne.channels.make_dig_montage(
            {name: (0.02 * x, 0.02 * y, 0.08) for name, (x, y) in zip(names, grid)},
            coord_frame="head",
        ))
    return raw


def background(seed=0):
    return np.random.default_rng(seed).normal(0, 2e-6, (16, N_TIMES))


def add_steps(data, times_s, size=100e-6):
    """Alternating up/down steps along the last (time) axis, so levels do not accumulate."""
    for k, time_s in enumerate(times_s):
        data[..., int(time_s * SFREQ):] += (-1) ** k * size


class TestPopDetection(unittest.TestCase):
    def test_counts_isolated_steps_and_flags_strictly_above_rate_cutoff(self):
        data = background()
        add_steps(data[5], [10, 30, 50])
        add_steps(data[6], [20])
        raw = pop_raw(data)
        raw.set_annotations(mne.Annotations([5.0], [0.0], ["event"]))
        original = raw.get_data().copy()
        report, details = step2.bad_ch_by_pop(raw, return_details=True)
        self.assertEqual(report, {"pop": ["E5"]})
        self.assertEqual(raw.info["bads"], ["E5"])
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertEqual(list(raw.annotations.description), ["event"])

        self.assertAlmostEqual(details["retained_min"], 1.0)
        n_pops = dict(zip(details["ch_names"], details["n_pops"]))
        self.assertEqual((n_pops["E5"], n_pops["E6"]), (3, 1))
        self.assertEqual(sum(n_pops.values()), 4)
        # Each pop is timed at the sample before its jump.
        np.testing.assert_allclose(details["pop_times_s"]["E5"], [9.999, 29.999, 49.999])
        np.testing.assert_allclose(details["pop_steps_uv"]["E5"], [100, -100, 100], atol=2)
        self.assertEqual(len(details["neighbors"]["E5"]), 6)
        self.assertTrue({"E1", "E4", "E6", "E9"} <= set(details["neighbors"]["E5"]))
        self.assertNotIn("Cz", details["ch_names"])

        # One pop per minute is not strictly above the default cutoff.
        report = step2.bad_ch_by_pop(pop_raw(data), rate_cutoff=0.5)
        self.assertEqual(report, {"pop": ["E5", "E6"]})

    def test_spikes_shared_steps_and_slow_ramps_are_not_pops(self):
        data = background()
        for time_s in np.arange(2, 58, 2.0):
            start = int(time_s * SFREQ)
            data[9, start:start + 5] -= 200e-6  # 5-ms spike back to baseline.
        add_steps(data, np.arange(3, 58, 2.0))  # Every channel steps together.
        for k, time_s in enumerate(np.arange(2.5, 58, 2.0)):
            # 400-uV, 200-ms ramps with a 70-uV jump halfway: the level changes
            # over 10-20 ms windows are only ~60% of those over 10-60 ms.
            start, sign = int(time_s * SFREQ), (-1) ** k
            data[12, start:start + 200] += sign * np.linspace(0, 400e-6, 200)
            data[12, start + 200:] += sign * 400e-6
            data[12, start + 100:] += sign * 70e-6
        report, details = step2.bad_ch_by_pop(pop_raw(data), return_details=True)
        self.assertEqual(report, {"pop": []})
        self.assertEqual(details["n_pops"].sum(), 0)

        # Without the neighbor comparison, all 28 shared steps count on every channel.
        _, details = step2.bad_ch_by_pop(pop_raw(data), neighbor_ratio=1e-9, return_details=True)
        np.testing.assert_array_equal(details["n_pops"], 28)

        report, details = step2.bad_ch_by_pop(
            pop_raw(data), sharpness_cutoff=0.5, return_details=True
        )
        self.assertEqual(report, {"pop": ["E12"]})
        self.assertEqual(details["n_pops"][details["ch_names"].index("E12")], 28)

    def test_bad_annotations_are_skipped_and_leave_the_rate_denominator(self):
        data = background()
        add_steps(data[5], [10, 30, 50])
        raw = pop_raw(data)
        raw.set_annotations(mne.Annotations([20.0], [20.0], ["BAD_existing"]))
        report, details = step2.bad_ch_by_pop(raw, return_details=True)
        self.assertEqual(report, {"pop": ["E5"]})
        self.assertAlmostEqual(details["retained_min"], 40 / 60, places=4)
        np.testing.assert_allclose(details["pop_times_s"]["E5"], [9.999, 49.999])
        np.testing.assert_allclose(details["pops_per_min"][5], 3.0, rtol=1e-3)

    def test_average_reference_and_existing_bads(self):
        data = background()
        add_steps(data[5], [10, 30, 50])
        raw = pop_raw(data)
        raw.info["bads"] = ["E6"]
        raw.set_eeg_reference("average", projection=False, verbose=False)
        report, details = step2.bad_ch_by_pop(raw, return_details=True)
        self.assertEqual(report, {"pop": ["E5"]})
        self.assertEqual(raw.info["bads"], ["E6", "E5"])
        self.assertIn("Cz", details["ch_names"])  # No longer a zero reference.
        self.assertNotIn("E6", details["ch_names"])
        self.assertFalse(any("E6" in names for names in details["neighbors"].values()))
        np.testing.assert_allclose(details["pop_times_s"]["E5"], [9.999, 29.999, 49.999])

    def test_local_flat_spans_are_neither_pops_nor_rate_denominator(self):
        data = background()
        for start in (10, 25, 40):  # Three 3-s drops of E5 to a 100-uV level.
            data[5, start * SFREQ:(start + 3) * SFREQ] = 100e-6
        raw = pop_raw(data)
        # Unbridged spans, as if the data had not been through detect_flatline_inplace.
        flatlines = step2.detect_flatline_inplace(raw.copy())
        step2.mask_segment_by_flatline(raw.copy(), flatlines)
        self.assertEqual(flatlines["ch_names"], ["E5"])
        report, details = step2.bad_ch_by_pop(raw, return_details=True)
        row = details["ch_names"].index("E5")
        self.assertEqual(report, {"pop": ["E5"]})  # Each span edge is a step.
        self.assertEqual(details["n_pops"][row], 6)
        self.assertEqual(details["scorable_min"][row], 1.0)
        raw.info["bads"] = []
        report, details = step2.bad_ch_by_pop(raw, flatlines=flatlines, return_details=True)
        self.assertEqual(report, {"pop": []})
        self.assertEqual(details["n_pops"].sum(), 0)
        self.assertAlmostEqual(details["scorable_min"][row], 51 / 60)
        np.testing.assert_allclose(np.delete(details["scorable_min"], row), 1.0)
        self.assertAlmostEqual(details["retained_min"], 1.0)

    def test_flat_neighbours_leave_the_neighbour_median(self):
        data = background()
        add_steps(data[5], [10, 30, 50])  # E5 steps by 100 uV...
        for row in (1, 4):  # ...with its nearest neighbours E1 and E4 (60 uV): shared.
            add_steps(data[row], [10, 30, 50], size=60e-6)
        raw = pop_raw(data)
        _, details = step2.bad_ch_by_pop(raw.copy(), return_details=True)
        others = [name for name in details["neighbors"]["E5"] if name not in ("E1", "E4")]
        self.assertEqual(len(others), 4)  # 4 of 16 flat at once is not a dropout.
        for name in others:  # E5's other neighbours are flat around each step.
            for time_s in (10, 30, 50):
                raw._data[raw.ch_names.index(name), (time_s - 1) * SFREQ:(time_s + 1) * SFREQ] = 0
        flatlines = step2.detect_flatline_inplace(raw.copy())
        step2.mask_segment_by_flatline(raw.copy(), flatlines)
        self.assertEqual(sorted(flatlines["ch_names"]), sorted(others))
        self.assertEqual(flatlines["global_segments_exact"], [])

        def e5_pops(**kwargs):
            _, details = step2.bad_ch_by_pop(raw.copy(), return_details=True, **kwargs)
            return details["n_pops"][details["ch_names"].index("E5")]
        self.assertEqual(e5_pops(), 3)  # Flat zeros pull the neighbour median to 0.
        self.assertEqual(e5_pops(flatlines=flatlines), 0)  # E1 and E4 share the step.

    def test_level_change_windows_are_symmetric_about_the_jump(self):
        # The jump at t lies between samples t and t + 1; both windows skip 10
        # samples, so a unit ramp changes by the 70-sample distance between centres.
        ramp = np.arange(2000.0)
        np.testing.assert_array_equal(step2._level_change(ramp, np.array([1000]), 10, 50), [70])
        np.testing.assert_array_equal(step2._level_change(ramp, np.array([59]), 10, 50), [70])

    def test_invalid_inputs(self):
        data = background()
        for kwargs in (
            {"jump_sd_cutoff": 0}, {"step_floor_uv": -1}, {"neighbor_ratio": np.nan},
            {"sharpness_cutoff": 1.5}, {"rate_cutoff": -1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                step2.bad_ch_by_pop(pop_raw(data), **kwargs)
        with self.assertRaisesRegex(ValueError, "sensor positions"):
            step2.bad_ch_by_pop(pop_raw(data, montage=False))
        raw = pop_raw(data)
        raw.info["bads"] = [f"E{i}" for i in range(10)]
        with self.assertRaisesRegex(ValueError, "seven"):
            step2.bad_ch_by_pop(raw)
        low_rate = pop_raw(data).resample(50, verbose=False)
        with self.assertRaisesRegex(ValueError, "100 Hz"):
            step2.bad_ch_by_pop(low_rate)


if __name__ == "__main__":
    unittest.main()
