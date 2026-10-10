import sys
import unittest
from pathlib import Path

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7",
         "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
         "P8", "PO3", "PO4", "O1", "Oz", "O2"]
REACH = 250  # 1 s at 250 Hz stands in for the 0.3 Hz high-pass reach.


def make_raw(n_times=5000):  # 20 s of native-reference noise
    rng = np.random.default_rng(311)
    data = rng.normal(scale=10e-6, size=(len(NAMES) + 1, n_times))
    data[:len(NAMES)] -= data[NAMES.index("Cz")]
    raw = mne.io.RawArray(data, mne.create_info(NAMES + ["EOG"], 250,
                                              ["eeg"] * len(NAMES) + ["eog"]), verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("colin27_1020"), verbose=False)
    return raw


def flatlines_for(raw, spans=None, global_segments=(), widen_samples=REACH):
    """Local flat intervals as detect_flatline_inplace saves them, already widened."""
    spans = spans or {}
    return {
        "ch_names": list(spans),
        "segments": [np.asarray(value, dtype=int).reshape(-1, 2) for value in spans.values()],
        "raw_ch_names": list(raw.ch_names), "eeg_ch_names": NAMES.copy(),
        "sfreq": float(raw.info["sfreq"]), "first_samp": int(raw.first_samp),
        "n_times": int(raw.n_times), "reference_channel": "Cz",
        "global_segments": [list(bounds) for bounds in global_segments],
        "widen_samples": widen_samples,
    }


def add_pop(raw, name, onset, step_uv=500.0):
    """A sustained step starting at sample onset."""
    raw._data[raw.ch_names.index(name), onset:] += step_uv * 1e-6


def expected_repair(raw, original, name, start, stop, invalid, fade_in=True, fade_out=True):
    """Spline estimate from the good donors, blended in by 0.25-s raised cosines
    at the requested ends."""
    geometry = step2._spline_geometry(raw)
    donors = [i for i in geometry["picks"] if raw.ch_names[i] not in raw.info["bads"]
              and raw.ch_names[i] not in invalid]
    target = raw.ch_names.index(name)
    estimate = (step2._spline_weights(geometry, donors, [target])
                @ original[donors, start:stop])[0]
    fade = min(int(round(0.25 * raw.info["sfreq"])), (stop - start) // 2)
    weight = np.ones(stop - start)
    rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)
    if fade_in:
        weight[:fade] = rise
    if fade_out:
        weight[stop - start - fade:] = rise[::-1]
    data = original[target, start:stop]
    return weight * estimate + (1 - weight) * data, estimate


class TestLocalPopInterpolation(unittest.TestCase):
    def test_repairs_the_reach_around_a_pop_and_nothing_else(self):
        raw = make_raw()
        raw.info["bads"] = ["T7"]
        raw.set_annotations(mne.Annotations([18], [0.5], ["BAD_manual"]))
        add_pop(raw, "C3", 2500)
        original = raw.get_data()
        annotations = raw.annotations.copy()

        result = step2.interpolate_local_pops_inplace(raw, flatlines_for(raw))

        self.assertEqual(result["ch_names"], ["C3"])
        np.testing.assert_array_equal(result["segments"][0], [[2250, 2750]])
        np.testing.assert_allclose(result["pop_times_s"]["C3"], [2499 / 250])
        np.testing.assert_allclose(result["pop_steps_uv"]["C3"], [500], rtol=0.05)
        expected, estimate = expected_repair(raw, original, "C3", 2250, 2750, {"C3"})
        row = raw.ch_names.index("C3")
        np.testing.assert_allclose(raw._data[row, 2250:2750], expected, rtol=0, atol=1e-15)
        # Away from the fades, the trace is the donor estimate, without the step.
        np.testing.assert_allclose(raw._data[row, 2312:2688], estimate[62:438],
                                   rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw._data[row, :2250], original[row, :2250])
        np.testing.assert_array_equal(raw._data[row, 2750:], original[row, 2750:])
        others = [i for i in range(len(raw.ch_names)) if i != row]
        np.testing.assert_array_equal(raw._data[others], original[others])
        np.testing.assert_array_equal(raw._data[raw.ch_names.index("Cz")], 0)
        self.assertEqual(raw.info["bads"], ["T7"])
        self.assertEqual(raw.annotations, annotations)

    def test_donors_exclude_concurrent_pops_and_overlapping_local_flats(self):
        raw = make_raw()
        add_pop(raw, "C3", 2500)
        add_pop(raw, "C4", 2600, step_uv=-400.0)
        original = raw.get_data()
        # P3 is flat inside both intervals; O1 only outside them, so it donates.
        flatlines = flatlines_for(raw, {"P3": [[2700, 2800]], "O1": [[4000, 4100]]})

        result = step2.interpolate_local_pops_inplace(raw, flatlines)

        self.assertEqual(result["ch_names"], ["C3", "C4"])
        for name, start, stop in [("C3", 2250, 2750), ("C4", 2350, 2850)]:
            expected, _ = expected_repair(raw, original, name, start, stop,
                                          {"C3", "C4", "P3"})
            np.testing.assert_allclose(raw._data[raw.ch_names.index(name), start:stop],
                                       expected, rtol=0, atol=1e-15)

    def test_global_dropouts_and_recording_edges_bound_the_intervals(self):
        raw = make_raw()
        raw.set_annotations(mne.Annotations([300 / 250], [50 / 250], ["BAD_by_flatline"]))
        add_pop(raw, "C3", 200)
        original = raw.get_data()

        result = step2.interpolate_local_pops_inplace(
            raw, flatlines_for(raw, global_segments=[(300, 350)]),
        )

        np.testing.assert_array_equal(result["segments"][0], [[0, 300], [350, 450]])
        row = raw.ch_names.index("C3")
        np.testing.assert_array_equal(raw._data[row, 300:350], original[row, 300:350])
        # No fade at the recording start or the dropout: there is nothing to blend.
        for start, stop, fade_in, fade_out in [(0, 300, False, False), (350, 450, False, True)]:
            expected, _ = expected_repair(raw, original, "C3", start, stop, {"C3"},
                                          fade_in, fade_out)
            np.testing.assert_allclose(raw._data[row, start:stop], expected,
                                       rtol=0, atol=1e-15)

    def test_frequent_pops_are_repaired_without_relabeling(self):
        raw = make_raw()
        for onset in [1000, 2500, 4000]:  # 9 pops/min, above bad_ch_by_pop's cutoff
            add_pop(raw, "C3", onset)
        original = raw.get_data()

        result = step2.interpolate_local_pops_inplace(raw, flatlines_for(raw))

        self.assertEqual(raw.info["bads"], [])
        np.testing.assert_array_equal(
            result["segments"][0], [[750, 1250], [2250, 2750], [3750, 4250]],
        )
        row = raw.ch_names.index("C3")
        np.testing.assert_array_equal(raw._data[row, 1250:2250], original[row, 1250:2250])

    def test_requires_a_good_native_reference(self):
        raw = make_raw()
        add_pop(raw, "C3", 2500)
        flatlines = flatlines_for(raw)
        eeg = slice(0, len(NAMES))
        average = raw.copy()  # average reference: no EEG channel is entirely zero
        average._data[eeg] -= average._data[eeg].mean(axis=0)
        bad_reference = raw.copy()  # validation has not moved the bad Cz yet
        bad_reference.info["bads"] = ["Cz"]
        for name, data in [("average", average), ("bad reference", bad_reference)]:
            with self.subTest(name=name):
                original = data.get_data()
                with self.assertRaisesRegex(ValueError, "native reference"):
                    step2.interpolate_local_pops_inplace(data, flatlines)
                np.testing.assert_array_equal(data.get_data(), original)
        # After validation moved the bad Cz's reference to FC1, FC1 is the zero
        # reference in effect: it donates and stays zero; the bad Cz does not.
        moved = bad_reference
        moved._data[eeg] -= moved._data[NAMES.index("FC1")].copy()
        original = moved.get_data()
        result = step2.interpolate_local_pops_inplace(moved, flatlines)
        self.assertEqual(result["ch_names"], ["C3"])
        expected, _ = expected_repair(moved, original, "C3", 2250, 2750, {"C3"})
        np.testing.assert_allclose(moved._data[moved.ch_names.index("C3"), 2250:2750], expected,
                                   rtol=0, atol=1e-15)
        np.testing.assert_array_equal(moved._data[moved.ch_names.index("FC1")], 0)
        self.assertEqual(moved.info["bads"], ["Cz"])

    def test_without_pops_or_reach_the_data_are_unchanged(self):
        raw = make_raw()
        original = raw.get_data()
        result = step2.interpolate_local_pops_inplace(raw, flatlines_for(raw))
        self.assertEqual(result["ch_names"], [])
        np.testing.assert_array_equal(raw.get_data(), original)

        add_pop(raw, "C3", 2500)
        popped = raw.get_data()
        with self.assertRaisesRegex(ValueError, "reach"):
            step2.interpolate_local_pops_inplace(raw, flatlines_for(raw, widen_samples=0))
        np.testing.assert_array_equal(raw.get_data(), popped)


if __name__ == "__main__":
    unittest.main()
