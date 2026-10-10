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

        self.assertIsNone(result)
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

        self.assertIsNone(result)
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

        self.assertIsNone(result)
        row = raw.ch_names.index("C3")
        np.testing.assert_array_equal(raw._data[row, 300:350], original[row, 300:350])
        np.testing.assert_array_equal(raw._data[row, 450:], original[row, 450:])
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

        self.assertIsNone(result)
        self.assertEqual(raw.info["bads"], [])
        row = raw.ch_names.index("C3")
        for start, stop in [(750, 1250), (2250, 2750), (3750, 4250)]:
            expected, _ = expected_repair(raw, original, "C3", start, stop, {"C3"})
            np.testing.assert_allclose(raw._data[row, start:stop], expected,
                                       rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw._data[row, 1250:2250], original[row, 1250:2250])

    def test_requires_the_native_reference(self):
        raw = make_raw()
        add_pop(raw, "C3", 2500)
        flatlines = flatlines_for(raw)
        eeg = slice(0, len(NAMES))
        average = raw.copy()  # average reference: Cz is not entirely zero
        average._data[eeg] -= average._data[eeg].mean(axis=0)
        moved = raw.copy()  # already re-referenced from Cz to FC1
        moved._data[eeg] -= moved._data[NAMES.index("FC1")].copy()
        unreplaceable = raw.copy()  # a bad Cz whose five nearest are bad too
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        unreplaceable.info["bads"] = [
            "Cz", *step2._nearest_channels(raw, picks, n_neighbours=5)[1]["Cz"],
        ]
        for name, data, message in [("average", average, "not entirely zero"),
                                    ("moved", moved, "not entirely zero"),
                                    ("unreplaceable", unreplaceable, "No valid native reference")]:
            with self.subTest(name=name):
                original = data.get_data()
                with self.assertRaisesRegex(ValueError, message):
                    step2.interpolate_local_pops_inplace(data, flatlines)
                np.testing.assert_array_equal(data.get_data(), original)

    def test_repairs_a_good_native_references_own_pops(self):
        raw = make_raw()
        cz, c3 = NAMES.index("Cz"), NAMES.index("C3")
        others = [row for row in range(len(NAMES)) if row != cz]
        # Cz's electrode steps by +400 uV. Unlike the high-passed data, the
        # synthetic step persists; shared by every channel, 500 uV would trip
        # the global voltage mask.
        raw._data[others, 2500:] -= 400e-6
        add_pop(raw, "FC1", 2600)  # inside Cz's repair interval
        add_pop(raw, "C3", 4000)
        popped = raw.get_data()

        step2.interpolate_local_pops_inplace(raw, flatlines_for(raw))

        # Cz's estimate, without FC1 as a donor and faded in and out, becomes
        # the reference: it is subtracted from every EEG channel, and Cz stays
        # zero. The other pops are then repaired on it.
        reference, _ = expected_repair(raw, popped, "Cz", 2250, 2750, {"Cz", "FC1"})
        expected = popped.copy()
        expected[:len(NAMES), 2250:2750] -= reference
        expected[cz] = 0
        rereferenced = expected.copy()
        for name, start, stop in [("FC1", 2350, 2850), ("C3", 3750, 4250)]:
            expected[raw.ch_names.index(name), start:stop], _ = expected_repair(
                raw, rereferenced, name, start, stop, {name},
            )
        np.testing.assert_allclose(raw._data, expected, rtol=0, atol=1e-15)
        for span in (np.s_[:2250], np.s_[2850:3750], np.s_[4250:]):
            np.testing.assert_array_equal(raw._data[:, span], popped[:, span])
        # Cz's step has left every channel.
        level = (raw._data[:len(NAMES), 2510:2560].mean(axis=1)
                 - raw._data[:len(NAMES), 2440:2490].mean(axis=1))
        self.assertLess(np.abs(level).max(), 30e-6)

    def test_bad_native_reference_has_its_replacements_pops_repaired_first(self):
        raw = make_raw()
        raw.info["bads"] = ["Cz"]
        replacement, _ = step2._candidate_reference_replacement(raw, "Cz")
        cz = NAMES.index("Cz")
        others = [row for row in range(len(NAMES)) if row != cz]
        raw._data[others, 2500:] += 300e-6  # Cz's electrode steps by -300 uV ...
        add_pop(raw, "C3", 2500)  # ... and hides C3's pop on the Cz reference.
        add_pop(raw, replacement, 1000, step_uv=400.0)  # as Cz's in the test above
        native = raw.get_data()
        flatlines = flatlines_for(raw)
        _, on_cz = step2.bad_ch_by_pop(raw.copy(), flatlines=flatlines, return_details=True)
        self.assertEqual(len(on_cz["pop_times_s"]["C3"]), 0)

        step2.interpolate_local_pops_inplace(raw, flatlines)

        # raw stays on the bad Cz; the replacement's pop and C3's are repaired.
        expected = native.copy()
        for name, start, stop in [(replacement, 750, 1250), ("C3", 2250, 2750)]:
            row = raw.ch_names.index(name)
            expected[row, start:stop], _ = expected_repair(raw, native, name, start, stop,
                                                           {name})
        np.testing.assert_allclose(raw._data, expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw._data[cz], 0)
        self.assertEqual(raw.info["bads"], ["Cz"])
        # On the repaired replacement, no pop is left, not even Cz's.
        with self.assertWarns(UserWarning):
            self.assertEqual(step2.validate_reference_inplace(raw, flatlines), replacement)
        _, left = step2.bad_ch_by_pop(raw, flatlines=flatlines, return_details=True)
        self.assertEqual(sum(map(len, left["pop_times_s"].values())), 0)

    def test_without_pops_or_reach_the_data_are_unchanged(self):
        raw = make_raw()
        original = raw.get_data()
        result = step2.interpolate_local_pops_inplace(raw, flatlines_for(raw))
        self.assertIsNone(result)
        np.testing.assert_array_equal(raw.get_data(), original)

        add_pop(raw, "C3", 2500)
        popped = raw.get_data()
        with self.assertRaisesRegex(ValueError, "reach"):
            step2.interpolate_local_pops_inplace(raw, flatlines_for(raw, widen_samples=0))
        np.testing.assert_array_equal(raw.get_data(), popped)


if __name__ == "__main__":
    unittest.main()
