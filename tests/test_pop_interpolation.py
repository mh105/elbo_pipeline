import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7",
         "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
         "P8", "PO3", "PO4", "O1", "Oz", "O2"]


def make_raw(n_times=5000, sfreq=250):  # 20 s of native-reference noise by default
    rng = np.random.default_rng(311)
    data = rng.normal(scale=10e-6, size=(len(NAMES) + 1, n_times))
    data[:len(NAMES)] -= data[NAMES.index("Cz")]
    raw = mne.io.RawArray(data, mne.create_info(NAMES + ["EOG"], sfreq,
                                              ["eeg"] * len(NAMES) + ["eog"]), verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("colin27_1020"), verbose=False)
    return raw


def flatlines_for(raw, spans=None, global_segments=()):
    """Prescribed channel spans without filter reach metadata."""
    spans = spans or {}
    return {
        "ch_names": list(spans),
        "segments_exact": [np.asarray(value, dtype=int).reshape(-1, 2) for value in spans.values()],
        "segments_bridged": [np.asarray(value, dtype=int).reshape(-1, 2) for value in spans.values()],
        "raw_ch_names": list(raw.ch_names), "eeg_ch_names": NAMES.copy(),
        "sfreq": float(raw.info["sfreq"]), "first_samp": int(raw.first_samp),
        "n_times": int(raw.n_times), "reference_channel": "Cz",
        "global_segments_exact": [list(bounds) for bounds in global_segments],
    }


def add_pop(raw, name, onset, step_uv=500.0):
    """A sustained step starting at sample onset."""
    raw._data[raw.ch_names.index(name), onset:] += step_uv * 1e-6


def spline_estimate(raw, original, name, start, stop, invalid):
    """Project the prescribed good donor set without changing its samples."""
    geometry = step2._spline_geometry(raw)
    donors = [i for i in geometry["picks"] if raw.ch_names[i] not in raw.info["bads"]
              and raw.ch_names[i] not in invalid]
    target = raw.ch_names.index(name)
    return (step2._spline_weights(geometry, donors, [target])
            @ original[donors, start:stop])[0]


def crossfade(before, after):
    rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(len(before)) + 0.5) / len(before))
    return (1 - rise) * before + rise * after


def expected_repair(raw, original, name, start, stop, invalid):
    """Spline estimate from the good donors, blended in by 0.25-s raised cosines
    at both ends."""
    estimate = spline_estimate(raw, original, name, start, stop, invalid)
    target = raw.ch_names.index(name)
    fade = min(int(round(0.25 * raw.info["sfreq"])), (stop - start) // 2)
    weight = np.ones(stop - start)
    rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)
    weight[:fade] = rise
    weight[stop - start - fade:] = rise[::-1]
    data = original[target, start:stop]
    return weight * estimate + (1 - weight) * data, estimate


class TestLocalPopInterpolation(unittest.TestCase):
    def test_repairs_one_second_on_each_side_of_a_pop_and_nothing_else(self):
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
        # P3 is unavailable only inside its flat span; O1 donates throughout.
        flatlines = flatlines_for(raw, {"P3": [[2700, 2800]], "O1": [[4000, 4100]]})

        result = step2.interpolate_local_pops_inplace(raw, flatlines)

        self.assertIsNone(result)
        expected = original.copy()
        for name, start, stop in [("C3", 2250, 2750), ("C4", 2350, 2850)]:
            row = raw.ch_names.index(name)
            alone = spline_estimate(raw, original, name, start, stop, {name})
            both = spline_estimate(raw, original, name, start, stop, {"C3", "C4"})
            no_flat = spline_estimate(raw, original, name, start, stop,
                                      {"C3", "C4", "P3"})
            if name == "C3":
                expected[row, start:stop] = both
                expected[row, 2250:2300] = crossfade(original[row, 2250:2300], alone[:50])
                # C4 fades out as a donor BEFORE its repair interval starts.
                expected[row, 2300:2350] = crossfade(alone[50:100], both[50:100])
                expected[row, 2638:2700] = crossfade(both[388:450], no_flat[388:450])
                expected[row, 2700:2725] = no_flat[450:475]
                expected[row, 2725:2750] = crossfade(no_flat[475:],
                                                    original[row, 2725:2750])
            else:
                after_c3 = spline_estimate(raw, original, name, start, stop, {name, "P3"})
                expected[row, start:stop] = both
                expected[row, 2350:2412] = crossfade(original[row, 2350:2412], both[:62])
                expected[row, 2638:2700] = crossfade(both[288:350], no_flat[288:350])
                expected[row, 2700:2750] = no_flat[350:400]
                # C3 returns only after its target span, P3 after its invalid span.
                expected[row, 2750:2775] = crossfade(no_flat[400:425], after_c3[400:425])
                expected[row, 2775:2800] = after_c3[425:450]
                expected[row, 2800:2825] = crossfade(after_c3[450:475], alone[450:475])
                expected[row, 2825:2850] = crossfade(alone[475:], original[row, 2825:2850])
        np.testing.assert_allclose(raw._data, expected, rtol=0, atol=1e-15)

    def test_invalid_only_samples_never_donate_and_are_not_repaired(self):
        raw = make_raw()
        original = raw.get_data()
        changed = raw.copy()
        invalid_row = raw.ch_names.index("P3")
        changed._data[invalid_row, 800:1100] += 1e6
        changed_input = changed.get_data()
        for data in (raw, changed):
            step2._interpolate_local_intervals_inplace(
                data, {"C3": [[500, 1500]]}, {"P3": [[800, 1100]]}, [],
            )

        row = raw.ch_names.index("C3")
        with_p3 = spline_estimate(raw, original, "C3", 500, 1500, {"C3"})
        without_p3 = spline_estimate(raw, original, "C3", 500, 1500, {"C3", "P3"})
        expected = with_p3.copy()
        expected[:62] = crossfade(original[row, 500:562], with_p3[:62])
        expected[238:300] = crossfade(with_p3[238:300], without_p3[238:300])
        expected[300:600] = without_p3[300:600]
        expected[600:662] = crossfade(without_p3[600:662], with_p3[600:662])
        expected[938:] = crossfade(with_p3[938:], original[row, 1438:1500])
        np.testing.assert_allclose(raw._data[row, 500:1500], expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw._data[row], changed._data[row])
        others = [i for i in range(len(raw.ch_names)) if i != row]
        np.testing.assert_array_equal(raw._data[others], original[others])
        np.testing.assert_array_equal(changed._data[others], changed_input[others])
        np.testing.assert_array_equal(raw._data[row, :500], original[row, :500])
        np.testing.assert_array_equal(raw._data[row, 1500:], original[row, 1500:])

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
        # Both ends blend with the input, including the recording and dropout boundaries.
        for start, stop in [(0, 300), (350, 450)]:
            expected, _ = expected_repair(raw, original, "C3", start, stop, {"C3"})
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

        # Cz uses FC1 only before FC1's repair interval, with the donor fade
        # completed at 2350. Its repaired trace is subtracted from every EEG
        # channel, keeping Cz zero; the other pops are then repaired on it.
        with_fc1 = spline_estimate(raw, popped, "Cz", 2250, 2750, {"Cz"})
        without_fc1 = spline_estimate(raw, popped, "Cz", 2250, 2750, {"Cz", "FC1"})
        reference = without_fc1.copy()
        reference[:50] = crossfade(popped[cz, 2250:2300], with_fc1[:50])
        reference[50:100] = crossfade(with_fc1[50:100], without_fc1[50:100])
        reference[-62:] = crossfade(without_fc1[-62:], popped[cz, 2688:2750])
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

    def test_without_pops_the_data_are_unchanged(self):
        raw = make_raw()
        original = raw.get_data()
        result = step2.interpolate_local_pops_inplace(raw, flatlines_for(raw))
        self.assertIsNone(result)
        np.testing.assert_array_equal(raw.get_data(), original)


class TestPopRepairWidth(unittest.TestCase):
    def test_default_and_custom_widths_are_independent_of_filters(self):
        for sfreq in (250, 500, 1000):
            for half_seconds, kwargs in [(1, {}), (2, {"repair_width": 4.0})]:
                with self.subTest(sfreq=sfreq, half_seconds=half_seconds), patch.object(
                    step2.mne.filter, "create_filter",
                    side_effect=AssertionError("Pop repair must not construct a filter."),
                ):
                    raw = make_raw(n_times=20 * sfreq, sfreq=sfreq)
                    onset, half_width = 10 * sfreq, half_seconds * sfreq
                    add_pop(raw, "C3", onset)
                    original = raw.get_data()
                    result = step2.interpolate_local_pops_inplace(raw, flatlines_for(raw), **kwargs)
                    self.assertIsNone(result)
                    start, stop = onset - half_width, onset + half_width
                    expected, _ = expected_repair(raw, original, "C3", start, stop, {"C3"})
                    row = raw.ch_names.index("C3")
                    np.testing.assert_allclose(raw._data[row, start:stop], expected,
                                               rtol=0, atol=1e-15)
                    np.testing.assert_array_equal(raw._data[row, :start], original[row, :start])
                    np.testing.assert_array_equal(raw._data[row, stop:], original[row, stop:])
                    others = [i for i in range(len(raw.ch_names)) if i != row]
                    np.testing.assert_array_equal(raw._data[others], original[others])

    def test_invalid_widths_raise_without_changing_data(self):
        for repair_width in (0, -1, np.nan, np.inf, 0.001):
            with self.subTest(repair_width=repair_width):
                raw = make_raw()
                add_pop(raw, "C3", 2500)
                original = raw.get_data()
                with self.assertRaisesRegex(ValueError, "repair_width"):
                    step2.interpolate_local_pops_inplace(
                        raw, flatlines_for(raw), repair_width=repair_width,
                    )
                np.testing.assert_array_equal(raw.get_data(), original)


if __name__ == "__main__":
    unittest.main()
