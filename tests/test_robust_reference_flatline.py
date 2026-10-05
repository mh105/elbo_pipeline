import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import spherical_spline as spline
import step2_bad_ch_time_detection as step2
from test_flatline_interpolation import flatlines_for, mne_reconstruction
from test_robust_average_reference import NAMES, make_native_raw


CRITERIA = {"bad_ch_by_voltage": "voltage", "bad_ch_by_HFnoise": "HFnoise",
            "bad_ch_by_pop": "pop", "bad_ch_by_smallness": "smallness",
            "bad_ch_by_std": "std", "bad_ch_by_BBnoise": "BBnoise",
            "bad_ch_by_correlation": "correlation", "bad_ch_by_RANSAC": "RANSAC"}
LOCAL_DETECTORS = {"bad_ch_by_HFnoise", "bad_ch_by_smallness", "bad_ch_by_std",
                   "bad_ch_by_BBnoise", "bad_ch_by_correlation", "bad_ch_by_RANSAC"}


def native_with_flatlines(global_dropout=False):
    raw = make_native_raw(seconds=8)
    raw.info["bads"] = ["T7", "EOG"]
    raw._data[NAMES.index("T7")] = np.nan
    raw._data[NAMES.index("F7"), 200:600] = 13e-6
    raw._data[NAMES.index("F3"), 400:800] = -7e-6
    spans = {"F7": [[200, 600]], "F3": [[400, 800]]}
    if global_dropout:
        for name in ["F4", "F8", "FC5", "FC1", "FC2", "FC6"]:
            spans[name] = [[450, 550]]
            raw._data[NAMES.index(name), 450:550] = 0
        raw.set_annotations(mne.Annotations([450 / 250], [100 / 250], ["BAD_flatline"]))
    return raw, flatlines_for(raw, spans)


class TestRobustReferenceFlatlines(unittest.TestCase):
    def test_forwards_local_masks_only_to_applicable_detectors(self):
        raw, flatlines = native_with_flatlines()
        with ExitStack() as stack:
            detectors = {name: stack.enter_context(patch.object(step2, name,
                           return_value={criterion: []})) for name, criterion in CRITERIA.items()}
            geometry = stack.enter_context(patch.object(step2, "_spline_geometry",
                                                         wraps=spline._spline_geometry))
            step2.robust_average_reference_inplace(raw, flatlines=flatlines)
        self.assertEqual(geometry.call_count, 1)
        for name, detector in detectors.items():
            self.assertEqual(detector.call_count, 1)
            kwargs = detector.call_args.kwargs
            if name in LOCAL_DETECTORS:
                self.assertIs(kwargs["flatlines"], flatlines)
            else:
                self.assertNotIn("flatlines", kwargs)
            if name == "bad_ch_by_RANSAC":
                self.assertIn("_geometry", kwargs)
                self.assertEqual(kwargs["_geometry"]["kernel"].dtype, np.dtype(np.float64))
            else:
                self.assertNotIn("_geometry", kwargs)

    def test_multiple_passes_reconstruct_joint_invalid_sets_and_preserve_dropouts(self):
        raw, flatlines = native_with_flatlines(global_dropout=True)
        original = raw.copy()
        expected = mne_reconstruction(original, ["T7", "F4"])
        intervals = [(200, 400, ["F7"]), (400, 450, ["F7", "F3"]),
                     (550, 600, ["F7", "F3"]), (600, 800, ["F3"])]
        for start, stop, local in intervals:
            repaired = mne_reconstruction(original, ["T7", "F4"] + local)
            rows = [raw.ch_names.index(name) for name in ["T7", "F4"] + local]
            expected._data[rows, start:stop] = repaired._data[rows, start:stop]

        def flag_f4(referenced, **kwargs):
            referenced.info["bads"] = list(dict.fromkeys(referenced.info["bads"] + ["F4"]))
            return {"HFnoise": ["F4"]}

        with ExitStack() as stack:
            detectors = {name: stack.enter_context(patch.object(step2, name,
                           return_value={criterion: []})) for name, criterion in CRITERIA.items()}
            detectors["bad_ch_by_HFnoise"].side_effect = flag_f4
            geometry = stack.enter_context(patch.object(step2, "_spline_geometry",
                                                         wraps=spline._spline_geometry))
            kernel = stack.enter_context(patch.object(spline, "_calc_g", wraps=spline._calc_g))
            report, bads, reference = step2.robust_average_reference_inplace(raw, flatlines=flatlines)
        self.assertEqual(geometry.call_count, 1)
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(detectors["bad_ch_by_HFnoise"].call_count, 2)
        self.assertEqual(report["HFnoise"], ["F4"])
        self.assertEqual(bads, ["F4", "T7"])
        self.assertEqual(raw.info["bads"], ["EOG"])
        eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        mean = expected._data[eeg].mean(axis=0)
        expected._data[eeg] -= mean
        np.testing.assert_allclose(reference, mean, rtol=0, atol=1e-15)
        np.testing.assert_allclose(raw._data[eeg], expected._data[eeg], rtol=0, atol=1e-15)
        np.testing.assert_allclose(raw._data[eeg].mean(axis=0), 0, rtol=0, atol=1e-18)
        np.testing.assert_array_equal(raw._data[NAMES.index("Cz")], -reference)
        np.testing.assert_array_equal(raw._data[-1], original._data[-1])
        self.assertEqual(raw.annotations, original.annotations)
        # Detector copies contain measured flat traces under their reference;
        # their eligibility mask carries the invalidity, rather than repaired data.
        for call in detectors["bad_ch_by_correlation"].call_args_list:
            referenced = call.args[0]
            np.testing.assert_allclose(referenced._data[NAMES.index("F7"), 200:600]
                                       - referenced._data[NAMES.index("Cz"), 200:600],
                                       13e-6, rtol=0, atol=1e-20)


if __name__ == "__main__":
    unittest.main()
