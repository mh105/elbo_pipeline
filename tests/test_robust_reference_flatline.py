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

import step2_bad_ch_time_detection as step2
from test_flatline_interpolation import flatlines_for, mne_reconstruction
from test_robust_average_reference import NAMES, make_native_raw


CRITERIA = {"bad_ch_by_voltage": ("voltage",), "bad_ch_by_HFnoise": ("HFnoise",),
            "bad_ch_by_pop": ("pop",), "bad_ch_by_std": ("std_high", "std_low"),
            "bad_ch_by_BBnoise": ("BBnoise",), "bad_ch_by_correlation": ("correlation",),
            "bad_ch_by_RANSAC": ("RANSAC",)}


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
    flatlines = flatlines_for(raw, spans)
    if global_dropout:
        step2._mask_segment_by_flatline(raw, flatlines)  # Annotate raw, as detection does.
    return raw, flatlines


class TestRobustReferenceFlatlines(unittest.TestCase):
    def test_forwards_local_masks_to_every_detector(self):
        raw, flatlines = native_with_flatlines()
        with ExitStack() as stack:
            detectors = {name: stack.enter_context(patch.object(step2, name,
                           return_value={criterion: [] for criterion in criteria}))
                         for name, criteria in CRITERIA.items()}
            geometry = stack.enter_context(patch.object(step2, "_spline_geometry",
                                                         wraps=step2._spline_geometry))
            report = step2.robust_average_reference_inplace(raw, flatlines=flatlines)
        self.assertEqual(geometry.call_count, 1)
        self.assertEqual(report, {criterion: [] for criteria in CRITERIA.values()
                                  for criterion in criteria})
        for name, detector in detectors.items():
            self.assertEqual(detector.call_count, 1)
            kwargs = detector.call_args.kwargs
            self.assertIs(kwargs["flatlines"], flatlines)
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
            return {"HFnoise": ["F4"]}, {}

        with ExitStack() as stack:
            detectors = {name: stack.enter_context(patch.object(step2, name,
                           return_value=({criterion: [] for criterion in criteria}, {})))
                         for name, criteria in CRITERIA.items()}
            detectors["bad_ch_by_HFnoise"].side_effect = flag_f4
            geometry = stack.enter_context(patch.object(step2, "_spline_geometry",
                                                         wraps=step2._spline_geometry))
            kernel = stack.enter_context(patch.object(step2, "_calc_g", wraps=step2._calc_g))
            report, details = step2.robust_average_reference_inplace(
                raw, flatlines=flatlines, return_details=True,
            )
        reference = details["final_reference"]
        self.assertEqual(geometry.call_count, 1)
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(detectors["bad_ch_by_HFnoise"].call_count, 2)
        self.assertEqual(details["initial_bads"], ["T7"])
        self.assertEqual(report["HFnoise"], ["F4"])
        self.assertEqual(set().union(*report.values()), {"F4"})
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
