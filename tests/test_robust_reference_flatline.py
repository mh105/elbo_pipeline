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
from test_flatline_interpolation import blended_reconstruction, flatlines_for, overlapping_parts
from test_robust_average_reference import NAMES, make_native_raw


CRITERIA = {"bad_ch_by_voltage": ("voltage",), "bad_ch_by_HFnoise": ("HFnoise",),
            "bad_ch_by_pop": ("pop",), "bad_ch_by_std": ("std",),
            "bad_ch_by_BBnoise": ("BBnoise",), "bad_ch_by_correlation": ("correlation",)}


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
    def setUp(self):
        plot_report = patch.object(step2, "plot_bad_ch_report")
        plot_report.start()
        self.addCleanup(plot_report.stop)

    def test_forwards_local_masks_to_every_detector(self):
        raw, flatlines = native_with_flatlines()
        with ExitStack() as stack:
            detectors = {name: stack.enter_context(patch.object(step2, name,
                           return_value={criterion: [] for criterion in criteria}))
                         for name, criteria in CRITERIA.items()}
            geometry = stack.enter_context(patch.object(step2, "_spline_geometry",
                                                         wraps=step2._spline_geometry))
            report = step2.bad_ch_under_robust_ref(raw, flatlines=flatlines)
        self.assertEqual(geometry.call_count, 1)
        self.assertEqual(report, {criterion: [] for criteria in CRITERIA.values()
                                  for criterion in criteria})
        for name, detector in detectors.items():
            self.assertEqual(detector.call_count, 1)
            kwargs = detector.call_args.kwargs
            self.assertIs(kwargs["flatlines"], flatlines)
            self.assertNotIn("_geometry", kwargs)

    def test_multiple_passes_reference_joint_invalid_sets_without_changing_data(self):
        raw, flatlines = native_with_flatlines(global_dropout=True)
        original = raw.copy()
        expected = blended_reconstruction(original, ["T7", "F4"], overlapping_parts(True))

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
            report, details = step2.bad_ch_under_robust_ref(
                raw, flatlines=flatlines, return_details=True,
            )
        reference = details["final_reference"]
        self.assertEqual(geometry.call_count, 1)
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(detectors["bad_ch_by_HFnoise"].call_count, 2)
        self.assertEqual(details["initial_bads"], ["T7"])
        self.assertEqual(report["HFnoise"], ["F4"])
        self.assertEqual(set().union(*report.values()), {"F4"})
        self.assertEqual(raw.info["bads"], ["T7", "EOG", "F4"])
        eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        np.testing.assert_allclose(reference, expected._data[eeg].mean(axis=0),
                                   rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
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
