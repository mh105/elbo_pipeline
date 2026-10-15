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


class TestBadChannelTimeDetection(unittest.TestCase):
    def setUp(self):
        self.raw = mne.io.RawArray(
            np.array([[1e-6] * 50, [2e-6] * 50, [3e-6] * 50, [0.0] * 50]),
            mne.create_info(["F3", "C3", "P3", "Cz"], 1000, "eeg"), verbose=False,
        )
        self.detected_flatlines = {"segments_exact": [], "segments_bridged": [],
                                  "global_segments_exact": []}
        self.pop_intervals = {"C3": [(10, 15)], "Cz": [(20, 25)]}
        self.asr_details = {"corrected_duration_s": 0.01}
        results = {
            "bad_ch_by_nan": {"nan": []},
            "detect_flatline_inplace": self.detected_flatlines,
            "bad_ch_by_flatline": {"flatline": []},
            "bad_ch_by_voltage": {"voltage": ["C3"]},
            "bad_ch_by_HFnoise": {"HFnoise": []},
            "bad_ch_by_pop": {"pop": []},
            "bad_ch_by_small": {"small": []},
            "plot_bad_ch_report": None,
            "bad_ch_under_robust_ref": {"correlation": []},
            "interpolate_local_artifacts_inplace": self.pop_intervals,
            "validate_reference_inplace": "F3",
            "apply_asr_inplace": self.asr_details,
        }
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.steps = {
            name: stack.enter_context(patch.object(step2, name, return_value=result))
            for name, result in results.items()
        }
        self.filter = stack.enter_context(patch.object(mne.io.BaseRaw, "filter"))

    def test_supplied_inputs_are_reused_without_initial_detection(self):
        supplied_report = {"flatline": ["P3"]}
        supplied_flatlines = self.detected_flatlines
        report, details, reference = step2.bad_ch_time_detection_inplace(
            self.raw, supplied_report, supplied_flatlines,
        )
        self.assertIs(report, supplied_report)
        self.assertIs(details, self.asr_details)
        self.assertEqual(reference, "F3")
        self.assertEqual(report["flatline"], ["P3"])
        self.assertNotIn("nan", report)
        for name in ("bad_ch_by_nan", "detect_flatline_inplace", "bad_ch_by_flatline"):
            self.steps[name].assert_not_called()
        for name in ("bad_ch_by_voltage", "bad_ch_by_HFnoise",
                     "bad_ch_by_pop", "bad_ch_by_small"):
            call = self.steps[name].call_args
            self.assertIs(call.args[0], self.raw)
            self.assertIs(call.kwargs["flatlines"], supplied_flatlines)
        self.assertIs(self.steps["apply_asr_inplace"].call_args.args[2], supplied_flatlines)
        self.assertIs(self.steps["apply_asr_inplace"].call_args.args[3],
                      self.pop_intervals)
        self.steps["interpolate_local_artifacts_inplace"].assert_called_once_with(
            self.raw, supplied_flatlines,
        )
        self.filter.assert_not_called()

    def test_preserves_supplied_report_and_appends_robust_results_by_criterion(self):
        report = {"impedance": ["P3"], "nan": ["C3"]}
        flatlines = {}
        self.steps["bad_ch_under_robust_ref"].return_value = {
            "nan": ["F3"], "voltage": ["P3"], "correlation": ["C3"],
        }
        plotted = []
        self.steps["plot_bad_ch_report"].side_effect = (
            lambda raw, current_report: plotted.append(
                {criterion: names.copy() for criterion, names in current_report.items()}
            )
        )

        returned_report, _, _ = step2.bad_ch_time_detection_inplace(self.raw, report, flatlines)

        self.assertIs(returned_report, report)
        self.assertEqual(report["impedance"], ["P3"])
        self.assertEqual(report["nan"], ["C3", "F3"])
        self.assertEqual(report["voltage"], ["C3", "P3"])
        self.assertEqual(report["correlation"], ["C3"])
        self.steps["plot_bad_ch_report"].assert_called_once_with(self.raw, report)
        self.assertEqual(plotted[0]["nan"], ["C3"])
        self.assertEqual(plotted[0]["voltage"], ["C3"])
        self.assertNotIn("correlation", plotted[0])
        self.filter.assert_not_called()

    def test_both_initial_inputs_are_required_before_any_processing(self):
        for kwargs in ({}, {"bad_ch_report": {}}, {"flatlines": {}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                step2.bad_ch_time_detection_inplace(self.raw, **kwargs)
        for mock in self.steps.values():
            mock.assert_not_called()
        self.filter.assert_not_called()

    def test_same_raw_carries_cleaning_changes_and_validated_reference_to_asr(self):
        report, flatlines = {}, {}
        progression = []

        def mark_voltage(raw, *, flatlines):
            self.assertIs(raw, self.raw)
            raw.info["bads"] = ["C3"]
            raw.set_annotations(mne.Annotations([0.01], [0.001], ["BAD_voltage_mask"]))
            return {"voltage": ["C3"]}

        def interpolate(raw, spans):
            self.assertIs(raw, self.raw)
            self.assertIs(spans, flatlines)
            np.testing.assert_array_equal(raw.get_data(picks=["Cz"]), 0)
            raw._data[2, 0] = 5e-6
            progression.append("artifacts")
            return self.pop_intervals

        def validate(raw, spans):
            self.assertEqual(progression, ["artifacts"])
            self.assertIs(raw, self.raw)
            self.assertIs(spans, flatlines)
            raw._data -= raw._data[0].copy()
            progression.append("validate")
            return "F3"

        def apply_asr(raw, reference, spans, pop_intervals):
            self.assertEqual(progression, ["artifacts", "validate"])
            self.assertIs(raw, self.raw)
            self.assertEqual(reference, "F3")
            self.assertIs(spans, flatlines)
            self.assertIs(pop_intervals, self.pop_intervals)
            self.assertEqual(raw.info["bads"], ["C3"])
            np.testing.assert_array_equal(raw.get_data(picks=[reference]), 0)
            raw._data[2] *= 0.5
            progression.append("asr")
            return self.asr_details

        self.steps["bad_ch_by_voltage"].side_effect = mark_voltage
        self.steps["interpolate_local_artifacts_inplace"].side_effect = interpolate
        self.steps["validate_reference_inplace"].side_effect = validate
        self.steps["apply_asr_inplace"].side_effect = apply_asr
        returned_report, details, reference = step2.bad_ch_time_detection_inplace(
            self.raw, report, flatlines,
        )

        self.assertIs(returned_report, report)
        self.assertIs(details, self.asr_details)
        self.assertEqual(reference, "F3")
        self.assertEqual(progression, ["artifacts", "validate", "asr"])
        self.assertAlmostEqual(self.raw._data[2, 0], 2e-6)
        np.testing.assert_allclose(self.raw._data[2, 1:], 1e-6, rtol=0, atol=1e-20)
        self.assertEqual(self.raw.info["bads"], ["C3"])
        self.assertEqual(list(self.raw.annotations.description), ["BAD_voltage_mask"])
        self.filter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
