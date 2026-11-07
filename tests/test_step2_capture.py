"""Synthetic tests for frozen Step 2 replay."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
from mne_denoise.asr import process_asr
from mne_denoise.asr._types import ASRState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step2_bad_ch_time_detection as step2
from step2_capture import Step2Capture, filter_step2_inplace
from test_asr import make_raw, flatlines_for


class TestCapture(unittest.TestCase):
    def roundtrip(self, capture):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step2.npz"
            capture.save(path)
            return Step2Capture.load(path)

    def test_filter_edges_channel_picks_and_skipped_asr(self):
        raw = make_raw(seconds=32)
        raw.set_annotations(mne.Annotations([15.0], [1.0], ["BAD_ACQ_SKIP"]))
        raw.info["bads"] = ["F3"]
        x = raw.get_data()
        ordinary = raw.copy().filter(.3, None, verbose=False)
        capture = Step2Capture(raw, "Cz")
        filter_step2_inplace(raw, capture=capture)
        np.testing.assert_array_equal(raw.get_data(), ordinary.get_data())
        capture.skip_asr("Skipped by caller")
        loaded = self.roundtrip(capture)
        self.assertTrue(loaded.stage_availability["filter_only"])
        self.assertFalse(loaded.stage_availability["repair_increment"])
        self.assertEqual(loaded.first_samp, 1000)
        self.assertEqual(len(loaded.operations[0]["bounds"]), 2)
        np.testing.assert_allclose(loaded.replay(x), raw.get_data(), atol=1e-15, rtol=1e-12)
        np.testing.assert_array_equal(raw.get_data()[-1], x[-1])  # EOG
        with self.assertRaisesRegex(ValueError, "Skipped by caller"):
            loaded.replay(x, stage="asr_only")

    def test_missing_filter_and_observed_empty_repairs_are_distinguishable(self):
        raw = make_raw(seconds=12)
        cap = Step2Capture(raw, "Cz")
        flat = flatlines_for(raw, {})
        step2.interpolate_local_artifacts_inplace(raw, flat, pop_intervals={}, capture=cap)
        step2.validate_reference_inplace(raw, flat, capture=cap)
        cap.skip_asr("insufficient calibration")
        status = self.roundtrip(cap).stage_availability
        self.assertEqual(status, {"filter_only": False, "repair_increment": True,
                                  "asr_increment": False, "whole_step2": False})
        self.assertFalse(any(op["kind"] == "repair" for op in cap.operations))

    def test_repairs_donor_transitions_and_reference(self):
        raw = make_raw(seconds=12)
        raw.info["bads"] = ["Cz"]
        flat = flatlines_for(raw, {})
        spans = {"F3": [(400, 1100)], "F7": [(800, 1300)]}
        capture = Step2Capture(raw, "Cz")
        def repair(r, cap):
            step2.interpolate_local_artifacts_inplace(r, flat, pop_intervals=spans, capture=cap)
            return step2.validate_reference_inplace(r, flat, capture=cap)
        ordinary = raw.copy()
        ref = repair(ordinary, None)
        x = raw.get_data()
        self.assertEqual(repair(raw, capture), ref)
        self.assertNotEqual(ref, "Cz")
        self.assertEqual(capture.output_reference, ref)
        np.testing.assert_array_equal(raw.get_data(), ordinary.get_data())
        ops = capture.operations
        self.assertTrue(any(op.get("other_weights") is not None for op in ops))
        np.testing.assert_allclose(self.roundtrip(capture).replay(x, stage="pre_asr"),
                                   raw.get_data(), atol=1e-15, rtol=1e-12)

    def test_asr_first_and_last_crossfades_on_independent_probes(self):
        # Odd window length, a hidden predecessor, and a short last block.
        sf = 250
        n = 731
        rng = np.random.default_rng(95)
        x = rng.normal(size=(4, n)) * 4e-6
        x[0] *= 45
        eye = np.eye(4)
        state = ASRState(M=eye, T=eye*2e-6, thresholds=np.full(4, 2e-6),
                         calibration_patterns=eye, filter_b=np.ones(1), filter_a=np.ones(1),
                         cov=eye, rank=4)
        raw = mne.io.RawArray(x, mne.create_info(4, sf, "eeg"), first_samp=703, verbose=False)
        ordinary, _ = process_asr(x, sf, state)
        actual, diag = process_asr(x, sf, state, store_reconstruction_matrices=True)
        np.testing.assert_array_equal(actual, ordinary)
        self.assertTrue(diag["sample_mask"][[0, -1]].all())
        cap = Step2Capture(raw, None)
        cap.record_asr(x, actual, np.arange(4), state, diag, process_asr)
        loaded = self.roundtrip(cap)
        first, last = loaded.operations[0], loaded.operations[-1]
        self.assertEqual((first["start"], last["stop"]), (0, n))
        self.assertFalse(np.allclose(first["previous"], eye))
        self.assertLess(last["blend"].size, diag["stepsize_samples"])
        step = diag["stepsize_samples"]
        phase = diag["lookahead_samples"] % step + 1
        self.assertAlmostEqual(first["blend"][0], (1 - np.cos(np.pi * phase / step)) / 2)
        self.assertEqual(last["blend"][-1], 1.)
        np.testing.assert_allclose(loaded.replay(x), ordinary, atol=1e-15, rtol=1e-12)
        # Applying a fixed transformation is linear even at the clipped edges.
        probe = rng.normal(size=x.shape)*1e-6
        np.testing.assert_allclose(loaded.replay(x + probe) - loaded.replay(x),
                                   loaded.replay(probe, stage="asr_only"), atol=1e-17)

    def test_native_asr_no_correction_vs_insufficient_support(self):
        raw = make_raw(seconds=80)
        x = raw.get_data()
        cap = Step2Capture(raw, "Cz")
        with patch.object(step2, "plot_asr_calibration"), patch.object(step2, "plot_asr_effect"), \
                patch.object(mne.time_frequency.Spectrum, "plot_topomap"):
            flat = flatlines_for(raw, {})
            plain = raw.copy()
            step2.apply_asr_inplace(plain, "Cz", flat, {})
            details = step2.apply_asr_inplace(raw, "Cz", flat, {}, capture=cap)
        self.assertEqual(cap.asr_status, "applied")
        self.assertNotIn("reconstruction_matrices", details["processing"])
        np.testing.assert_array_equal(plain.get_data(), raw.get_data())
        np.testing.assert_allclose(cap.replay(x), plain.get_data(), atol=1e-15)
        short = make_raw(seconds=8)
        skip = Step2Capture(short, "Cz")
        with self.assertRaisesRegex(ValueError, "60 seconds"):
            step2.apply_asr_inplace(short, "Cz", flatlines_for(short, {}), {}, capture=skip)
        self.assertEqual(skip.asr_status, "skipped")
        np.testing.assert_array_equal(skip.replay(short.get_data()), short.get_data())

    def test_applied_identity_is_available_for_asr_only_replay(self):
        raw = make_raw(seconds=1)
        cap = Step2Capture(raw, "Cz")
        picks = step2._mask_eeg_picks(raw, "Cz")
        data = raw.get_data(picks=picks)
        cap.record_asr(data, data, picks, None, {"sample_mask": np.zeros(raw.n_times, bool)}, None)
        loaded = self.roundtrip(cap)
        self.assertEqual(loaded.asr_status, "applied")
        np.testing.assert_array_equal(loaded.replay(raw.get_data(), stage="asr_only"), raw.get_data())

    def test_unavailable_capture_preserves_native_asr_output(self):
        from test_asr import FakeASR
        raw = make_raw(seconds=75)
        cap = Step2Capture(raw, "Cz")
        fake = FakeASR()
        with patch.object(step2, "calibrate_asr", side_effect=fake.calibrate), \
                patch.object(step2, "process_asr", side_effect=fake.process), \
                patch.object(step2, "plot_asr_calibration"), patch.object(step2, "plot_asr_effect"), \
                patch.object(mne.time_frequency.Spectrum, "plot_topomap"), \
                self.assertWarnsRegex(UserWarning, "replay is unavailable"):
            step2.apply_asr_inplace(raw, "Cz", flatlines_for(raw, {}), {}, capture=cap)
        self.assertEqual(cap.asr_status, "unavailable")  # fake has no reconstruction matrices
        np.testing.assert_array_equal(raw.get_data(picks=step2._mask_eeg_picks(raw, "Cz")), fake.corrected)
        with self.assertRaisesRegex(ValueError, "capture unavailable"):
            cap.replay(raw.get_data())


if __name__ == "__main__":
    unittest.main()
