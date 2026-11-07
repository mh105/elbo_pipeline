"""Synthetic regressions for Step 2's recoverable decisions and saved handoff."""
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step2_bad_ch_time_detection as step2
from step2_capture import Step2Capture, filter_step2_inplace
from test_asr import FakeASR, flatlines_for, make_raw


@pytest.fixture
def quiet_figures(monkeypatch):
    for name in ("plot_bad_ch_report", "plot_voltage_traces", "plot_asr_effect",
                 "plot_asr_calibration"):
        monkeypatch.setattr(step2, name, lambda *a, **k: None)
    monkeypatch.setattr(mne.time_frequency.Spectrum, "plot_topomap", lambda *a, **k: None)


def test_offset_handling_only_changes_flatness_screen():
    raw = make_raw(seconds=8)
    flats = flatlines_for(raw, {"F3": [(100, 999)]})
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg = [i for i in eeg if raw.ch_names[i] != "Cz"]
    raw._data[eeg] += .012
    before = raw.get_data()
    with pytest.raises(step2.NoScorableSupport):
        step2.bad_ch_by_flatline(raw, flatlines=flats, voltage_mask=True)
    result = step2.assess_persistent_flatness(raw, flatlines=flats, voltage_mask=True)
    assert result == {"flatline": ["F3"]}
    assert flats["persistent_flatness"]["status"] == "completed"
    np.testing.assert_array_equal(raw.get_data(), before)
    raw._data[eeg] += .002 * np.sin(2*np.pi*32*raw.times)
    before = raw.get_data()
    with pytest.warns(RuntimeWarning, match="Persistent flatness"):
        assert step2.assess_persistent_flatness(raw, flatlines=flats, voltage_mask=True) == {}
    assert flats["persistent_flatness"]["status"] == "unavailable"
    assert flats["segments_bridged"][0].tolist() == [[100, 999]]
    assert raw.info["bads"] == ["F3"]
    np.testing.assert_array_equal(raw.get_data(), before)
    misaligned = dict(flats, first_samp=-1)
    with pytest.raises(ValueError, match="alignment"):
        step2.assess_persistent_flatness(raw, flatlines=misaligned)


def noisy_reference():
    names = "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Cz".split()
    montage = mne.channels.make_standard_montage("colin27_1020")
    xyz = np.array([montage.get_positions()["ch_pos"][name] for name in names])
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    weights = np.column_stack([xyz, xyz[:, 0]*xyz[:, 1], xyz[:, 1]**2])
    t = np.arange(120 * 256)/256
    rng = np.random.default_rng(8503)
    sources = np.array([lfilter([1], [1, -.92], rng.normal(size=t.size))
                        + 2*np.sin(2*np.pi*f*t) for f in (6, 10, 18, 4, 12)])
    data = 5e-6*((weights-weights[-1]) @ sources) + rng.normal(0, .3e-6, (len(names), t.size))
    data[5, 10240:10266] = data[5, 10239]
    data[4, 17920:18432] += 120e-6
    data[-1] = 0.
    raw = mne.io.RawArray(data, mne.create_info(names, 256, "eeg"), first_samp=123, verbose=False)
    raw.set_montage(montage)
    raw.set_annotations(mne.Annotations([3., 7.], [.01, .01], ["tone/a", "tone/b"]))
    flat = step2.detect_flatline_inplace(raw)
    step2.mask_segment_by_flatline(raw, flat)
    raw._data[:-1] += .0012*np.sin(2*np.pi*70*t)
    with pytest.warns(RuntimeWarning, match="Persistent flatness"):
        flags = step2.assess_persistent_flatness(raw, flatlines=flat, voltage_mask=True)
    assert flags == {}
    return raw, flat


def test_noisy_reference_recovers_before_asr_and_replays(quiet_figures, tmp_path):
    raw, flats = noisy_reference()
    source = raw.get_data()
    cap = Step2Capture(raw, "Cz")
    filter_step2_inplace(raw, capture=cap)
    with pytest.warns(RuntimeWarning, match="check unavailable"):
        details, ref = step2.bad_ch_time_detection_inplace(
            raw, {}, flats, capture=cap, output_dir=tmp_path/"run")
    report = details["step2_report"]
    assert report["persistent_flatness"]["status"] == "unavailable"
    assert all(r["status"] == "unavailable" for r in report["initial_checks"].values())
    final = report["reference_recovery_passes"][-1]
    assert ref != "Cz" and ref == final["reference"]
    assert all(r["status"] == "completed" for r in final["checks"].values())
    assert final["checks"]["small_signal"]["status"] == "completed"
    assert report["asr"]["status"] == "applied" and cap.asr_status == "applied"
    assert sum(op["kind"] == "reference" for op in cap.operations) == 2
    np.testing.assert_allclose(cap.replay(source), raw.get_data(), rtol=1e-10, atol=1e-12)

    assert json.loads((tmp_path/"run"/"step2_report.json").read_text())["handoff"] == "post_asr"
    with mne.io.read_raw_fif(tmp_path/"run"/"post_asr_raw.fif", verbose=False) as saved:
        np.testing.assert_array_equal(raw.get_data(), saved.get_data())
        assert saved.info["bads"] == raw.info["bads"]
        assert saved.first_samp == raw.first_samp and saved.ch_names == raw.ch_names
        np.testing.assert_array_equal(mne.events_from_annotations(raw, verbose=False)[0],
                                      mne.events_from_annotations(saved, verbose=False)[0])


def bypass_channel_methods(monkeypatch):
    for name in ("bad_ch_by_voltage", "bad_ch_by_HFnoise", "bad_ch_by_pop", "bad_ch_by_small",
                 "bad_ch_under_robust_ref"):
        monkeypatch.setattr(step2, name, lambda *a, **k: {})
    monkeypatch.setattr(step2, "interpolate_local_artifacts_inplace", lambda *a, **k: {})
    monkeypatch.setattr(step2, "validate_reference_inplace", lambda *a, **k: "Cz")
    monkeypatch.setattr(step2, "mask_segment_by_voltage", lambda *a, **k: None)


@pytest.mark.parametrize("phase, duration, retained", [("candidate", 35, None), ("auto_selected", 75, 2500),
                                                      ("auto_selected", 75, 0)])
def test_skipped_asr_handoff_is_full_length_lossless(phase, duration, retained, quiet_figures,
                                                     monkeypatch, tmp_path):
    raw = make_raw(seconds=duration)
    raw.set_annotations(mne.Annotations([1., 2.], [.01, .01], ["tone/a", "tone/b"]))
    raw.info["chs"][0]["cal"] = 1.123456789
    flats = flatlines_for(raw, {"F3": [(np.int64(60), np.int64(89))]})
    cap = Step2Capture(raw, "Cz")
    before = raw.get_data()
    bypass_channel_methods(monkeypatch)
    fake = FakeASR(retained)
    monkeypatch.setattr(step2, "calibrate_asr", fake.calibrate)
    monkeypatch.setattr(step2, "process_asr", lambda *a, **k: pytest.fail("Unexpected ASR processing"))
    with pytest.warns(RuntimeWarning, match="ASR skipped"):
        result, ref = step2.bad_ch_time_detection_inplace(
            raw, {}, flats, capture=cap, output_dir=tmp_path/"checkpoints")
    report = json.loads((tmp_path/"checkpoints"/"step2_report.json").read_text())
    assert ref == "Cz" and report["status"] == "completed" and result["status"] == "skipped"
    assert report["asr"]["phase"] == phase and report["asr"]["replay_status"] == "skipped"
    available = raw.n_times - 29
    assert report["asr"]["candidate_samples"] == available
    assert report["asr"]["retained_samples"] == (available if retained is None else retained)
    assert report["asr"]["minimum_seconds"] == 60.
    assert report["bridged_flat_intervals"]["F3"] == [[60, 89]]
    assert report["handoff"] == "pre_asr" and "post_asr" not in report["checkpoints"]
    assert not (tmp_path/"checkpoints"/"post_asr_raw.fif").exists()
    with mne.io.read_raw_fif(tmp_path/"checkpoints"/"pre_asr_raw.fif", verbose=False) as saved:
        np.testing.assert_array_equal(saved.get_data(), before)
        np.testing.assert_array_equal(mne.events_from_annotations(raw, verbose=False)[0],
                                      mne.events_from_annotations(saved, verbose=False)[0])
    np.testing.assert_array_equal(cap.replay(before), before)
    assert raw.info["chs"][0]["cal"] == 1.123456789


def test_new_labels_restart_donor_selection_and_persistent_unknown_is_explicit(
        monkeypatch, quiet_figures, tmp_path):
    raw = make_raw(seconds=12)
    original = raw.get_data()
    flat = flatlines_for(raw, {})
    bypass_channel_methods(monkeypatch)
    def unavailable(*args, **kwargs):
        raise step2.NoScorableSupport("No eligible windows")
    monkeypatch.setattr(step2, "bad_ch_by_voltage", unavailable)
    def small(candidate, *, reference_channel=None, flatlines):
        if reference_channel is None:
            raise step2.NoScorableSupport("Missing native support")
        assert reference_channel == "F3"
        if "O1" not in candidate.info["bads"]:
            candidate.info["bads"] += ["O1"]
            return {"small": ["O1"]}
        return {"small": []}
    monkeypatch.setattr(step2, "bad_ch_by_small", small)
    passes = []
    def repair(candidate, flatlines, *, capture=None):
        np.testing.assert_array_equal(candidate.get_data(), original)
        passes.append((candidate is raw, list(candidate.info["bads"]), capture))
        candidate._data[0, 30:37] += 9e-6 if "O1" in candidate.info["bads"] else 3e-6
        return {"Fp1": [(np.int64(30), np.int64(37))]}
    monkeypatch.setattr(step2, "interpolate_local_artifacts_inplace", repair)
    monkeypatch.setattr(step2, "validate_reference_inplace", lambda *a, **k: "F3")
    monkeypatch.setattr(step2, "apply_asr_inplace", lambda *a, **k: {"corrected_duration_s": 0.})
    def broken_plot(*args):
        raise RuntimeError("Backend offline")
    monkeypatch.setattr(step2, "plot_bad_ch_report", broken_plot)
    with pytest.warns(RuntimeWarning):
        result, ref = step2.bad_ch_time_detection_inplace(raw, {}, flat, output_dir=tmp_path/"run")
    assert [(real, bads) for real, bads, cap in passes] == [
        (False, []), (False, ["O1"]), (True, ["O1"])]
    report = result["step2_report"]
    assert report["reference_recovery_passes"][-1]["checks"]["voltage"]["status"] == "unavailable"
    assert report["unanswered_channel_checks"] == ["voltage"]
    assert report["production_matches_recovery_plan"]
    assert report["channel_flags"]["small"] == ["O1"] and ref == "F3"
    assert report["diagnostic_warnings"] == [
        {"plot":"initial channel report", "error":"RuntimeError", "reason":"Backend offline"}]
    disk = json.loads((tmp_path/"run"/"step2_report.json").read_text())
    assert disk["pop_intervals"] == {"Fp1": [[30, 37]]}


@pytest.mark.parametrize("broken, last", [("bad_ch_by_small", "filtered"),
                                          ("bad_ch_under_robust_ref", "filtered"),
                                          ("apply_asr_inplace", "pre_asr")])
def test_unexpected_failure_preserves_checkpoint_without_valid_handoff(broken, last, quiet_figures,
                                                                       monkeypatch, tmp_path):
    raw = make_raw(seconds=12)
    bypass_channel_methods(monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError("Unrelated failure")
    monkeypatch.setattr(step2, broken, fail)
    with pytest.raises(ValueError, match="Unrelated failure"):
        step2.bad_ch_time_detection_inplace(raw, {}, flatlines_for(raw, {}), output_dir=tmp_path/"run")
    saved = json.loads((tmp_path/"run"/"step2_report.json").read_text())
    assert saved["status"] == "failed" and saved["last_completed_checkpoint"] == last
    assert (tmp_path/"run"/saved["checkpoints"][last]).is_file()
    assert "handoff" not in saved and not saved["reference_recovery_passes"]


def test_unexpected_retry_error_is_reported_without_modifying_filtered_eeg(
        monkeypatch, quiet_figures, tmp_path):
    raw = make_raw(seconds=12)
    before = raw.get_data()
    bypass_channel_methods(monkeypatch)
    def small(candidate, *, reference_channel=None, flatlines):
        if reference_channel is None:
            raise step2.NoScorableSupport("No eligible windows")
        raise ValueError("Geometry mismatch on corrected copy")
    monkeypatch.setattr(step2, "bad_ch_by_small", small)
    with pytest.warns(RuntimeWarning, match="small_signal"), pytest.raises(ValueError, match="Geometry"):
        step2.bad_ch_time_detection_inplace(raw, {}, flatlines_for(raw, {}), output_dir=tmp_path/"run")
    report = json.loads((tmp_path/"run"/"step2_report.json").read_text())
    assert report["status"] == "failed" and report["stage"] == "reference_recovery"
    assert report["last_completed_checkpoint"] == "filtered" and "handoff" not in report
    np.testing.assert_array_equal(before, raw.get_data())


def test_optional_scalp_view_reports_missing_ten_second_interval():
    raw = make_raw(seconds=8)
    warnings = []
    with step2.collect_plot_warnings(warnings), pytest.warns(UserWarning, match="No unmasked 10-s"):
        assert step2.plot_bad_ch_topo_traces(raw, {}) is None
    assert warnings == [{"plot": "channel scalp traces",
                         "reason": "No unmasked 10-s window; scalp traces not plotted."}]
