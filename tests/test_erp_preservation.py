"""Synthetic regression cases; no EEG files, fit, viewer or network are needed."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from erp_preservation import (
    Assessment,
    Model,
    Term,
    Windows,
    load_case,
    plot_model,
    plot_waveforms,
    write_assessment,
)
from erp_preservation.metrics import measure
from step2_capture import Step2Capture
from step3_capture import Step3Capture

NAMES = ["A", "B", "C", "Ref", "AUX"]
XYZ = np.array([[0.2, 0.4, 1], [-0.2, 0, 1], [0, -0.6, 1], [0, 0.1, 1], [np.nan] * 3])
SF = 200
EVENTS = {"beep": [102, 280, 330, 710], "boop": [306, 800, 1060]}
LABELS = ("beep", "boop")


def capture2(status="applied"):
    raw = mne.io.RawArray(
        np.zeros((5, 1000)),
        mne.create_info(NAMES, SF, ["eeg"] * 4 + ["eog"]),
        first_samp=100,
        verbose=False,
    )
    cap = Step2Capture(raw, "Ref")
    cap.output_reference = "B"
    cap.repairs_observed = cap.reference_check_observed = True
    cap.operations.append({"kind": "filter", "picks": [0, 1, 2, 3], "taps": np.array([1.]),
                           "phase": "zero", "pad": "reflect_limited", "bounds": [[0, cap.n_times]]})
    cap.operations.append({"kind": "reference", "pick": 1})
    cap.asr_status = status
    cap.asr_picks = [0, 2, 3]
    if status == "applied":
        cap.operations.append(
            {
                "kind": "asr",
                "start": 190,
                "stop": 330,
                "previous": np.eye(3),
                "current": np.diag([0.6, 0.8, 1.0]),
                "blend": np.linspace(0, 1, 140),
            }
        )
    return cap


def capture3():
    matrix = np.eye(5)
    matrix[0, 0] = 0.5
    matrix[2, 2] = 0.75
    return Step3Capture(
        {
            "channels": NAMES,
            "fitted_channels": ["A", "C", "Ref"],
            "bad_channels": [],
            "reference_channel": "B",
            "n_times": 1000,
            "first_samp": 100,
            "sfreq": SF,
            "exclude": [1],
            "fit_id": "synthetic",
            "n_components": 3,
        },
        matrix,
        np.array([0.003, 0, 0.002, 0.001, 0]),
    )


def model():
    return Model(
        "known response",
        "custom",
        (
            Term([0, 1, 0.5, 0], [1, 0, -0.3, 0, 0], {"beep": 4, "boop": 2}),
            Term([0, 0, 1], [0, 0.5, 1, 0, 0], {"beep": 1, "boop": -2}),
        ),
        [1, 0, -0.3, 0, 0],
        1,
        (0, 0.1),
        (0, 0.1),
    )


def selected(result, stage, condition="beep", style="baseline_corrected"):
    return next(
        r
        for r in result["records"]
        if r["condition"] == condition and r["stage"] == stage and r.get("convention") == style
    )


def test_default_catalog_and_geometry_on_another_montage():
    cap = capture2()
    assay = Assessment(cap, EVENTS)
    cases = list(assay.auditory_models(XYZ, conditions=LABELS))
    assert len(cases) == 420
    assert [sum(c.family == f for c in cases) for f in ("legacy", "component", "composite")] == [
        36,
        312,
        72,
    ]
    assert len({c.name for c in cases}) == 420
    for c in cases:
        for t in c.terms:
            assert t.weights[3] == 0 and t.weights[-1] == 0
            assert np.max(abs(t.weights)) == pytest.approx(1)
    different = list(
        Assessment(capture3(), EVENTS, upstream_step2=cap).auditory_models(XYZ, conditions=LABELS)
    )
    for a, b in zip(cases, different):
        np.testing.assert_array_equal(a.focus, b.focus)
    broken = XYZ.copy()
    broken[0] = 0.0
    with pytest.raises(ValueError, match="finite EEG positions"):
        list(assay.auditory_models(broken, conditions=LABELS))


def test_step2_stages_agree_with_direct_probe_and_keep_overlap():
    cap = capture2()
    assay = Assessment(cap, EVENTS, contrast=("boop", "beep"), bad_intervals=[(178, 180)])
    result = assay.assess(model())
    # Independently superimpose pulses on full continuous EEG, including a
    # boundary-incomplete event whose tail can affect a different epoch.
    probe = np.zeros((5, 1000))
    for t in model().terms:
        for label, amp in t.amplitudes_uv.items():
            for event in EVENTS[label]:
                i = event - cap.first_samp
                pulse = t.pulse[: cap.n_times - i]
                probe[:, i : i + len(pulse)] += amp * 1e-6 * t.weights[:, None] * pulse
    snapshots = cap.replay_checkpoints(probe)
    for k, x in snapshots.items():
        np.testing.assert_array_equal(x, cap.replay(probe, stage=k))
    x = probe.copy()
    x[:4] -= probe[1].copy()
    pairs = {
        "filter_only": (x, x),
        "repair_increment": (x, snapshots["pre_asr"]),
        "asr_increment": (snapshots["pre_asr"], snapshots["post_asr"]),
        "whole_step2": (x, snapshots["post_asr"]),
    }
    rep = cap.channels.index(result["representative"])
    for stage, (b, a) in pairs.items():
        r = selected(result, stage)
        assert r["requested"] == 4
        np.testing.assert_array_equal(r["used"], [280, 330, 710])
        for j, onset in enumerate(r["used"] - 100):
            lo, hi = onset - 40, onset + 280
            for arr, field in ((b, "event_before"), (a, "event_after")):
                expected = arr[rep, lo:hi].copy()
                expected -= expected[:40].mean()
                np.testing.assert_allclose(r[field][j], expected, atol=1e-20)
    assert assay.coverage["beep"][1]["bad"] and assay.coverage["beep"][1]["asr"]
    assert not assay.coverage["beep"][0]["complete"]
    assert len(result["contrasts"]) == 4


def test_step3_is_independent_of_step2_and_of_fitted_ica_offset():
    prior = capture2()
    cap = capture3()
    assess = Assessment(cap, EVENTS, upstream_step2=prior)
    result = assess.assess(model())
    direct = np.zeros((5, 1000))
    for term in model().terms:
        weights = term.weights.copy()
        weights[:4] -= weights[1]
        for label, amp in term.amplitudes_uv.items():
            for sample in EVENTS[label]:
                i = sample - 100
                direct[:, i : i + len(term.pulse)] += 1e-6 * amp * weights[:, None] * term.pulse
    background = np.random.default_rng(52).normal(size=direct.shape) * 2e-5
    F = lambda x: cap.matrix @ x + cap.offset_v[:, None]
    expected = F(background + direct) - F(background)
    rep = NAMES.index(result["representative"])
    r = selected(result, "ica", style="unbaselined")
    trials = [expected[rep, n - 100 - 40 : n - 100 + 280] for n in r["used"]]
    np.testing.assert_allclose(r["mean_after"], np.mean(trials, axis=0), atol=1e-18)
    assert assess.coverage["boop"][0]["asr"]
    cap.metadata["first_samp"] += 1
    with pytest.raises(ValueError, match="Upstream"):
        Assessment(cap, EVENTS, upstream_step2=prior)


@pytest.mark.parametrize(
    "status,whole", [("skipped", True), ("unavailable", False), ("not_run", False)]
)
def test_missing_asr_is_not_reported_as_an_identity_measurement(status, whole, tmp_path):
    cap = capture2(status)
    assay = Assessment(cap, EVENTS)
    directory = write_assessment(tmp_path / "run", assay, [model()], plot=False)
    context = json.loads((directory / "context.json").read_text())
    assert context["stages"]["asr_increment"] == status
    case = load_case(directory / "case_0000.npz")
    stages = {r["stage"] for r in case["records"]}
    assert ("whole_step2" in stages) is whole and "asr_increment" not in stages
    if whole:
        assert np.allclose(selected(case, "whole_step2")["metrics"]["gain"], 1.0)
    assert context["coverage"]["beep"][1]["bad"] is None
    assert context["coverage"]["beep"][1]["asr"] is (False if status == "skipped" else None)


def test_known_distortions_and_undefined_peaks():
    sf = 100
    w = Windows(epoch=(-0.1, 0.6), baseline=(-0.1, 0), wide_mean=(0.1, 0.4))
    b = np.zeros((2, 2, 70))
    b[0, :, 28:31] = np.array([1, 2, 1]) * 1e-6
    a = 0.65 * np.roll(b, 3, axis=2)
    metrics, _, _ = measure(b, a, sf, 0, 1, (0.1, 0.4), (0.1, 0.4), w, True)
    assert metrics["peak_shift_ms"][0] == 30
    assert metrics["peak_attenuation_percent"][0] == pytest.approx(35)
    assert metrics["rms_error_uv"][0] > 0
    assert metrics["low_peak_reference"][1] and np.isnan(metrics["peak_attenuation_percent"][1])
    assert metrics["output_shape_undefined"][1] and np.isnan(metrics["correlation"][1])
    assert metrics["peak_interpretation_flag"][1]
    rev, _, _ = measure(b, -b, sf, 0, 1, (0.1, 0.4), (0.1, 0.4), w, True)
    assert rev["gain"][0] == pytest.approx(-1.0) and rev["correlation"][0] == pytest.approx(-1.0)
    assert rev["mean_after_uv"][0] == pytest.approx(-rev["mean_before_uv"][0])


def test_saved_outputs_plots_and_no_complete_epochs(tmp_path):
    m = model()
    cap = capture3()
    w = Windows(epoch=(-0.1, 0.5), baseline=(-0.1, -0.02), wide_mean=(0.1, 0.4))
    assay = Assessment(cap, EVENTS, w, contrast=("boop", "beep"))
    root = write_assessment(tmp_path / "result", assay, [m], full_events=True)
    assert (root / "preservation.pdf").stat().st_size > 500
    case = load_case(root / "case_0000.npz")
    assert len(case["contrasts"]) == 1
    q = selected(case, "ica")
    assert q["event_before"].shape == (len(q["used"]), 120)
    f = plot_waveforms([("ICA", case, "ica")], condition="beep", event_sample=280)
    np.testing.assert_allclose(f.axes[0].lines[0].get_ydata(), q["event_after"][0] * 1e6)
    plt.close(f)
    f = plot_model(m, assay, XYZ, condition="beep")
    assert len(f.axes) == 5
    plt.close(f)
    boundary = Assessment(cap, {"beep": [100], "boop": []}, w)
    out = write_assessment(tmp_path / "empty_epochs", boundary, [m], plot=False)
    records = load_case(out / "case_0000.npz")["records"]
    assert all(r["status"] == "no_complete_epochs" for r in records)
    with pytest.raises(ValueError, match="absolute integer"):
        Assessment(cap, {"beep": [-1]})
    with pytest.raises(ValueError, match="capture channel order"):
        boundary.assess(
            Model(
                "invalid",
                "custom",
                (Term([1.0], [1.0], {"beep": 2}),),
                [1.0],
                1,
                (0, 0.1),
                (0, 0.1),
            )
        )


def test_negative_pulse_offset_and_overlap_are_scored_against_the_full_train():
    assay = Assessment(capture3(), EVENTS)
    wave = np.ones(60)
    m = Model(
        "overlap",
        "custom",
        (Term(wave, [1, 0, 0, 0, 0], {"beep": 2, "boop": -1}, -10),),
        [1, 0, 0, 0, 0],
        1,
        (0, 0.2),
        (0, 0.2),
    )
    r = selected(assay.assess(m), "ica", style="unbaselined")
    full = np.zeros(1000)
    for label, amp in {"beep": 2, "boop": -1}.items():
        for n in EVENTS[label]:
            start = n - assay.first - 10
            full[max(start, 0) : min(start + 60, 1000)] += amp * 1e-6
    expected = np.mean(
        [full[n - assay.first - 40 : n - assay.first + 280] for n in r["used"]], axis=0
    )
    np.testing.assert_allclose(r["mean_before"], expected, atol=1e-19)
    np.testing.assert_allclose(r["mean_after"], expected * 0.5, atol=1e-19)


def test_composite_assessment_matches_direct_mne_with_artificial_background():
    from step3_capture import reconstruct_ica
    from test_ica_viewer import make_inputs

    ica, _, raw, _ = make_inputs(mixed=True)
    after, cap = reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[2])
    onsets = {"first": np.array([2100, 2450]), "second": np.array([2330, 4400])}
    ev = {k: v + raw.first_samp for k, v in onsets.items()}
    assay = Assessment(cap, ev, contrast=("second", "first"))
    xyz = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    m = next(
        model for model in assay.auditory_models(xyz, conditions=ev) if model.family == "composite"
    )
    result = assay.assess(m)
    probe = np.zeros_like(raw.get_data())
    for t in m.terms:
        for label, amplitude in t.amplitudes_uv.items():
            for n in onsets[label]:
                probe[:, n : n + len(t.pulse)] += 1e-6 * amplitude * t.weights[:, None] * t.pulse
    trial = raw.copy()
    trial._data += probe
    model = ica.copy()
    model.exclude = [2]
    model.apply(trial, verbose=False)
    difference = trial.get_data() - after.get_data()
    electrode = raw.ch_names.index(result["representative"])
    for label in ev:
        r = selected(result, "ica", condition=label, style="unbaselined")
        expected = np.array([difference[electrode, n - 200 : n + 1400] for n in onsets[label]])
        np.testing.assert_allclose(r["mean_after"], expected.mean(0), atol=1e-15, rtol=1e-10)


def test_identical_condition_models_can_have_overlap_and_keep_their_known_contrast(tmp_path):
    cap = capture3()
    cap.matrix[:] = np.eye(len(NAMES))
    ev = {"beep": [360, 440, 800], "boop": [410, 720]}
    prior = capture2()
    assay = Assessment(cap, ev, upstream_step2=prior, contrast=("boop", "beep"))
    m = next(
        c
        for c in assay.auditory_models(XYZ, conditions=LABELS)
        if c.info.get("amplitudes") == "identical_conditions"
    )
    root = write_assessment(tmp_path / "contrast", assay, [m], plot=False)
    r = load_case(root / "case_0000.npz")
    (con,) = r["contrasts"]
    assert np.max(np.abs(con["before"])) > 1e-8
    np.testing.assert_allclose(con["before"], con["after"], atol=1e-20)
    row = selected(r, "ica", condition="boop")
    assert "event_before" not in row
    sample = row["examples"]["largest_peak_change"]["sample"]
    fig = plot_waveforms([("ICA", r, "ica")], condition="boop", event_sample=sample)
    plt.close(fig)
    context = json.loads((root / "context.json").read_text())
    assert len(context["capture_fingerprint"]) == len(context["upstream_step2_fingerprint"]) == 64


def test_real_sample_grid_validation_and_zero_duration_bad_marker():
    cap = capture2()
    assay = Assessment(cap, EVENTS, bad_intervals=iter([(177, 177), (180, 186)]))
    assert assay.coverage["beep"][1]["bad"]
    assert not Assessment(cap, EVENTS, bad_intervals=[(177, 177)]).coverage["beep"][1]["bad"]
    with pytest.raises(ValueError, match="sample intervals"):
        Assessment(cap, EVENTS, bad_intervals=[(-3, 25)])
    cap.operations.append({"kind": "filter"})
    with pytest.raises(ValueError, match="chronological|filtering"):
        cap.replay_checkpoints(np.zeros((len(NAMES), cap.n_times)))


def test_missing_filter_preserves_independent_comparisons(tmp_path):
    cap = capture2()
    cap.operations.pop(0)  # Local repairs and ASR were observed; filtering was not.
    assay = Assessment(cap, EVENTS)
    directory = write_assessment(tmp_path / "late", assay, [model()], plot=False)
    ctx = json.loads((directory / "context.json").read_text())
    assert ctx["stages"] == {"filter_only": "unavailable", "repair_increment": "available",
                             "asr_increment": "applied", "whole_step2": "unavailable"}
    assert ctx["capture"]["asr_probe_input"] == "standalone_in_effective_reference"
    r = load_case(directory / "case_0000.npz")
    assert {a["stage"] for a in r["records"]} == {"repair_increment", "asr_increment"}
    # ASR acts independently on a known signal in the effective reference.
    probe = sum(assay._input_terms(model()))
    standalone = assay._in_reference(probe)
    pairs = assay._step2_pairs(model())
    np.testing.assert_array_equal(pairs["asr_increment"][0], standalone)
    np.testing.assert_array_equal(pairs["asr_increment"][1], cap.replay(standalone, stage="asr_only"))

    cap.operations.clear()
    cap.repairs_observed = cap.reference_check_observed = False
    cap.asr_status = "skipped"
    assert cap.replay_checkpoints(np.zeros((len(NAMES), cap.n_times))) == {}
    assert Assessment(cap, EVENTS).assess(model())["records"] == []


def test_disappearing_peak_is_counted_as_loss_without_a_latency_estimate(tmp_path):
    cap = capture3()
    cap.matrix[0] = 0.
    assay = Assessment(cap, EVENTS)
    directory = write_assessment(tmp_path / "lost", assay, [model()], plot=False)
    row = selected(load_case(directory / "case_0000.npz"), "ica")
    assert row["metrics"]["peak_interpretation_flag"].all()
    assert np.isfinite(row["metrics"]["peak_shift_ms"]).all()  # raw diagnostic retained
    assert "largest_peak_change" in row["examples"]
    assert "largest_peak_shift" not in row["examples"]
    assert "largest_legacy_peak_shift" not in row["examples"]
    summary = json.loads((directory / "summary.json").read_text())
    dist = next(d for d in summary["distributions"] if d["condition"] == "beep"
                and d["convention"] == "baseline_corrected")
    for name in ("peak_shift_ms", "legacy_peak_shift_ms"):
        assert dist["events"][name] == {"count": 3, "undefined": 3}
        assert dist["averages"][name] == {"count": 1, "undefined": 1}
    assert dist["events"]["peak_attenuation_percent"]["median"] == pytest.approx(100)


def test_window_coverage_and_unavailable_measurements(tmp_path):
    windows = Windows(epoch=(-0.1, 0.6), baseline=(-0.1, 0), wide_mean=(0.3, 0.9))
    m = model()
    m.mean_window = (0.7, 0.9)
    m.peak_window = (0.65, 0.8)
    assay = Assessment(capture3(), EVENTS, windows, contrast=LABELS)
    directory = write_assessment(tmp_path / "windows", assay, [m], plot=False)
    data = load_case(directory / "case_0000.npz")
    spec = data["measurement_windows"]
    assert spec["wide_mean"] == dict(requested_s=[.3, .9], used_s=[.3, .6], samples=60, status="partial")
    assert spec["baseline"]["status"] == "complete"
    assert spec["mean"]["status"] == spec["peak"]["status"] == "unavailable"
    row = selected(data, "ica")
    assert np.isnan(row["metrics"]["peak_before_uv"]).all()
    assert np.isnan(row["metrics"]["mean_after_uv"]).all()
    assert np.isfinite(row["metrics"]["wide_mean_after_uv"]).all()
    assert data["contrasts"][0]["windows"]["mean"]["change_uv"] is None
    no_baseline = Windows(epoch=(0., .6), baseline=(-.1, 0), wide_mean=(.1,.4))
    output = Assessment(capture3(), EVENTS, no_baseline, contrast=LABELS).assess(model())
    assert selected(output, "ica", style="unbaselined")["status"] == "measured"
    assert selected(output, "ica")["status"] == "no_baseline_samples"
    assert output["contrasts"] == []


def test_interrupted_assessment_can_retry_without_overwriting_completed_output(tmp_path):
    assay = Assessment(capture3(), EVENTS)
    destination = tmp_path / "review"
    def stopped():
        yield model()
        raise RuntimeError("model stream interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        write_assessment(destination, assay, stopped(), plot=False)
    assert not destination.exists()
    # A hard stop may leave unpublished staging data. It does not reserve the final name.
    (tmp_path / ".review.incomplete-abandoned").mkdir()
    write_assessment(destination, assay, [model()], plot=False)
    original = (destination / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_assessment(destination, assay, [], plot=False)
    assert (destination / "summary.json").read_bytes() == original


def test_v1_archive_is_readable_without_inventing_stage_completion(tmp_path):
    cap = capture2()
    path = tmp_path / "capture.npz"
    cap.save(path)
    # v1 stored applied operations, without markers for two completed procedures.
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files if k != "manifest"}
        manifest = json.loads(saved["manifest"].item())
    manifest["version"] = 1
    del manifest["repairs_observed"], manifest["reference_check_observed"]
    np.savez_compressed(path, manifest=json.dumps(manifest), **arrays)
    old = Step2Capture.load(path)
    assert old.stage_availability == dict(filter_only=True, repair_increment=False,
                                          asr_increment=True, whole_step2=False)
    x = np.random.default_rng(123).normal(size=(len(NAMES), cap.n_times))
    np.testing.assert_array_equal(old.replay(x), cap.replay(x))
