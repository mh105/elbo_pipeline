"""Synthetic checks of recorded reconstruction and replay with a fixed ICA choice."""
import json
from pathlib import Path
import sys
import warnings

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step3_capture import Step3Capture, reconstruct_ica, recorded_ica_change, save_ica_report
from step3_ica_viewer import storage
from test_ica_viewer import make_inputs


@pytest.fixture(scope="module")
def fitted():
    return make_inputs(mixed=True)


def native(ica, raw, exclude, n_pca_components=None):
    fit = ica.copy()
    fit.exclude = list(exclude)
    result = raw.copy().load_data()
    fit.apply(result, n_pca_components=n_pca_components, verbose=False)
    return result


@pytest.mark.parametrize("exclude", [[], [0], [1, 3]])
def test_saved_operator_matches_mne_on_recording_and_independent_background(fitted, exclude, tmp_path):
    ica, _, raw, _ = fitted
    x = raw.get_data()
    original_exclude = list(ica.exclude)
    after, cap = reconstruct_ica(raw, ica, exclude=exclude, reference_channel="Cz")
    assert cap.metadata["exclude"] == exclude and cap.metadata["fitted_channels"] == ica.ch_names
    assert cap.metadata["resolved_pca_components"] == 4
    np.testing.assert_array_equal(after.get_data(), native(ica, raw, exclude).get_data())
    np.testing.assert_array_equal(raw.get_data(), x)
    assert ica.exclude == original_exclude
    cap.save(tmp_path/"transform.npz")
    restored = Step3Capture.load(tmp_path/"transform.npz")
    assert restored.metadata == cap.metadata
    rng = np.random.default_rng(427)
    background = rng.normal(size=(len(raw.ch_names), 401))*13e-6
    shape = np.exp(-((np.arange(401)-119)/22)**2) - .4*np.exp(-((np.arange(401)-85)/11)**2)
    s = np.array([1., -.5, .2, .8, 0])[:, None] * shape * 4e-6
    def F(x):
        trial = mne.io.RawArray(x.copy(), raw.info.copy(), verbose=False)
        return native(ica, trial, exclude).get_data()
    observed = F(background + s) - F(background)
    synthetic = restored.replay(s, ch_names=raw.ch_names)
    np.testing.assert_allclose(synthetic, observed, atol=1e-15, rtol=1e-10)
    np.testing.assert_allclose(synthetic, F(s)-F(np.zeros_like(s)), atol=1e-15, rtol=1e-10)
    np.testing.assert_allclose(cap.replay(x, ch_names=raw.ch_names) + cap.offset_v[:, None],
                               after.get_data(), atol=1e-15, rtol=1e-10)
    if exclude:
        assert np.any(np.abs(cap.offset_v) > 1e-10)
    else:
        np.testing.assert_allclose(synthetic, s, atol=1e-15, rtol=1e-10)


def test_empty_exclusions_respect_pca_reconstruction_setting(fitted):
    _, epochs, raw, _ = fitted
    model = mne.preprocessing.ICA(n_components=3, method="infomax", rng=128, max_iter=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(epochs, verbose=False)
    after, cap = reconstruct_ica(raw, model, exclude=[], n_pca_components=3, reference_channel="Cz")
    assert cap.metadata["resolved_pca_components"] == 3
    assert np.max(abs(after.get_data()-raw.get_data())) > 1e-7
    np.testing.assert_array_equal(after.get_data(), native(model, raw, [], 3).get_data())
    np.testing.assert_allclose(cap.replay(raw.get_data(), ch_names=raw.ch_names) + cap.offset_v[:, None],
                               after.get_data(), atol=1e-15, rtol=1e-10)


def test_saved_labeling_revisions_and_explicit_branches_are_independent(fitted, tmp_path, monkeypatch):
    ica, epochs, raw, labels = fitted
    path = tmp_path/"input-eeg.fif"
    path.write_bytes(b"synthetic source")
    monkeypatch.setattr(storage, "ensure_cache", lambda directory: directory)
    viewer = storage.prepare_bundle(ica, epochs, raw, eeg_path=path,
                                    reference_channel="Cz", iclabel_table=labels)
    storage.save_review(viewer, [], expected_revision=0)
    storage.create_review_version(viewer, expected_revision=1)
    storage.save_review(viewer, [2,3], expected_revision=2, version="v2")
    before = json.dumps(storage.read_review(viewer))
    for version, expected in ((None, []), ("v2", [2,3])):
        after, cap = reconstruct_ica(raw, ica, reference_channel="Cz", viewer_dir=viewer, version=version)
        assert cap.metadata["choice"]["version"] == (version or "v1")
        assert cap.metadata["choice"]["source"] == "viewer" and cap.metadata["choice"]["revision"] == 3
        assert cap.metadata["exclude"] == expected
        assert cap.metadata["fit_id"] == storage.read_review(viewer)["fit_id"]
        np.testing.assert_array_equal(after.get_data(), native(ica, raw, expected).get_data())
    _, explicit = reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[1], viewer_dir=viewer)
    assert explicit.metadata["choice"] == {"source":"explicit"} and explicit.metadata["exclude"] == [1]
    assert before == json.dumps(storage.read_review(viewer)) and ica.exclude == [0]
    with pytest.raises(ValueError, match="version"):
        reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[1], viewer_dir=viewer, version="v2")
    # Exercise the same final-review entry point used by both example scripts.
    from step3_ica_cleaning import apply_reviewed_ica
    monkeypatch.setattr("step3_ica_cleaning.summarize_ica_exclusions",
                        lambda live, epochs, before: tuple(live.exclude))
    live = ica.copy()
    cleaned, summary = apply_reviewed_ica(raw, live, epochs, viewer, reference_channel="Cz",
                                          version="v2")
    assert summary == (2, 3)
    np.testing.assert_array_equal(cleaned.get_data(), native(ica, raw, [2, 3]).get_data())
    saved_capture, = (viewer/"reconstructions").glob("*/ica_transform.npz")
    assert Step3Capture.load(saved_capture).metadata["choice"]["version"] == "v2"
    assert before == json.dumps(storage.read_review(viewer))

    def unavailable(*args):
        raise ValueError("No samples for the variance summary")
    monkeypatch.setattr("step3_ica_cleaning.summarize_ica_exclusions", unavailable)
    with pytest.warns(RuntimeWarning, match="summary unavailable"):
        cleaned, figure = apply_reviewed_ica(raw, live, epochs, viewer, reference_channel="Cz")
    assert figure is None and live.exclude == []
    np.testing.assert_array_equal(cleaned.get_data(), native(ica, raw, []).get_data())
    warning, = (viewer/"reconstructions").glob("*/summary_warning.txt")
    assert "No samples" in warning.read_text()

    changed = ica.copy()
    changed.mixing_matrix_[0, 0] += 1.
    with pytest.raises(ValueError, match="does not match"):
        reconstruct_ica(raw, changed, reference_channel="Cz", viewer_dir=viewer)
    with pytest.raises(ValueError, match="sample grid"):
        reconstruct_ica(raw.copy().crop(tmax=2.), ica, reference_channel="Cz", viewer_dir=viewer)


def test_rms_report_matches_direct_calculation_and_counts_bad_samples(fitted, tmp_path):
    ica, _, input_raw, _ = fitted
    raw = input_raw.copy().crop(tmax=2.225)
    raw.set_annotations(mne.Annotations([.5], [1.5], ["BAD_motion"]))
    after, cap = reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[1])
    for policy in ("include", "omit"):
        folder = save_ica_report(tmp_path, raw, after, cap, bad_samples=policy)
        report = json.loads((folder/"eeg_change.json").read_text())
        assert (folder/"eeg_change.png").stat().st_size > 0
        assert len(report["channels"]) == len(ica.ch_names) and len(report["windows"]) == 3
        assert report["bad_intervals"] == [[500, 2000]]
        indices = np.arange(raw.n_times)
        if policy == "omit":
            indices = indices[(indices < 500) | (indices >= 2000)]
            assert report["windows"][1]["change_rms_uv"] is None
        assert report["n_used"] == indices.size == sum(row["n_used"] for row in report["windows"])
        first = raw.get_data(picks=ica.ch_names)[:, indices]*1e6
        second = after.get_data(picks=ica.ch_names)[:, indices]*1e6
        relative = 100*np.linalg.norm(first-second, axis=1)/np.linalg.norm(first, axis=1)
        np.testing.assert_allclose([row["relative_change_percent"] for row in report["channels"]], relative)
        for row in report["windows"]:
            where = (indices >= row["start"]) & (indices < row["stop"])
            if where.any():
                assert row["change_rms_uv"] == pytest.approx(np.sqrt(np.mean((first[:, where]-second[:, where])**2)))
    _, other = reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[2])
    with pytest.raises(ValueError, match="reproduce"):
        recorded_ica_change(raw, after, other)


def test_other_channels_pass_through_and_undefined_ratios_are_reported(fitted):
    ica, _, input_raw, _ = fitted
    raw = input_raw.copy()
    spare = mne.io.RawArray(np.full((1, raw.n_times), np.nan), mne.create_info(["Spare"],1000,"eeg"),
                            first_samp=raw.first_samp, verbose=False)
    raw.add_channels([spare], force_update_info=True)
    raw.info["bads"] = ["Spare"]
    raw._data[0] = 0.
    after, cap = reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[1])
    x = raw.get_data()
    np.testing.assert_array_equal(cap.replay(x, ch_names=raw.ch_names)[-2:], x[-2:])
    np.testing.assert_array_equal(after.get_data()[-2:], x[-2:])
    assert recorded_ica_change(raw, after, cap)["channels"][0]["relative_change_percent"] is None
    with pytest.raises(ValueError, match="channel names and order"):
        cap.replay(x[::-1], ch_names=raw.ch_names[::-1])
    with pytest.raises(ValueError, match="fit order"):
        reconstruct_ica(raw.copy().reorder_channels(raw.ch_names[::-1]), ica, reference_channel="Cz", exclude=[])
    with pytest.raises(ValueError, match="out-of-range"):
        reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[ica.n_components_])


def test_all_bad_samples_are_reported_without_artificial_zero(fitted):
    ica, _, input_raw, _ = fitted
    raw = input_raw.copy()
    raw.set_annotations(mne.Annotations([0.], [raw.n_times/1000], ["BAD_all"]))
    after, cap = reconstruct_ica(raw, ica, reference_channel="Cz", exclude=[1])
    change = recorded_ica_change(raw, after, cap, bad_samples="omit")
    assert change["n_used"] == 0
    assert all(row["relative_change_percent"] is None for row in change["channels"])
    assert all(row["change_rms_uv"] is None for row in change["windows"])


def test_same_grid_different_recording_and_older_viewer(fitted, tmp_path, monkeypatch):
    ica, epochs, raw, labels = fitted
    source = tmp_path / "source.fif"
    source.write_bytes(b"synthetic source")
    monkeypatch.setattr(storage, "ensure_cache", lambda directory: directory)
    viewer = storage.prepare_bundle(ica, epochs, raw, eeg_path=source,
                                    reference_channel="Cz", iclabel_table=labels)
    settings = dict(reference_channel="Cz", exclude=[1], viewer_dir=viewer)
    _, cap = reconstruct_ica(raw, ica, **settings)
    assert cap.metadata["recording_check"]["status"] == "verified_samples"
    other = raw.copy()
    other._data[0, 5] += 2e-6
    with pytest.raises(ValueError, match="processed recording differs"):
        reconstruct_ica(other, ica, **settings)
    after, cap = reconstruct_ica(other, ica, **settings, allow_other_recording=True)
    assert cap.metadata["recording_check"]["status"] == "different_recording"
    np.testing.assert_array_equal(after.get_data(), native(ica, other, [1]).get_data())

    # Legacy manifests supply grid information but no digest of the processed samples.
    manifest = viewer / "manifest.json"
    data = json.loads(manifest.read_text())
    del data["original_raw"]["samples_sha256"]
    manifest.write_text(json.dumps(data))
    _, cap = reconstruct_ica(other, ica, **settings)
    assert cap.metadata["recording_check"] == {"status": "grid_only"}


def test_combined_report_verifies_once_and_separate_report_rechecks(fitted, tmp_path, monkeypatch):
    from step3_capture import reconstruct_ica_report
    ica, _, raw, _ = fitted
    original = Step3Capture.verify_recording
    calls = []
    def verify(cap, before, after):
        calls.append(cap)
        return original(cap, before, after)
    monkeypatch.setattr(Step3Capture, "verify_recording", verify)
    after, cap, path = reconstruct_ica_report(tmp_path, raw, ica, reference_channel="Cz", exclude=[1])
    assert calls == [cap]
    loaded = Step3Capture.load(path / "ica_transform.npz")
    recorded_ica_change(raw, after, loaded)
    assert len(calls) == 2 and calls[-1] is loaded
