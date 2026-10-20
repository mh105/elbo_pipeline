"""Persistence and numerical contracts for the reusable ICA review viewer."""

import base64
import json
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import mne
import numpy as np
import pandas as pd


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from step3_ica_viewer import build as builder, server, storage


CLASSES = (
    "brain", "muscle artifact", "eye blink", "heart beat", "line noise",
    "channel noise", "other",
)


def make_inputs(*, mixed=False):
    """A real fitted transform, with original-rate event samples after decimation."""
    rng = np.random.default_rng(518)
    names = ["Fp1", "Fp2", "C3", "C4"]
    data = rng.laplace(scale=8e-6, size=(3, len(names), 500))
    if mixed:
        mixing = rng.normal(size=(4, 4))
        mixing /= np.linalg.norm(mixing, axis=0)
        data = np.einsum("ij,ejt->eit", mixing, data)
    info = mne.create_info(names, 250, "eeg")
    info.set_montage("colin27_1020")
    with info._unlock():
        info["highpass"], info["lowpass"] = 1.0, 100.0
    events = np.column_stack(([5500, 7500, 11500], np.zeros(3, int), np.ones(3, int)))
    epochs = mne.EpochsArray(data, info, events=events, baseline=None, verbose=False)
    epochs._raw_sfreq = 1000.0
    ica = mne.preprocessing.ICA(
        n_components=4, method="infomax", fit_params={"extended": True},
        max_iter=24, rng=518,
    )
    with warnings.catch_warnings(), mne.use_log_level("ERROR"):
        warnings.simplefilter("ignore", RuntimeWarning)
        ica.fit(epochs, picks="eeg")
    ica.exclude = [0]
    raw_info = mne.create_info(names + ["Cz"], 1000, "eeg")
    raw_info.set_montage("colin27_1020")
    raw_data = np.vstack((rng.normal(scale=8e-6, size=(4, 8000)), np.zeros(8000)))
    raw = mne.io.RawArray(raw_data, raw_info,
                          first_samp=5500, verbose=False)
    probabilities = np.full((4, 7), 0.01)
    probabilities[0, 2] = 0.94
    probabilities[1:, 0] = 0.94
    table = pd.DataFrame(probabilities, columns=CLASSES)
    table.insert(0, "component", np.arange(4))
    table["winning_label"] = ["eye blink", "brain", "brain", "brain"]
    table["winning_probability"] = 0.94
    table["iclabel_auto_exclude"] = [True, False, False, False]
    table["excluded"] = table["iclabel_auto_exclude"]
    return ica, epochs, raw, table


class TestViewerReviewStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ica, cls.epochs, cls.raw, cls.labels = make_inputs()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ica-viewer-test-")
        self.addCleanup(self.tmp.cleanup)
        self.eeg_path = Path(self.tmp.name) / "sub-test_task-test_eeg.fif"
        self.eeg_path.write_bytes(b"immutable original EEG recording")

    def prepare(self, ica=None):
        return storage.prepare_bundle(
            self.ica if ica is None else ica, self.epochs, self.raw,
            eeg_path=self.eeg_path, reference_channel="Cz", iclabel_table=self.labels,
        )

    def decision_file(self, viewer_dir):
        return next(Path(viewer_dir).glob("*_ica_exclude.json"))

    def test_reprepare_preserves_saved_acceptance_and_exact_json_bytes(self):
        viewer_dir = self.prepare()
        self.assertEqual(storage.read_review(viewer_dir)["exclude"], [0])
        saved = storage.save_review(viewer_dir, [2], expected_revision=0)
        decision = self.decision_file(viewer_dir)
        # Preserve formatting too: preparation must never rewrite the review.
        decision.write_text(json.dumps(saved, indent=4) + "\n", encoding="utf-8")
        before = decision.read_bytes()
        changed_exclusions = self.ica.copy()
        changed_exclusions.exclude = [0, 1, 3]
        self.assertEqual(self.prepare(changed_exclusions), viewer_dir)
        self.assertEqual(decision.read_bytes(), before)
        self.assertEqual(storage.load_exclusions(viewer_dir, ica=changed_exclusions), [2])
        self.assertEqual(changed_exclusions.exclude, [0, 1, 3])

    def test_empty_review_is_complete_list_and_loader_does_not_mutate_ica(self):
        viewer_dir = self.prepare()
        storage.save_review(viewer_dir, [], expected_revision=0)
        self.assertEqual(storage.load_exclusions(viewer_dir, ica=self.ica), [])
        self.assertEqual(self.ica.exclude, [0])

    def test_changed_numerical_transform_gets_new_fit_and_cannot_load_old_review(self):
        viewer_dir = self.prepare()
        changed = self.ica.copy()
        changed.unmixing_matrix_[0, 0] += 0.125
        changed._update_mixing_matrix()
        other_dir = self.prepare(changed)
        self.assertNotEqual(viewer_dir, other_dir)
        with self.assertRaises(ValueError):
            storage.load_exclusions(viewer_dir, ica=changed)
        self.assertEqual(storage.read_review(viewer_dir)["exclude"], [0])

    def test_corrupt_or_invalid_existing_review_is_never_overwritten(self):
        viewer_dir = self.prepare()
        path = self.decision_file(viewer_dir)
        initial = storage.read_review(viewer_dir)
        invalid_files = [b"{broken JSON", b"[]"]
        for invalid in ([True], [-1], [4], [0, 0], [1.5]):
            invalid_files.append(json.dumps(dict(initial, exclude=invalid)).encode())
        invalid_files.append(json.dumps(dict(initial, fit_id="different-fit")).encode())
        for content in invalid_files:
            with self.subTest(content=content):
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(path.read_bytes(), content)
                with self.assertRaises(ValueError):
                    storage.load_exclusions(viewer_dir, ica=self.ica)

    def test_invalid_updates_and_stale_revision_preserve_last_saved_decision(self):
        viewer_dir = self.prepare()
        saved = storage.save_review(viewer_dir, [1, 3], expected_revision=0)
        self.assertEqual(saved["revision"], 1)
        before = self.decision_file(viewer_dir).read_bytes()
        for invalid in ([True], [4], [-1], [1, 1], [1.0]):
            with self.subTest(exclude=invalid), self.assertRaises(ValueError):
                storage.save_review(viewer_dir, invalid, expected_revision=1)
            self.assertEqual(self.decision_file(viewer_dir).read_bytes(), before)
        with self.assertRaises(storage.ReviewConflict):
            storage.save_review(viewer_dir, [], expected_revision=0)
        self.assertEqual(self.decision_file(viewer_dir).read_bytes(), before)

    def test_preparation_does_not_mutate_source_objects(self):
        epochs_before = self.epochs.get_data(copy=True)
        raw_before = self.raw.get_data().copy()
        labels_before = self.labels.copy(deep=True)
        transform_before = self.ica.unmixing_matrix_.copy()
        self.prepare()
        np.testing.assert_array_equal(self.epochs.get_data(), epochs_before)
        np.testing.assert_array_equal(self.raw.get_data(), raw_before)
        np.testing.assert_array_equal(self.ica.unmixing_matrix_, transform_before)
        pd.testing.assert_frame_equal(self.labels, labels_before)
        self.assertEqual(self.ica.exclude, [0])

    def test_http_autosave_is_durable_and_stale_writer_gets_conflict(self):
        viewer_dir = self.prepare()
        url = server.open_viewer(viewer_dir, open_browser=False)
        self.addCleanup(server.close_viewer, viewer_dir)
        endpoint = url.rsplit("/", 1)[0] + "/api/review"
        local_http = build_opener(ProxyHandler({}))
        with local_http.open(endpoint) as response:
            initial = json.load(response)
        request = Request(endpoint, data=json.dumps({
            "exclude": [2, 3], "expected_revision": initial["revision"],
        }).encode(), headers={"Content-Type": "application/json"}, method="PUT")
        with local_http.open(request) as response:
            saved = json.load(response)
        self.assertEqual(saved["exclude"], [2, 3])
        self.assertEqual(storage.load_exclusions(viewer_dir, ica=self.ica), [2, 3])
        with self.assertRaises(HTTPError) as error:
            local_http.open(request)
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(json.load(error.exception)["review"]["exclude"], [2, 3])
        self.assertEqual(storage.read_review(viewer_dir)["revision"], saved["revision"])
        server.close_viewer(viewer_dir)
        self.assertEqual(storage.load_exclusions(viewer_dir, ica=self.ica), [2, 3])


class TestViewerBuild(unittest.TestCase):
    def test_fresh_fit_preserves_native_mixing_matrix_across_fif_round_trip(self):
        ica, epochs, raw, labels = make_inputs(mixed=True)
        original_mixing = ica.mixing_matrix_.copy()
        original_maps = ica.get_components().copy()
        original_unmixing = ica.unmixing_matrix_.copy()
        with tempfile.TemporaryDirectory(prefix="ica-viewer-mixing-") as tmp:
            tmp = Path(tmp)
            fif_path = tmp / "roundtrip-ica.fif"
            ica.save(fif_path, verbose=False)
            restored = mne.preprocessing.read_ica(fif_path, verbose=False)
            # MNE recomputes pinv after loading the reordered unmixing rows;
            # the fitted, reordered mixing columns differ by floating-point ULPs.
            self.assertFalse(np.array_equal(restored.mixing_matrix_, original_mixing))
            np.testing.assert_array_equal(restored.unmixing_matrix_, original_unmixing)
            eeg_path = tmp / "sub-mixing_eeg.fif"
            eeg_path.write_bytes(b"source recording")
            viewer_dir = storage.prepare_bundle(
                ica, epochs, raw, eeg_path=eeg_path,
                reference_channel="Cz", iclabel_table=labels,
            )
            manifest = json.loads((viewer_dir / "manifest.json").read_text())
            np.testing.assert_array_equal(
                np.load(viewer_dir / manifest["files"]["mixing_matrix"], allow_pickle=False),
                original_mixing,
            )
            storage.save_review(viewer_dir, [], expected_revision=0)
            decision_path = viewer_dir / manifest["files"]["decision"]
            saved_bytes = decision_path.read_bytes()
            self.assertEqual(storage.prepare_bundle(
                ica, epochs, raw, eeg_path=eeg_path,
                reference_channel="Cz", iclabel_table=labels,
            ), viewer_dir)
            self.assertEqual(decision_path.read_bytes(), saved_bytes)
            shutil.rmtree(storage.cache_directory(viewer_dir))
            cache = storage.ensure_cache(viewer_dir)
            payload = json.loads((cache / "review_metadata.json").read_text())
            voltage_maps = ica.pre_whitener_ * original_maps * 1e6
            for component in payload["components"]:
                index = component["id"]
                channel = np.argmax(np.abs(voltage_maps[:, index]))
                self.assertEqual(component["voltage_channel"], ica.ch_names[channel])
                self.assertEqual(component["voltage_uV_per_source_unit"], voltage_maps[channel, index])
            direct = tmp / "direct"
            direct.mkdir()
            builder._topomaps(direct, ica)
            for image in (direct / "topomaps").glob("*.png"):
                self.assertEqual((cache / "topomaps" / image.name).read_bytes(), image.read_bytes())
        np.testing.assert_array_equal(ica.mixing_matrix_, original_mixing)
        np.testing.assert_array_equal(ica.unmixing_matrix_, original_unmixing)
        np.testing.assert_array_equal(ica.get_components(), original_maps)
        self.assertEqual(ica.exclude, [0])

    def test_nonunity_fif_calibration_preserves_exact_samples_when_rebuilding(self):
        ica, epochs, raw, labels = make_inputs()
        epochs.info["chs"][0]["cal"] = 2.415431e-5
        original = epochs.get_data(copy=True)
        original_sources = ica.get_sources(epochs).get_data().transpose(1, 0, 2).reshape(4, -1)
        with tempfile.TemporaryDirectory(prefix="ica-viewer-calibration-") as tmp:
            eeg_path = Path(tmp) / "sub-calibration_eeg.fif"
            eeg_path.write_bytes(b"source recording")
            viewer_dir = storage.prepare_bundle(
                ica, epochs, raw, eeg_path=eeg_path,
                reference_channel="Cz", iclabel_table=labels,
            )
            archived = mne.read_epochs(viewer_dir / "inputs" / "training-epo.fif",
                                       preload=True, verbose=False)
            # This fixture exercises FIF's calibration-rounding case, rather
            # than accidentally testing the already exact unity-calibration path.
            self.assertFalse(np.array_equal(archived.get_data(), original))
            manifest = json.loads((viewer_dir / "manifest.json").read_text())
            np.testing.assert_array_equal(
                np.load(viewer_dir / manifest["files"]["exact_data"], allow_pickle=False),
                original,
            )
            shutil.rmtree(storage.cache_directory(viewer_dir))
            rebuilt = storage.ensure_cache(viewer_dir)
            payload = json.loads((rebuilt / "review_metadata.json").read_text())
            for component, source in zip(payload["components"], original_sources):
                center = np.median(source)
                scale = 1.4826 * np.median(np.abs(source - center))
                self.assertEqual(component["source_median"], center)
                self.assertEqual(component["source_scale"], scale)
                script = (rebuilt / "data" / f'{component["name"]}.js').read_text()
                normalized = np.frombuffer(base64.b64decode(script.split('"')[1]), dtype="<f4")
                np.testing.assert_array_equal(normalized, ((source - center) / scale).astype("<f4"))
        np.testing.assert_array_equal(epochs.get_data(), original)
        self.assertEqual(epochs.info["chs"][0]["cal"], 2.415431e-5)
        self.assertEqual(ica.exclude, [0])

    def test_removed_time_jump_is_not_detected_but_contiguous_step_is(self):
        rng = np.random.default_rng(9)
        source = rng.normal(scale=0.01, size=1000)
        source[500:] += 1000
        gap, _ = builder._analyze_component(source, [[0, 500], [500, 1000]], 250)
        continuous, _ = builder._analyze_component(source, [[0, 1000]], 250)
        self.assertFalse(gap["threshold_runs"])
        self.assertTrue(continuous["threshold_runs"])
        self.assertTrue(any(run["start_s"] <= 2 <= run["end_s"]
                            for run in continuous["threshold_runs"]))

    def test_build_uses_exact_sources_native_scale_and_original_time_map(self):
        ica, epochs, raw, labels = make_inputs()
        with tempfile.TemporaryDirectory(prefix="ica-viewer-build-") as tmp:
            eeg_path = Path(tmp) / "sub-test_eeg.fif"
            eeg_path.write_bytes(b"source recording")
            viewer_dir = storage.prepare_bundle(
                ica, epochs, raw, eeg_path=eeg_path,
                reference_channel="Cz", iclabel_table=labels,
            )
            epoch_table = pd.read_csv(viewer_dir / "inputs" / "epochs.csv")
            payload = builder.build_viewer(
                storage.cache_directory(viewer_dir), ica, epochs, labels,
                recording_filename=eeg_path.name, reference_channel="Cz",
                epoch_table=epoch_table, review_id="test-review", initial_excluded=[0],
            )
            self.assertEqual(payload["segments"], [[0, 1000], [1000, 1500]])
            self.assertEqual(payload["discontinuities"][0]["retained_sample"], 1000)
            sources = ica.get_sources(epochs).get_data().transpose(1, 0, 2).reshape(4, -1)
            for component, source in zip(payload["components"], sources):
                name = component["name"]
                script = (storage.cache_directory(viewer_dir) / "data" / f"{name}.js").read_text()
                encoded = script.split('"')[1]
                normalized = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
                reconstructed = normalized * component["source_scale"] + component["source_median"]
                np.testing.assert_allclose(reconstructed, source, atol=2e-6, rtol=1e-6)
                # Verify physical calibration against MNE's actual removal,
                # independently of the builder's mixing-weight formula.
                remove_one = ica.copy()
                remove_one.exclude = []
                with mne.use_log_level("ERROR"):
                    cleaned = remove_one.apply(epochs.copy(), exclude=[component["id"]])
                channel = epochs.ch_names.index(component["voltage_channel"])
                removed_uv = (epochs.get_data() - cleaned.get_data())[:, channel, :].ravel() * 1e6
                np.testing.assert_allclose(
                    removed_uv, source * component["voltage_uV_per_source_unit"],
                    atol=1e-9, rtol=1e-10,
                )
            self.assertTrue((storage.cache_directory(viewer_dir) / "index.html").exists())
            self.assertTrue((storage.cache_directory(viewer_dir) / "topomaps" / "ICA000.png").exists())


if __name__ == "__main__":
    unittest.main()
