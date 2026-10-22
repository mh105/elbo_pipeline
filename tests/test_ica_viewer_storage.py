"""Tolerant archive round trips must not weaken analysis identity checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from step3_ica_viewer import storage
from test_ica_viewer import make_inputs


class TestViewerArchiveIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ica, cls.epochs, cls.raw, cls.labels = make_inputs(mixed=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ica-storage-test-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.eeg_path = Path(self.tmp.name) / "sub-test_eeg.fif"
        self.eeg_path.write_bytes(b"immutable original EEG recording")
        # These tests exercise archive/identity decisions independently of
        # plotting, which the existing builder and HTTP tests cover.
        cache = patch.object(storage, "ensure_cache", side_effect=lambda path: Path(path) / "cache")
        cache.start()
        self.addCleanup(cache.stop)

    def prepare(self, ica=None, epochs=None, raw=None, reference_channel="Cz"):
        return storage.prepare_bundle(
            self.ica if ica is None else ica,
            self.epochs if epochs is None else epochs,
            self.raw if raw is None else raw,
            eeg_path=self.eeg_path, reference_channel=reference_channel,
            iclabel_table=self.labels,
        )

    def manifest(self, bundle):
        return json.loads((bundle / "manifest.json").read_text())

    def test_fif_rounding_reuses_original_fit_and_preserves_saved_labels(self):
        epochs = self.epochs.copy()
        epochs.info["chs"][0]["cal"] = 2.415431e-5
        bundle = self.prepare(epochs=epochs)
        manifest = self.manifest(bundle)
        self.assertEqual(manifest["archive_version"], 2)
        storage.save_review(bundle, [1, 3], expected_revision=0)
        decisions = bundle / manifest["files"]["decision"]
        previous = decisions.read_bytes()
        restored_epochs = mne.read_epochs(bundle / "inputs/training-epo.fif",
                                          preload=True, verbose=False)
        restored_ica = mne.preprocessing.read_ica(bundle / "inputs/fitted-ica.fif", verbose=False)
        self.assertFalse(np.array_equal(restored_epochs.get_data(), epochs.get_data()))
        self.assertFalse(np.array_equal(restored_ica.mixing_matrix_, self.ica.mixing_matrix_))
        self.assertEqual(self.prepare(ica=restored_ica, epochs=restored_epochs), bundle)
        self.assertEqual(storage.load_exclusions(bundle, ica=restored_ica), [1, 3])
        self.assertEqual(decisions.read_bytes(), previous)
        self.assertEqual(list(bundle.parent.glob("*/manifest.json")), [bundle / "manifest.json"])
        self.assertFalse(list((bundle / "inputs").glob("*.npy")))

    def test_small_numerical_rounding_reuses_fit_but_real_sample_change_does_not(self):
        bundle = self.prepare()
        rounded = self.epochs.copy()
        rounded._data *= 1 + 1e-8
        self.assertEqual(self.prepare(epochs=rounded), bundle)
        changed = self.epochs.copy()
        changed._data[1, 2, 75] += 1e-9
        self.assertNotEqual(self.prepare(epochs=changed), bundle)
        rounded_ica = self.ica.copy()
        for field in ("mixing_matrix_", "unmixing_matrix_", "pca_components_",
                      "pca_mean_", "pre_whitener_"):
            values = getattr(rounded_ica, field)
            setattr(rounded_ica, field, np.nextafter(values, np.inf))
        self.assertEqual(self.prepare(ica=rounded_ica), bundle)
        self.assertEqual(storage.load_exclusions(bundle, ica=rounded_ica), [0])

    def test_changed_mapping_geometry_reference_and_recording_get_distinct_fits(self):
        bundle = self.prepare()
        mapping = self.epochs.copy()
        mapping.events[-1, 0] -= 1000
        selection = self.epochs.copy()
        selection.selection[-1] += 1
        selection.drop_log = ((), (), ("USER",), ())
        geometry = self.epochs.copy()
        geometry.info["chs"][0]["loc"][0] += 1e-3
        geometry_model = self.ica.copy()
        geometry_model.info["chs"][0]["loc"][0] += 1e-3
        event_id = self.epochs.copy()
        event_id.event_id = {"different event meaning": 1}
        for label, epochs, model in (
                ("event samples", mapping, self.ica), ("retained epochs", selection, self.ica),
                ("electrodes", geometry, geometry_model), ("event meaning", event_id, self.ica)):
            with self.subTest(identity=label):
                self.assertNotEqual(self.prepare(ica=model, epochs=epochs), bundle)
        alternate_reference = self.raw.copy()
        info = mne.create_info(["Pz"], 1000, "eeg")
        zero = mne.io.RawArray(np.zeros((1, self.raw.n_times)), info,
                                first_samp=self.raw.first_samp, verbose=False)
        alternate_reference.add_channels([zero], force_update_info=True)
        self.assertNotEqual(self.prepare(raw=alternate_reference, reference_channel="Pz"), bundle)
        self.eeg_path.write_bytes(b"different original recording")
        self.assertNotEqual(self.prepare(), bundle)

    def test_transform_component_order_and_reference_semantics_cannot_load_old_labels(self):
        bundle = self.prepare()
        numerical = self.ica.copy()
        numerical.unmixing_matrix_[0, 0] += 0.01
        numerical._update_mixing_matrix()
        reordered = self.ica.copy()
        reordered.unmixing_matrix_ = reordered.unmixing_matrix_[[1, 0, 2, 3]]
        reordered._update_mixing_matrix()
        reference = self.ica.copy()
        with reference.info._unlock():
            reference.info["custom_ref_applied"] = 1
        geometry = self.ica.copy()
        geometry.info["chs"][0]["loc"][0] += 1e-3
        frames = self.ica.copy()
        frames.info["chs"][0]["coord_frame"] = 1
        for label, model in (("transform", numerical), ("component order", reordered),
                             ("reference state", reference), ("electrode positions", geometry),
                             ("electrode frames", frames)):
            with self.subTest(identity=label), self.assertRaises(ValueError):
                storage.load_exclusions(bundle, ica=model)
        # Compare each model array at its own scale. A broad absolute tolerance
        # suitable for unitless PCA matrices would hide a changed whitener.
        for field in ("mixing_matrix_", "unmixing_matrix_", "pca_components_",
                      "pca_mean_", "pre_whitener_"):
            changed = self.ica.copy()
            values = getattr(changed, field)
            values.flat[0] += max(np.max(np.abs(values)), 1e-30) * 1e-6
            with self.subTest(array=field), self.assertRaises(ValueError):
                storage.load_exclusions(bundle, ica=changed)
        self.assertNotEqual(self.prepare(ica=numerical), bundle)
        self.assertNotEqual(self.prepare(ica=reordered), bundle)

    def test_channel_order_filter_sampling_and_nonzero_reference_are_rejected(self):
        self.prepare()
        order = self.epochs.copy().reorder_channels(["Fp2", "Fp1", "C3", "C4"])
        filtering = self.epochs.copy()
        with filtering.info._unlock():
            filtering.info["lowpass"] = 90.0
        sampling = self.epochs.copy()
        with sampling.info._unlock():
            sampling.info["sfreq"] = 200.0
        for label, epochs in (("channel order", order), ("filter", filtering),
                              ("sampling rate", sampling)):
            with self.subTest(identity=label), self.assertRaises(ValueError):
                self.prepare(epochs=epochs)
        bad_reference = self.raw.copy()
        bad_reference._data[-1, 0] = 1e-6
        with self.assertRaises(ValueError):
            self.prepare(raw=bad_reference)
        inconsistent_geometry = self.ica.copy()
        inconsistent_geometry.info["chs"][0]["loc"][0] += 1e-3
        with self.assertRaises(ValueError):
            self.prepare(ica=inconsistent_geometry)

    def legacy_bundle(self):
        """Recreate the exact-supplement fields used by existing schema-1 fits."""
        epochs = self.epochs.copy()
        epochs.info["chs"][0]["cal"] = 2.415431e-5
        bundle = self.prepare(epochs=epochs)
        manifest = self.manifest(bundle)
        manifest.pop("archive_version", None)
        manifest.pop("archive_model_identity", None)
        manifest.pop("archive_training_samples_identity", None)
        for key, filename, values in (
                ("exact_data", "training-data.npy", epochs.get_data()),
                ("mixing_matrix", "mixing-matrix.npy", self.ica.mixing_matrix_)):
            path = bundle / "inputs" / filename
            np.save(path, values, allow_pickle=False)
            manifest["files"][key] = str(path.relative_to(bundle))
            manifest["input_files"].append(dict(
                path=str(path.relative_to(bundle)), sha256=storage._file_hash(path)))
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        return bundle

    def test_legacy_migration_preserves_identity_and_all_label_versions(self):
        bundle = self.legacy_bundle()
        storage.save_review(bundle, [1], expected_revision=0)
        storage.create_review_version(bundle, expected_revision=1)
        storage.save_review(bundle, [2, 3], expected_revision=2, version="v2")
        storage.set_stable_review_version(bundle, "v2", expected_revision=3)
        old = self.manifest(bundle)
        decision = bundle / old["files"]["decision"]
        saved = decision.read_bytes()
        self.assertEqual(storage.migrate_bundle(bundle), bundle)
        migrated = self.manifest(bundle)
        self.assertEqual(migrated["fit_id"], old["fit_id"])
        self.assertEqual(migrated["archive_version"], 2)
        self.assertEqual(decision.read_bytes(), saved)
        self.assertEqual(storage.load_exclusions(bundle, ica=self.ica), [2, 3])
        self.assertEqual(storage.read_review(bundle, "v1")["exclude"], [1])
        self.assertFalse(list((bundle / "inputs").glob("*.npy")))
        self.assertNotIn("exact_data", migrated["files"])
        self.assertNotIn("mixing_matrix", migrated["files"])
        self.assertEqual(self.prepare(), bundle)

    def test_migration_validates_archives_before_removing_supplements(self):
        bundle = self.legacy_bundle()
        manifest_path = bundle / "manifest.json"
        manifest = self.manifest(bundle)
        supplement = bundle / manifest["files"]["exact_data"]
        # Checksums alone cannot replace the model/sample identity checks.
        values = np.load(supplement, allow_pickle=False)
        values[0, 0, 0] += 1e-6
        np.save(supplement, values, allow_pickle=False)
        next(item for item in manifest["input_files"]
             if item["path"] == manifest["files"]["exact_data"])["sha256"] = storage._file_hash(supplement)
        manifest_path.write_text(json.dumps(manifest))
        before = manifest_path.read_bytes()
        with self.assertRaises(ValueError):
            storage.migrate_bundle(bundle)
        self.assertTrue(supplement.is_file())
        self.assertTrue((bundle / manifest["files"]["mixing_matrix"]).is_file())
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_archive_corruption_is_rejected_before_archived_data_is_loaded(self):
        bundle = self.prepare()
        path = bundle / "inputs/training-epo.fif"
        content = bytearray(path.read_bytes())
        content[-100] ^= 1
        path.write_bytes(content)
        with self.assertRaises(ValueError):
            storage._load_archived(bundle, self.manifest(bundle))


if __name__ == "__main__":
    unittest.main()
