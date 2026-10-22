"""Reopen the newest saved fit without fitting or reading archived EEG again."""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener

import mne


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step3_ica_cleaning as step3
from step3_ica_viewer import build as builder, server, storage
from test_ica_viewer import make_inputs


class TestViewerReopen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(prefix="ica-reopen-fixture-", dir="/tmp")
        cls.addClassCleanup(cls.fixture.cleanup)
        eeg_path = (Path(cls.fixture.name) / "sub-test_task-test_eeg.fif").resolve()
        eeg_path.write_bytes(b"immutable original EEG recording")
        ica, epochs, raw, labels = make_inputs()
        cls.saved_fits = []
        for index, timestamp in enumerate(("2026-10-20T10:00:00Z", "2026-10-21T10:00:00Z")):
            transform = ica.copy()
            if index:
                transform.unmixing_matrix_[0, 0] += 0.125
                transform._update_mixing_matrix()
            with patch.object(storage, "_now", return_value=timestamp):
                cls.saved_fits.append(storage.prepare_bundle(
                    transform, epochs, raw, eeg_path=eeg_path,
                    reference_channel="Cz", iclabel_table=labels,
                ))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ica-reopen-test-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.eeg_path = (Path(self.tmp.name) / "sub-test_task-test_eeg.fif").resolve()
        self.eeg_path.write_bytes(b"immutable original EEG recording")
        self.parent = self.eeg_path.parent / ".ica_viewer" / self.eeg_path.name
        self.bundles = []
        for saved in self.saved_fits:
            bundle = self.parent / saved.name
            shutil.copytree(saved, bundle)
            self.update_manifest(bundle, source_path=str(self.eeg_path))
            self.bundles.append(bundle)

    def update_manifest(self, bundle, **changes):
        path = bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(changes)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def decision_file(self, bundle):
        manifest = json.loads((bundle / "manifest.json").read_text())
        return bundle / manifest["files"]["decision"]

    def test_newest_fit_is_independent_of_manual_review_and_cache_modification(self):
        older, newer = self.bundles
        storage.save_review(older, [2, 3], expected_revision=0)
        # A review or regenerated HTML from an old fit does not make it a new fit.
        future_ns = 2000000000000000000
        for path in (older, self.decision_file(older), storage.cache_directory(older) / "index.html"):
            os.utime(path, ns=(future_ns, future_ns))
        self.assertEqual(storage.find_latest_bundle(self.eeg_path), newer)

    def test_creation_timestamp_ties_use_manifest_mtime_then_directory_name(self):
        older, newer = self.bundles
        for bundle in self.bundles:
            self.update_manifest(bundle, created_at_utc="2026-10-21T10:00:00Z")
        timestamp_ns = 1800000000000000000
        os.utime(older / "manifest.json", ns=(timestamp_ns, timestamp_ns))
        os.utime(newer / "manifest.json", ns=(timestamp_ns + 1000, timestamp_ns + 1000))
        self.assertEqual(storage.find_latest_bundle(self.eeg_path), newer)
        os.utime(older / "manifest.json", ns=(timestamp_ns + 2000, timestamp_ns + 2000))
        self.assertEqual(storage.find_latest_bundle(self.eeg_path), older)
        os.utime(newer / "manifest.json", ns=(timestamp_ns + 2000, timestamp_ns + 2000))
        self.assertEqual(storage.find_latest_bundle(self.eeg_path), max(self.bundles, key=lambda item: item.name))

    def test_lookup_is_scoped_to_recording_and_ignores_partial_directories(self):
        other_path = self.eeg_path.with_name("sub-other_task-test_eeg.fif")
        other_bundle = other_path.parent / ".ica_viewer" / other_path.name / self.bundles[0].name
        shutil.copytree(self.bundles[0], other_bundle)
        self.update_manifest(other_bundle, source_path=str(other_path), recording_filename=other_path.name,
                             created_at_utc="2099-10-21T10:00:00Z")
        partial = self.parent / ".preparing-incomplete"
        shutil.copytree(self.bundles[0], partial)
        self.update_manifest(partial, created_at_utc="2099-10-21T10:00:00Z")
        self.assertEqual(storage.find_latest_bundle(self.eeg_path), self.bundles[1])

    def test_source_path_and_filename_must_both_match(self):
        older, newer = self.bundles
        original = (newer / "manifest.json").read_bytes()
        for changes in (
                {"source_path": str(self.eeg_path.with_name("different_eeg.fif"))},
                {"recording_filename": "different_eeg.fif"}):
            with self.subTest(changes=changes):
                (newer / "manifest.json").write_bytes(original)
                self.update_manifest(newer, **changes)
                self.assertEqual(storage.find_latest_bundle(self.eeg_path), older)

    def test_missing_prepared_recording_has_actionable_error(self):
        missing = self.eeg_path.with_name("not_prepared_eeg.fif")
        missing.write_bytes(b"unprepared EEG")
        with self.assertRaisesRegex(FileNotFoundError, "prepare_ica_viewer"):
            storage.find_latest_bundle(missing)
        with self.assertRaisesRegex(FileNotFoundError, "prepare_ica_viewer"):
            step3.open_ica_viewer(eeg_path=missing, open_browser=False)

    def test_keyword_lookup_can_use_the_original_path_without_the_original_file(self):
        self.eeg_path.unlink()
        self.assertEqual(storage.find_latest_bundle(self.eeg_path), self.bundles[1])
        with patch.object(server, "open_viewer", return_value="http://127.0.0.1/viewer") as open_viewer:
            with redirect_stdout(StringIO()):
                self.assertEqual(step3.open_ica_viewer(eeg_path=self.eeg_path, open_browser=False),
                                 "http://127.0.0.1/viewer")
            open_viewer.assert_called_once_with(self.bundles[1], open_browser=False)

    def test_malformed_saved_manifest_does_not_silently_open_another_fit(self):
        (self.bundles[1] / "manifest.json").write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            storage.find_latest_bundle(self.eeg_path)

    def test_reopening_reuses_cache_and_preserves_final_version_without_eeg_reads(self):
        older, newer = self.bundles
        storage.save_review(older, [1], expected_revision=0)
        storage.create_review_version(newer, expected_revision=0)
        storage.save_review(newer, [2, 3], expected_revision=1, version="v2")
        storage.set_stable_review_version(newer, "v2", expected_revision=2)
        decisions_before = {bundle: self.decision_file(bundle).read_bytes() for bundle in self.bundles}
        path_open = Path.open

        def refuse_eeg_read(path, *args, **kwargs):
            if path.resolve() == self.eeg_path.resolve():
                raise AssertionError("The original EEG must not be opened to reopen its viewer.")
            return path_open(path, *args, **kwargs)

        with ExitStack() as stack:
            for target, name in (
                    (storage, "_verify_archived"), (storage, "prepare_bundle"),
                    (storage, "_source_files"), (builder, "build_viewer"),
                    (mne, "read_epochs"), (mne.preprocessing, "read_ica"),
                    (step3, "fit_ica"), (step3, "prepare_ica_viewer")):
                stack.enter_context(patch.object(target, name, side_effect=AssertionError(
                    f"Unexpected expensive operation during cache reuse: {name}")))
            stack.enter_context(patch.object(Path, "open", refuse_eeg_read))
            stack.enter_context(redirect_stdout(StringIO()))
            urls = [
                step3.open_ica_viewer(newer, open_browser=False),
                step3.open_ica_viewer(eeg_path=self.eeg_path, open_browser=False),
                step3.open_ica_viewer(self.eeg_path, open_browser=False),
            ]
            self.addCleanup(server.close_viewer, newer)
            self.assertEqual(urls, [urls[0]] * 3)
            local_http = build_opener(ProxyHandler({}))
            with local_http.open(urls[0].rsplit("/", 1)[0] + "/api/review") as response:
                review = json.load(response)
            self.assertEqual(review["active_version"], "v2")
            self.assertEqual(review["stable_version"], "v2")
            self.assertEqual(review["exclude"], [2, 3])
        for bundle, previous in decisions_before.items():
            self.assertEqual(self.decision_file(bundle).read_bytes(), previous)

    def test_public_helper_rejects_ambiguous_or_missing_arguments(self):
        with patch.object(server, "open_viewer") as open_viewer:
            with self.assertRaises(ValueError):
                step3.open_ica_viewer(open_browser=False)
            with self.assertRaises(ValueError):
                step3.open_ica_viewer(self.bundles[0], eeg_path=self.eeg_path, open_browser=False)
            open_viewer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
