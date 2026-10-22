"""Reconstructed EEG envelopes agree with independent MNE ICA application."""

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from step3_ica_viewer import envelope, server, storage
from test_ica_viewer import make_inputs


def mne_reconstruction(ica, epochs, exclude):
    """Replace the full exclusion list instead of unioning with the fit's list."""
    transform = ica.copy()
    transform.exclude = []
    with mne.use_log_level("ERROR"):
        cleaned = transform.apply(epochs.copy(), exclude=exclude)
    return cleaned.get_data(picks=ica.ch_names).transpose(1, 0, 2).reshape(
        len(ica.ch_names), -1
    ) * 1e6


def expected_bins(data_uv, sfreq, start_s, end_s, segments):
    """Direct extrema over every channel/sample, including clipped edge bins."""
    first, last = round(start_s * sfreq), round(end_s * sfreq)
    width = round(0.020 * sfreq)
    starts, ends, lower, upper, groups = [], [], [], [], []
    for segment_first, segment_last in segments:
        start, end = max(first, segment_first), min(last, segment_last)
        if start >= end:
            continue
        group_first = len(starts)
        while start < end:
            stop = min(end, (start // width + 1) * width)
            starts.append(start / sfreq)
            ends.append(stop / sfreq)
            lower.append(data_uv[:, start:stop].min())
            upper.append(data_uv[:, start:stop].max())
            start = stop
        groups.append([group_first, len(starts)])
    return starts, ends, lower, upper, groups


class TestIcaEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ica, cls.epochs, cls.raw, cls.labels = make_inputs(mixed=True)
        cls.reduced = mne.preprocessing.ICA(
            n_components=2, method="infomax", fit_params={"extended": True},
            max_iter=24, rng=518,
        )
        with warnings.catch_warnings(), mne.use_log_level("ERROR"):
            warnings.simplefilter("ignore", RuntimeWarning)
            cls.reduced.fit(cls.epochs, picks="eeg")
        cls.reduced.n_pca_components = 3
        cls.reduced.exclude = [0]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ica-envelope-test-")
        self.addCleanup(self.tmp.cleanup)

    def write_cache(self, ica, segments):
        cache = Path(self.tmp.name) / "cache"
        cache.mkdir()
        sources = ica.get_sources(self.epochs).get_data().transpose(1, 0, 2).reshape(
            ica.n_components_, -1
        )
        envelope._write_reconstruction(cache, ica, self.epochs, sources, segments)
        return envelope.EnvelopeData(cache)

    def assert_envelope(self, result, ica, exclude, start_s, end_s, segments,
                        *, key="current"):
        expected = expected_bins(
            mne_reconstruction(ica, self.epochs, exclude), 250, start_s, end_s,
            segments,
        )
        self.assertEqual(result["sfreq"], 250)
        self.assertEqual(result["bin_samples"], 5)
        self.assertEqual(result["start_s"], round(start_s * 250) / 250)
        self.assertEqual(result["end_s"], round(end_s * 250) / 250)
        np.testing.assert_array_equal(result["bin_starts_s"], expected[0])
        np.testing.assert_array_equal(result["bin_ends_s"], expected[1])
        self.assertEqual(result["segments"], expected[4])
        np.testing.assert_allclose(result[key]["lower_uv"], expected[2],
                                   rtol=1e-11, atol=1e-9)
        np.testing.assert_allclose(result[key]["upper_uv"], expected[3],
                                   rtol=1e-11, atol=1e-9)

    def test_complete_exclusion_lists_match_mne_without_mutating_inputs(self):
        segments = [[0, 1000], [1000, 1500]]
        samples = self.epochs.get_data(copy=True)
        mixing = self.ica.mixing_matrix_.copy()
        unmixing = self.ica.unmixing_matrix_.copy()
        data = self.write_cache(self.ica, segments)
        for exclude in ([], [1], [0, 2], [0, 1, 2, 3]):
            with self.subTest(exclude=exclude):
                result = data.calculate(0, 6, exclude)
                self.assert_envelope(result, self.ica, exclude, 0, 6, segments)
                self.assertIsNone(result["before"])
        np.testing.assert_array_equal(self.epochs.get_data(), samples)
        np.testing.assert_array_equal(self.ica.mixing_matrix_, mixing)
        np.testing.assert_array_equal(self.ica.unmixing_matrix_, unmixing)
        self.assertEqual(self.ica.exclude, [0])

    def test_reduced_ica_preserves_mean_and_configured_residual_pca(self):
        segments = [[0, 1500]]
        data = self.write_cache(self.reduced, segments)
        for exclude in ([], [1], [0, 1]):
            with self.subTest(exclude=exclude):
                result = data.calculate(0, 6, exclude)
                self.assert_envelope(result, self.reduced, exclude, 0, 6, segments)
        rejected = mne_reconstruction(self.reduced, self.epochs, [0, 1])
        # Rejecting every IC still preserves the third PCA dimension and mean.
        self.assertGreater(np.ptp(rejected), 1.0)
        self.assertGreater(np.abs(rejected.mean(axis=1)).max(), 0.01)
        self.assertEqual(self.reduced.n_pca_components, 3)
        self.assertEqual(self.reduced.exclude, [0])

    def test_latest_toggle_comparison_and_bins_do_not_cross_removed_time(self):
        # A non-grid-aligned join also splits the 20 ms bin containing it.
        segments = [[0, 997], [997, 1500]]
        data = self.write_cache(self.ica, segments)
        first, last = 3.972, 4.028
        for before, after in (([1], [1, 2]), ([1, 2], [2]), ([2], [0, 3])):
            with self.subTest(before=before, after=after):
                result = data.calculate(first, last, after, before)
                self.assert_envelope(result, self.ica, after, first, last, segments)
                self.assert_envelope(result, self.ica, before, first, last, segments,
                                     key="before")
        self.assertEqual(result["bin_starts_s"], [3.972, 3.98, 3.988, 4.0, 4.02])
        self.assertEqual(result["bin_ends_s"], [3.98, 3.988, 4.0, 4.02, 4.028])
        self.assertEqual(result["segments"], [[0, 2], [2, 5]])

    def test_full_recording_is_reconstructed_in_bounded_chunks(self):
        epochs = mne.concatenate_epochs([self.epochs] * 3, verbose="ERROR")
        segments = [[0, 2497], [2497, 4500]]
        cache = Path(self.tmp.name) / "cache"
        cache.mkdir()
        sources = self.reduced.get_sources(epochs).get_data().transpose(1, 0, 2).reshape(
            self.reduced.n_components_, -1
        )
        envelope._write_reconstruction(cache, self.reduced, epochs, sources, segments)
        data = envelope.EnvelopeData(cache)
        # Include clipped edge bins and a real join next to a chunk boundary.
        first, last = 0.012, 17.988
        for previous, excluded in (([0], [0, 1]), ([0, 1], [1])):
            with self.subTest(previous=previous, excluded=excluded):
                with mock.patch.object(data, "_reconstruct", wraps=data._reconstruct) as reconstruct:
                    result = data.calculate(first, last, excluded, previous)
                self.assertGreater(len(reconstruct.call_args_list), 1)
                for call in reconstruct.call_args_list:
                    self.assertLessEqual(call.args[1] - call.args[0], 2500)
                for key, exclusions in (("current", excluded), ("before", previous)):
                    expected = expected_bins(mne_reconstruction(self.reduced, epochs, exclusions),
                                             250, first, last, segments)
                    np.testing.assert_array_equal(result["bin_starts_s"], expected[0])
                    np.testing.assert_array_equal(result["bin_ends_s"], expected[1])
                    np.testing.assert_allclose(result[key]["lower_uv"], expected[2],
                                               rtol=1e-11, atol=1e-9)
                    np.testing.assert_allclose(result[key]["upper_uv"], expected[3],
                                               rtol=1e-11, atol=1e-9)
                    self.assertEqual(result["segments"], expected[4])

    def test_http_envelope_is_read_only_and_rejects_invalid_requests(self):
        eeg_path = Path(self.tmp.name) / "sub-envelope_eeg.fif"
        eeg_path.write_bytes(b"original recording")
        viewer_dir = storage.prepare_bundle(
            self.ica, self.epochs, self.raw, eeg_path=eeg_path,
            reference_channel="Cz", iclabel_table=self.labels,
        )
        review_path = next(viewer_dir.glob("*_ica_exclude.json"))
        original_review = review_path.read_bytes()
        url = server.open_viewer(viewer_dir, open_browser=False)
        self.addCleanup(server.close_viewer, viewer_dir)
        endpoint = url.rsplit("/", 1)[0] + "/api/envelope"
        local_http = build_opener(ProxyHandler({}))

        def request(body):
            item = Request(endpoint, data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"}, method="POST")
            with local_http.open(item) as response:
                return json.load(response)

        valid = dict(start_s=0.008, end_s=4.028, exclude=[1, 2], before_exclude=[1])
        result = request(valid)
        self.assert_envelope(result, self.ica, [1, 2], 0.008, 4.028,
                             [[0, 1000], [1000, 1500]])
        self.assert_envelope(result, self.ica, [1], 0.008, 4.028,
                             [[0, 1000], [1000, 1500]], key="before")
        current = request(dict(valid, before_exclude=None))
        self.assertIsNone(current["before"])
        invalid = [dict(valid, exclude=value) for value in ([True], [1.0], [-1], [4], [1, 1])]
        invalid.extend([
            dict(valid, before_exclude=[4]), dict(valid, start_s=True),
            dict(valid, start_s=-1), dict(valid, start_s=5, end_s=4),
            dict(valid, start_s=float("nan")), dict(valid, end_s=float("inf")),
            dict(valid, extra=True), {key: value for key, value in valid.items()
                                      if key != "before_exclude"}, [],
        ])
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(HTTPError) as error:
                request(body)
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(review_path.read_bytes(), original_review)
        self.assertEqual(storage.load_exclusions(viewer_dir), [0])


if __name__ == "__main__":
    unittest.main()
