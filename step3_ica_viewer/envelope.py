"""Physical EEG reconstruction and 20-ms envelopes for ICA review."""

from pathlib import Path

import mne
import numpy as np


def _write_reconstruction(cache_dir, ica, fit_epochs, sources, segments):
    """Archive exact activations and the small model needed to match ICA.apply.

    Retain the PCA mean and any non-ICA PCA components included by MNE's
    configured reconstruction. No full reconstructed recording is stored.
    """
    if ica.noise_cov is not None:
        raise ValueError("Envelope reconstruction requires standardization-whitened ICA.")
    with mne.utils.use_log_level("ERROR"):
        n_pca = ica._check_n_pca_components(ica.n_pca_components)
    if not ica.n_components_ <= n_pca <= len(ica.pca_components_):
        raise ValueError("The ICA reconstruction has an invalid PCA component count.")
    directory = Path(cache_dir) / "reconstruction"
    directory.mkdir(exist_ok=True)
    np.save(directory / "sources.npy", np.asarray(sources, dtype=np.float64),
            allow_pickle=False)
    whitening = np.asarray(ica.pre_whitener_)
    mixing_uv = whitening * ica.get_components() * 1e6
    mean = (np.zeros(len(ica.ch_names)) if ica.pca_mean_ is None
            else np.asarray(ica.pca_mean_))
    mean_uv = whitening[:, 0] * mean * 1e6
    residual_vectors = ica.pca_components_[ica.n_components_:n_pca]
    residual_mixing_uv = whitening * residual_vectors.T * 1e6
    if len(residual_vectors):
        data = np.hstack(fit_epochs.get_data(picks=ica._get_picks(fit_epochs)))
        with mne.utils.use_log_level("ERROR"):
            data = ica._pre_whiten(data)
        data -= mean[:, None]
        residual_sources = residual_vectors @ data
        np.save(directory / "residual_sources.npy", residual_sources, allow_pickle=False)
    np.savez(directory / "model.npz", mixing_uv=mixing_uv, mean_uv=mean_uv,
             residual_mixing_uv=residual_mixing_uv,
             sfreq=float(fit_epochs.info["sfreq"]),
             segments=np.asarray(segments, dtype=np.int64))


class EnvelopeData:
    """Memory-map fitting activations and reconstruct bounded sample chunks."""

    def __init__(self, cache_dir):
        directory = Path(cache_dir) / "reconstruction"
        with np.load(directory / "model.npz", allow_pickle=False) as model:
            self.mixing_uv = model["mixing_uv"]
            self.mean_uv = model["mean_uv"]
            self.residual_mixing_uv = model["residual_mixing_uv"]
            self.sfreq = float(model["sfreq"])
            self.segments = model["segments"]
        self.sources = np.load(directory / "sources.npy", mmap_mode="r", allow_pickle=False)
        if (self.sources.ndim != 2 or self.mixing_uv.ndim != 2
                or self.residual_mixing_uv.ndim != 2):
            raise ValueError("The envelope reconstruction cache has invalid array dimensions.")
        self.residual_sources = (np.load(directory / "residual_sources.npy", mmap_mode="r",
                                        allow_pickle=False)
                                 if self.residual_mixing_uv.shape[1] else None)
        self.n_components, self.n_samples = self.sources.shape
        n_channels = self.mixing_uv.shape[0]
        if (self.sfreq != 250 or self.mixing_uv.shape[1] != self.n_components
                or self.mean_uv.shape != (n_channels,)
                or self.residual_mixing_uv.shape[0] != n_channels
                or (self.residual_sources is not None and self.residual_sources.shape !=
                    (self.residual_mixing_uv.shape[1], self.n_samples))
                or self.segments.ndim != 2 or self.segments.shape[1] != 2
                or not len(self.segments) or self.segments[0, 0] != 0
                or self.segments[-1, 1] != self.n_samples
                or np.any(self.segments[:, 1] <= self.segments[:, 0])
                or not np.array_equal(self.segments[1:, 0], self.segments[:-1, 1])
                or not np.isfinite(self.mixing_uv).all()
                or not np.isfinite(self.mean_uv).all()
                or not np.isfinite(self.residual_mixing_uv).all()):
            raise ValueError("The envelope reconstruction cache is inconsistent.")
        self.bin_samples = round(0.020 * self.sfreq)

    def _indices(self, values):
        if (not isinstance(values, list)
                or any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
                       or not 0 <= value < self.n_components for value in values)
                or len(set(values)) != len(values)):
            raise ValueError("Envelope exclusions must be unique valid component integers.")
        return set(values)

    def _reconstruct(self, start, end, exclude):
        keep = [index for index in range(self.n_components) if index not in exclude]
        data = self.mixing_uv[:, keep] @ self.sources[keep, start:end]
        data += self.mean_uv[:, None]
        if self.residual_sources is not None:
            data += self.residual_mixing_uv @ self.residual_sources[:, start:end]
        return data

    def calculate(self, start_s, end_s, exclude, before_exclude=None):
        """Return 20-ms channel/sample extrema, split at omitted-time joins."""
        if (any(isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
                for value in (start_s, end_s))
                or not np.isfinite([start_s, end_s]).all()
                or not 0 <= start_s < end_s <= self.n_samples / self.sfreq):
            raise ValueError("Envelope times must describe a finite retained-data window.")
        # Match the browser's rounding of nonnegative sample coordinates.
        start, end = (int(np.floor(value * self.sfreq + 0.5))
                      for value in (start_s, end_s))
        if end <= start:
            raise ValueError("The envelope window contains no samples.")
        excluded = self._indices(exclude)
        previous = None if before_exclude is None else self._indices(before_exclude)
        current = self._reconstruct(start, end, excluded)
        before = None
        if previous is not None:
            changed = excluded ^ previous
            if len(changed) == 1:
                index = changed.pop()
                contribution = self.mixing_uv[:, index, None] * self.sources[index, start:end]
                before = current + contribution if index in excluded else current - contribution
            else:
                before = self._reconstruct(start, end, previous)
        if not np.isfinite(current).all() or (before is not None and not np.isfinite(before).all()):
            raise ValueError("Reconstructed envelope data contain nonfinite samples.")

        # Global five-sample boundaries survive panning. Clip edge bins to the
        # visible window, and split a bin at every real removed-time join.
        boundaries = np.unique(np.r_[start,
            np.arange((start // self.bin_samples + 1) * self.bin_samples,
                      end, self.bin_samples),
            self.segments[1:, 0][(self.segments[1:, 0] > start) &
                                    (self.segments[1:, 0] < end)], end])
        starts, ends = boundaries[:-1], boundaries[1:]
        local_starts = starts - start
        def extrema(data):
            lower = np.minimum.reduceat(data.min(axis=0), local_starts)
            upper = np.maximum.reduceat(data.max(axis=0), local_starts)
            return dict(lower_uv=lower.tolist(), upper_uv=upper.tolist())

        bin_segments = []
        for segment_start, segment_end in self.segments:
            if segment_end > start and segment_start < end:
                first = int(np.searchsorted(starts, max(start, segment_start)))
                last = int(np.searchsorted(ends, min(end, segment_end), side="right"))
                bin_segments.append([first, last])
        return dict(start_s=start / self.sfreq, end_s=end / self.sfreq,
                    sfreq=self.sfreq, bin_samples=self.bin_samples,
                    bin_starts_s=(starts / self.sfreq).tolist(),
                    bin_ends_s=(ends / self.sfreq).tolist(), segments=bin_segments,
                    current=extrema(current), before=None if before is None else extrema(before))
