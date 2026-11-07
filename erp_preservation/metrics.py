"""Measure paired event × electrode × sample arrays, in volts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Windows:
    epoch: tuple = (-0.2, 1.4)
    baseline: tuple = (-0.2, 0.0)
    wide_mean: tuple = (0.2, 0.9)
    weak_uv: float = 0.1

    def bounds(self, sfreq):
        if not np.isfinite(self.weak_uv) or self.weak_uv <= 0:
            raise ValueError("weak_uv must be finite and positive.")
        if len(self.epoch) != 2 or not np.isfinite(self.epoch).all():
            raise ValueError("Epoch bounds must be finite.")
        lo, hi = (round(t * sfreq) for t in self.epoch)
        if not lo <= 0 < hi:
            raise ValueError("Epoch must include stimulus onset and poststimulus samples.")
        return lo, hi

    def mask(self, sfreq, interval, *, require_samples=True):
        lo, hi = self.bounds(sfreq)
        # Preserve the original half-open window convention at float boundaries.
        if len(interval) != 2 or not np.isfinite(interval).all() or interval[1] <= interval[0]:
            raise ValueError("Measurement window must have increasing finite bounds.")
        t = np.arange(hi - lo) / sfreq + lo / sfreq
        selected = (t >= interval[0]) & (t < interval[1])
        if require_samples and not selected.any():
            raise ValueError(f"Window {interval} contains no epoch samples.")
        return selected

    def coverage(self, sfreq, interval):
        """Requested times and the half-open interval of sample bins actually measured."""
        lo, hi = self.bounds(sfreq)
        idx = np.flatnonzero(self.mask(sfreq, interval, require_samples=False))
        if not idx.size:
            return dict(requested_s=list(interval), used_s=None, samples=0, status="unavailable")
        complete = lo / sfreq <= interval[0] and interval[1] <= hi / sfreq
        return dict(requested_s=list(interval), used_s=[(lo + int(idx[0])) / sfreq,
                    (lo + int(idx[-1]) + 1) / sfreq], samples=int(idx.size),
                    status="complete" if complete else "partial")



def measure(before, after, sfreq, rep, sign, peak, mean, windows, corrected):
    b, a = before.copy(), after.copy()
    if corrected:
        baseline = windows.mask(sfreq, windows.baseline)
        # Use the contiguous slice also used by the original assay.
        idx = np.flatnonzero(baseline)
        sl = slice(idx[0], idx[-1] + 1)
        b -= b[:, :, sl].mean(2, keepdims=True)
        a -= a[:, :, sl].mean(2, keepdims=True)
    start = -windows.bounds(sfreq)[0]
    x, y = b[:, :, start:], a[:, :, start:]
    axis = (1, 2)
    power = (x * x).sum(axis)
    reference = np.sqrt((x * x).mean(axis))
    delta = y - x
    error = np.sqrt((delta * delta).mean(axis))
    weak = reference < windows.weak_uv * 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = (x * y).sum(axis) / power
        relative = np.sqrt((delta * delta).sum(axis) / power)
    xf, yf = x.reshape(len(x), -1), y.reshape(len(y), -1)
    cx = xf - xf.mean(1, keepdims=True)
    cy = yf - yf.mean(1, keepdims=True)
    norm = np.sqrt((cx * cx).sum(1) * (cy * cy).sum(1))
    no_shape = np.linalg.norm(cy, axis=1) < 1e-20
    corr = np.divide((cx * cy).sum(1), norm, out=np.zeros_like(norm), where=norm > 0)
    old_sign = np.sign(x[:, rep][np.arange(len(x)), np.argmax(abs(x[:, rep]), axis=1)])
    xb, ya = x[:, rep] * old_sign[:, None], y[:, rep] * old_sign[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        old_ratio = ya.max(1) / xb.max(1)
    mask = windows.mask(sfreq, peak, require_samples=False)
    bt, at = b[:, rep, mask] * sign, a[:, rep, mask] * sign
    if mask.any():
        bp, ap = bt.max(1) * 1e6, at.max(1) * 1e6
        shift = (at.argmax(1) - bt.argmax(1)) * 1000 / sfreq
    else:
        bp, ap, shift = (np.full(len(b), np.nan) for _ in range(3))
    low = ~np.isfinite(bp) | (bp < windows.weak_uv)
    stats = {
        "reference_rms_uv": reference * 1e6,
        "rms_error_uv": error * 1e6,
        "relative_l2_error": np.divide(
            error, reference, out=np.full_like(error, np.nan), where=~weak
        ),
        "low_reference_rms": weak,
        "peak_before_uv": bp,
        "peak_after_uv": ap,
        "peak_change_uv": ap - bp,
        "peak_attenuation_percent": np.divide(
            100 * (bp - ap), bp, out=np.full_like(bp, np.nan), where=~low
        ),
        "low_peak_reference": low,
        "peak_shift_ms": shift,
        "peak_interpretation_flag": low | (ap < windows.weak_uv),
        "gain": np.where(weak, np.nan, gain),
        "correlation": np.where(weak | no_shape, np.nan, corr),
        "relative_error": np.where(weak, np.nan, relative),
        "output_shape_undefined": no_shape,
        "legacy_gain": gain,
        "legacy_relative_error": relative,
        "legacy_correlation": corr,
        "legacy_peak_amplitude_ratio": old_ratio,
        "legacy_peak_interpretation_flag": (xb.max(1) * 1e6 < windows.weak_uv) | (ya.max(1) * 1e6 < windows.weak_uv),
        "legacy_peak_shift_ms": (ya.argmax(1) - xb.argmax(1)) * 1000 / sfreq,
    }
    for label, interval in (("mean", mean), ("wide_mean", windows.wide_mean)):
        mask = windows.mask(sfreq, interval, require_samples=False)
        if mask.any():
            v, w = b[:, rep, mask].mean(1) * 1e6, a[:, rep, mask].mean(1) * 1e6
        else:
            v, w = (np.full(len(b), np.nan) for _ in range(2))
        stats.update({label + "_before_uv": v, label + "_after_uv": w, label + "_change_uv": w - v})
    return stats, b[:, rep].copy(), a[:, rep].copy()
