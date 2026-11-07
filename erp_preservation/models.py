"""Known temporal signals and signed scalp weights for fixed-transform replay."""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class Term:
    """Sampled pulse, dimensionless input-reference scalp weights, condition amplitudes (µV)."""

    pulse: np.ndarray
    weights: np.ndarray
    amplitudes_uv: dict
    offset_samples: int = 0

    def __post_init__(self):
        self.pulse = np.asarray(self.pulse, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if isinstance(self.offset_samples, (bool, np.bool_)) or not isinstance(
            self.offset_samples, (int, np.integer)
        ):
            raise TypeError("Pulse offset must be an integer number of samples.")


@dataclass
class Model:
    """A sum of terms; focus weights select the reporting electrode and peak polarity."""

    name: str
    family: str
    terms: tuple
    focus: np.ndarray
    polarity: int
    peak_window: tuple
    mean_window: tuple
    info: dict = None

    def __post_init__(self):
        self.terms = tuple(self.terms)
        self.focus = np.asarray(self.focus, dtype=float)


LEGACY = (
    ("early_narrow", 0.25, 0.10),
    ("middle", 0.40, 0.18),
    ("late", 0.60, 0.25),
    ("late_broad", 0.80, 0.40),
    ("biphasic", 0.40, 0.18),
    ("asymmetric", 0.60, 0.25),
)
LEGACY_MAPS = (
    ("central_broad", (0, 0, 1), 0.75),
    ("posterior_broad", (0, -0.65, 0.76), 0.75),
    ("left_posterior", (-0.5, -0.45, 0.74), 0.55),
)
MAPS = (
    ("frontocentral", (0, 0.55, 0.84), 0.75),
    ("central", (0, 0, 1), 0.75),
    ("posterior", (0, -0.65, 0.76), 0.75),
)
COMPONENTS = (
    ("P1", 1, (40, 60, 80), (25, 40, 60)),
    ("N1", -1, (80, 110, 140), (40, 60, 90)),
    ("P2", 1, (140, 180, 220), (60, 90, 130)),
    ("N2", -1, (200, 250, 320), (70, 110, 180)),
    ("P3", 1, (300, 400, 600, 800), (120, 180, 350, 400)),
)
PROFILES = {
    "nominal": ((60, 40), (110, 60), (180, 90), (250, 110), (400, 180)),
    "delayed": ((80, 40), (140, 60), (220, 90), (320, 110), (600, 180)),
    "broad_overlap": ((60, 60), (110, 90), (180, 130), (320, 180), (600, 350)),
    "late_broad": ((80, 60), (140, 90), (220, 130), (320, 180), (800, 400)),
}
LAYOUTS = {
    "distributed": (0, 0, 1, 0, 2),
    "shared_central": (1, 1, 1, 1, 1),
    "alternative_frontocentral_P3": (1, 0, 0, 1, 0),
}
AMPLITUDES = {
    "balanced": ((1, -3, 2, -1, 1), (1, -3, 2, -2, 5)),
    "weak_P3": ((1, -3, 2, -1, 0.5), (1, -3, 2, -2, 2.5)),
    "strong_N2": ((1, -3, 2, -2, 1), (1, -3, 2, -5, 5)),
    "strong_early": ((2, -6, 4, -1, 1), (2, -6, 4, -2, 5)),
    "absent_P3": ((1, -3, 2, -1, 0), (1, -3, 2, -2, 0)),
    "identical_conditions": ((1, -3, 2, -2, 3), (1, -3, 2, -2, 3)),
}


def pulse(sfreq, peak_s, width_s, *, shape="component"):
    """Unit-peak pulse of length 1.4 s. Width is the Gaussian FWHM before tapering."""
    t = np.arange(round(1.4 * sfreq)) / sfreq
    sigma = width_s / (2 * np.sqrt(2 * np.log(2)))
    if shape == "asymmetric":
        sigma = np.where(t <= peak_s, sigma, sigma * 1.8)
    x = np.exp(-0.5 * ((t - peak_s) / sigma) ** 2)
    if shape == "biphasic":
        x -= 0.4 * np.exp(-0.5 * ((t - 0.18) / (0.1 / (2 * np.sqrt(2 * np.log(2))))) ** 2)
    ramp_s = 0.01 if shape == "component" else 0.03
    rise = 0.5 * (1 - np.cos(np.pi * np.minimum(t / ramp_s, 1.0)))
    fall = 0.5 * (1 - np.cos(np.pi * np.minimum((1.4 - t) / 0.1, 1.0)))
    if shape == "component":
        x *= rise
        x *= fall
    else:
        x *= rise * fall
    return x / np.max(x)


def auditory_models(sfreq, locations, *, conditions, model_reference, eeg_picks=None):
    """Yield the auditory default catalog on head coordinates in capture order.

    conditions gives (standard label, target label). model_reference is the
    channel index used to normalize the original model, including for Step 3;
    None uses the spatial weights as specified by the head-coordinate model.
    """
    labels = tuple(conditions)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("The auditory default requires two distinct condition labels.")
    xyz = np.asarray(locations, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Supply a sample rate and one XYZ position per channel.")
    picks = np.arange(len(xyz)) if eeg_picks is None else np.asarray(eeg_picks, dtype=int)
    if model_reference is not None and model_reference not in picks:
        raise ValueError("The model reference must be an EEG electrode.")
    lengths = np.linalg.norm(xyz[picks], axis=1)
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError(
            "Default maps need finite EEG positions; custom maps can be supplied instead."
        )
    directions = xyz[picks] / lengths[:, None]

    def make_map(spec):
        name, center, width = spec
        center = np.asarray(center, dtype=float)
        center /= np.linalg.norm(center)
        angle = np.arccos(np.clip(directions @ center, -1, 1))
        weights = np.zeros(len(xyz))
        weights[picks] = np.exp(-0.5 * (angle / width) ** 2)
        if model_reference is not None:
            weights[picks] -= weights[model_reference]
        scale = np.max(np.abs(weights))
        if scale == 0:
            raise ValueError(f"Map {name} has no spatial contrast on this montage.")
        return weights / scale

    maps = [make_map(spec) for spec in MAPS]
    for (shape, peak, width), spec in product(LEGACY, LEGACY_MAPS):
        weights = make_map(spec)
        wave = pulse(sfreq, peak, width, shape=shape)
        for label in labels:
            yield Model(
                f"{shape}/{spec[0]}/{label}",
                "legacy",
                (Term(wave, weights, {label: 5}),),
                weights,
                1,
                (0.0, 1.4),
                (0.3, 0.6),
                {"shape": shape, "map": spec[0]},
            )
    for component, sign, peaks, widths in COMPONENTS:
        for peak, width, idx, label in product(peaks, widths, range(len(maps)), labels):
            weights = maps[idx]
            term = Term(pulse(sfreq, peak / 1000, width / 1000), weights, {label: 5 * sign})
            yield Model(
                f"{component}/{peak}/{width}/{MAPS[idx][0]}/{label}",
                "component",
                (term,),
                weights,
                sign,
                (max(0, (peak - width) / 1000), min(1.4, (peak + width) / 1000)),
                (max(0, (peak - width / 2) / 1000), min(1.4, (peak + width / 2) / 1000)),
                {"component": component, "peak_ms": peak, "width_ms": width, "map": MAPS[idx][0]},
            )
    for profile, layout, amplitudes in product(PROFILES, LAYOUTS, AMPLITUDES):
        terms = []
        for k, ((peak, width), idx) in enumerate(zip(PROFILES[profile], LAYOUTS[layout])):
            amps = {label: AMPLITUDES[amplitudes][c][k] for c, label in enumerate(labels)}
            terms.append(Term(pulse(sfreq, peak / 1000, width / 1000), maps[idx], amps))
        yield Model(
            f"{profile}/{layout}/{amplitudes}",
            "composite",
            tuple(terms),
            terms[-1].weights,
            1,
            (0.25, 1.0),
            (0.3, 0.6),
            {"profile": profile, "layout": layout, "amplitudes": amplitudes},
        )
