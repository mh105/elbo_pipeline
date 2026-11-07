"""Step 2 recovery reporting, optional diagnostics, and restart checkpoints."""
import json
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np


class NoScorableSupport(ValueError):
    """A detector had no eligible samples or complete windows."""


class InsufficientASRCalibration(ValueError):
    """ASR cannot fit its calibration model. No correction has been applied."""

    def __init__(self, phase, retained, candidate, sfreq, minimum):
        self.details = dict(status="skipped", reason="insufficient_calibration",
                            phase=phase, retained_samples=int(retained),
                            candidate_samples=int(candidate), minimum_samples=int(minimum),
                            retained_seconds=retained / sfreq, minimum_seconds=minimum / sfreq)
        super().__init__(f"ASR requires at least 60 seconds of retained calibration data; "
                         f"only {retained / sfreq:.3f} seconds remain.")


_plot_warnings = ContextVar("step2_plot_warnings", default=None)


@contextmanager
def collect_plot_warnings(records):
    token = _plot_warnings.set(records)
    try:
        yield
    finally:
        _plot_warnings.reset(token)


def warn_missing_plot(label, reason):
    records = _plot_warnings.get()
    if records is not None:
        records.append(dict(plot=label, reason=reason))
    warnings.warn(reason, stacklevel=2)


def diagnostic(label, plot, *args, **kwargs):
    """Only a figure's failure is optional. Keep detector failures visible."""
    records = _plot_warnings.get()
    if records is None:
        return plot(*args, **kwargs)
    try:
        return plot(*args, **kwargs)
    except Exception as exc:
        records.append(dict(plot=label, error=type(exc).__name__, reason=str(exc)))
        warnings.warn(f"Could not produce {label}: {exc}", RuntimeWarning, stacklevel=2)


def _json_numeric(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} cannot be written in the Step 2 report")


class Checkpoints:
    """Publish only completed FIFs; an optional new directory is owned by one run."""

    def __init__(self, directory, report):
        self.directory = Path(directory) if directory is not None else None
        self.report = report
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)

    def write_report(self):
        if self.directory is not None:
            tmp = self.directory / ".report.tmp"
            tmp.write_text(json.dumps(self.report, default=_json_numeric, allow_nan=False,
                                      indent=2) + "\n", encoding="utf-8")
            tmp.replace(self.directory / "step2_report.json")

    def save(self, stage, raw):
        if self.directory is not None:
            # FIF stores calibration coefficients in single precision. The loaded
            # samples are already in physical units; use unity storage gains so
            # a round-trip does not perturb samples or change the working Raw.
            with raw.copy() as physical:
                for ch in physical.info["chs"]:
                    ch["cal"] = ch["range"] = 1.
                physical._cals = np.ones(len(physical.ch_names))
                physical.save(self.directory / f"{stage}_raw.fif", fmt="double", verbose=False)
            self.report["checkpoints"][stage] = f"{stage}_raw.fif"
        self.report["last_completed_checkpoint"] = stage
        self.write_report()
