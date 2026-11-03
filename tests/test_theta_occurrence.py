"""Check batch theta extraction against retained trial identities and caches."""

import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step7_theta_occurrence as step7


class DummyDecOsc:
    """Picklable fitted models with known selections and center frequencies."""

    calls = []
    profiles = {
        0: (0, [1, 5, 2], 1, [1, 3.5, 7.5, 9]),
        2: (2, [1, 5, 2], 2, [3.499, 7.501]),
        3: (0, [5, 2, 1], 0, [1, 8]),
        4: (1, [1, 5, 2], 1, [4, 6, 12]),
        5: (0, [1, 5, 2], 1, [5, 15]),
    }

    def __init__(self, y, Fs, **settings):
        self.y = y.copy()
        self.Fs = Fs
        self.settings = settings
        self.original_trial = int(round((y[0] - 2.05) / 10))
        initial, likelihoods, selected, frequencies = self.profiles[self.original_trial]
        self.knee_index = initial
        self.ll = np.array(likelihoods)
        self.frequencies = [np.array([4.5]) for _ in likelihoods]
        self.frequencies[selected] = np.array(frequencies)
        self.calls.append(self)

    def get_knee_osc(self):
        return SimpleNamespace(freq=self.frequencies[self.knee_index].copy())

    def plot_trace(self):
        return self.y


@pytest.fixture
def subject_files(tmp_path, monkeypatch):
    subject_dir = tmp_path / "sub-99"
    (subject_dir / "eeg").mkdir(parents=True)
    (subject_dir / "beh").mkdir()
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    base = 10 * np.arange(7)[:, None] + np.arange(301)[None, :] / 1000
    data = np.stack([base + offset for offset in (1, 3, 10000, 20000)], axis=1) * 1e-6
    info = mne.create_info(["E3", "E4", "E5", "E99"], 100, "eeg")
    info["bads"] = ["E5"]
    events = np.column_stack((1000 + 400 * np.arange(7),
                              np.zeros(7, dtype=int), np.ones(7, dtype=int)))
    epochs = mne.EpochsArray(data, info, events=events, tmin=-0.5,
                             baseline=None, verbose=False)
    with mne.use_log_level("ERROR"):
        epochs.drop([1], reason="AUTOREJECT")
        epochs.save(subject_dir / "eeg" / "sub-99_task-ColorPrecMemStudy_run-1_eeg-clean-epo.fif",
                    fmt="double", verbose=False)
    behavior_file = subject_dir / "beh" / "sub-99_task-ColorPrecMemStudy_labelled_beh.csv"
    pd.DataFrame({"response_type": ["HIT", "MISS", "HIT", "MISS", "HIT", "MISS", "OTHER"]}).to_csv(
        behavior_file, index=False,
    )
    monkeypatch.setattr(DummyDecOsc, "calls", [])
    monkeypatch.setattr(step7, "DecOsc", DummyDecOsc)
    return SimpleNamespace(subject_dir=subject_dir, output_dir=output_dir,
                           behavior_file=behavior_file, data=data)


def test_fit_preserves_step6_signal_selection_and_full_models(subject_files):
    files = subject_files
    summary = step7.fit_subject(files.subject_dir, files.output_dir)
    with (files.output_dir / "sub-99_dosc.pkl").open("rb") as stream:
        saved = pickle.load(stream)

    assert saved["frontal_channels"] == ["E3", "E4"]
    np.testing.assert_array_equal(saved["hit_trial_indices"], [0, 2, 4])
    np.testing.assert_array_equal(saved["miss_trial_indices"], [3, 5])
    models = saved["hit_oscillators"] + saved["miss_oscillators"]
    assert len(DummyDecOsc.calls) == len(models) == 5
    assert [model.original_trial for model in models] == [0, 2, 4, 3, 5]
    assert [model.knee_index for model in models] == [1, 2, 1, 0, 1]
    for model in models:
        assert isinstance(model, DummyDecOsc)
        assert model.Fs == 100
        assert model.settings == {"noise_start": None, "osc_range": 7, "iterate": True}
        expected = files.data[model.original_trial, :2, 50:].mean(axis=0) * 1e6
        assert model.y.shape == (251,)
        np.testing.assert_allclose(model.y, expected, rtol=1e-12)
        np.testing.assert_array_equal(model.plot_trace(), model.y)

    for name, expected in (("hit_theta_freq", [[3.5, 7.5], [], [4, 6]]),
                           ("miss_theta_freq", [[], [5]])):
        assert saved[name].dtype == object
        assert saved[name].shape == (len(expected),)
        for actual, frequencies in zip(saved[name], expected):
            np.testing.assert_array_equal(actual, frequencies)
    assert summary["n_hit"] == 3
    assert summary["n_hit_theta"] == 2
    assert summary["n_hit_without_theta"] == 1
    assert summary["hit_theta_pct"] == pytest.approx(200 / 3)
    assert summary["n_miss"] == 2
    assert summary["n_miss_theta"] == summary["n_miss_without_theta"] == 1
    assert summary["miss_theta_pct"] == 50
    assert summary["hit_minus_miss_pct_points"] == pytest.approx(50 / 3)
    assert saved["summary"] == summary


def test_cache_reuses_completed_fit_and_invalidates_changed_input(subject_files, monkeypatch):
    files = subject_files
    first = step7.fit_subject(files.subject_dir, files.output_dir)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("An unchanged cache must bypass data loading and fitting.")

    with monkeypatch.context() as cached:
        cached.setattr(step7, "DecOsc", unexpected_call)
        cached.setattr(step7.mne, "read_epochs", unexpected_call)
        assert step7.fit_subject(files.subject_dir, files.output_dir) == first

    behavior = pd.read_csv(files.behavior_file)
    behavior.loc[4, "response_type"] = "MISS"
    behavior.to_csv(files.behavior_file, index=False)
    updated = step7.fit_subject(files.subject_dir, files.output_dir)
    assert len(DummyDecOsc.calls) == 10
    assert updated["n_hit"] == 2
    assert updated["n_miss"] == 3
    assert updated["hit_theta_pct"] == 50
    assert updated["miss_theta_pct"] == pytest.approx(200 / 3)
    assert updated["hit_minus_miss_pct_points"] == pytest.approx(-50 / 3)


def test_analysis_skips_requested_subjects_before_loading(tmp_path, monkeypatch):
    for subject_id in ("sub-29", "sub-03", "sub-15", "sub-02", "sub-10"):
        (tmp_path / subject_id).mkdir()
    (tmp_path / "unrelated").mkdir()
    cache_dir = tmp_path / "cache"
    calls = []

    def fit_subject(subject_dir, actual_cache_dir):
        assert actual_cache_dir == cache_dir
        assert subject_dir.name not in {"sub-10", "sub-15", "sub-29"}
        calls.append(subject_dir.name)
        return {"subject_id": subject_dir.name, "hit_theta_pct": 0, "miss_theta_pct": 0}

    monkeypatch.setattr(step7, "fit_subject", fit_subject)
    summary = step7.analyze_subjects(tmp_path, cache_dir)
    assert calls == ["sub-02", "sub-03"]
    assert summary["subject_id"].tolist() == ["sub-02", "sub-03"]
    assert summary["hit_theta_pct"].tolist() == [0, 0]


def test_comparison_saves_both_visualizations_and_paired_result(tmp_path):
    summary = pd.DataFrame({
        "subject_id": ["sub-02", "sub-03", "sub-04"],
        "n_hit": [10, 10, 10], "n_hit_theta": [5, 6, 4],
        "n_hit_without_theta": [5, 4, 6], "hit_theta_pct": [50, 60, 40],
        "n_miss": [10, 10, 10], "n_miss_theta": [4, 4, 3],
        "n_miss_without_theta": [6, 6, 7], "miss_theta_pct": [40, 40, 30],
        "hit_minus_miss_pct_points": [10, 20, 10],
    })
    figure = step7.compare_subjects(summary, tmp_path)
    try:
        assert len(figure.axes) == 2
        paired, distribution = figure.axes
        assert paired.get_ylabel() == "Trials with theta oscillator (%)"
        assert distribution.get_xlabel() == "HIT - MISS (percentage points)"
        assert sum(patch.get_height() for patch in distribution.patches) == 3
        assert (tmp_path / "theta_distribution.png").stat().st_size > 0
        report = (tmp_path / "comparison.txt").read_text()
        p_value = float(re.search(r"p\s*=\s*([0-9.eE+-]+)", report).group(1))
        assert p_value == pytest.approx(0.05719095841793663, rel=1e-4)
        assert "Subjects: 3" in report
        assert "+13.33 percentage points" in report
    finally:
        plt.close(figure)
