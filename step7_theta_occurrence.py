"""Compare HIT/MISS theta detection rates, excluding sub-10, sub-15, sub-29.

Use the Step 6 fit settings, 3.5-7.5 Hz inclusive, and the index-0 adjustment.
Reuse valid per-subject model caches; fit uncached subjects from cleaned epochs.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from somata.oscillator_search import DecomposedOscillatorModel as DecOsc


EXCLUDED_SUBJECTS = ("sub-10", "sub-15", "sub-29")


def fit_subject(subject_dir, cache_dir):
    """Return subject counts; cache full models, theta arrays, and trial indices."""
    subject_id = subject_dir.name
    epochs_file = (subject_dir / "eeg"
                   / f"{subject_id}_task-ColorPrecMemStudy_run-1_eeg-clean-epo.fif")
    behavior_file = (subject_dir / "beh"
                     / f"{subject_id}_task-ColorPrecMemStudy_labelled_beh.csv")
    frontal_cluster = [
        "E3", "E4", "E5", "E10", "E11", "E12", "E16", "E18", "E19", "E23",
    ]
    fit_settings = dict(noise_start=None, osc_range=7, iterate=True)
    signature = {
        "inputs": [(p.stat().st_size, p.stat().st_mtime_ns)
                   for p in (epochs_file, behavior_file)],
        "frontal_cluster": frontal_cluster,
        "fit_settings": fit_settings,
        "fit_start": 0.0,
        "theta_range": (3.5, 7.5),
        "zero_knee_fallback": "max_ll",
    }
    cache_file = cache_dir / f"{subject_id}_dosc.pkl"
    if cache_file.exists():
        with cache_file.open("rb") as stream:
            result = pickle.load(stream)
        if result["signature"] == signature:
            return result["summary"]

    behavior = pd.read_csv(behavior_file, keep_default_na=False)
    epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
    if len(behavior) != len(epochs.drop_log):
        raise ValueError(f"{subject_id}: behavior rows do not match epoch trials.")
    available_channels = set(epochs.ch_names) - set(epochs.info["bads"])
    valid_frontal_cluster = [c for c in frontal_cluster if c in available_channels]
    frontal_epoch_signals = epochs.get_data(
        picks=valid_frontal_cluster, units="uV"
    ).mean(axis=1)
    response_types = behavior.iloc[epochs.selection]["response_type"].to_numpy()
    if not np.any(response_types == "HIT") or not np.any(response_types == "MISS"):
        raise ValueError(f"{subject_id}: at least one HIT and MISS trial is required.")
    Fs = epochs.info["sfreq"]
    stimulus_onset_idx = epochs.time_as_index(0, use_rounding=True)[0]
    result = {"signature": signature, "frontal_channels": valid_frontal_cluster}
    summary = {"subject_id": subject_id}

    for response_type, prefix in (("HIT", "hit"), ("MISS", "miss")):
        condition_mask = response_types == response_type
        signals = frontal_epoch_signals[condition_mask]
        oscillators = []
        theta_freq = np.empty(len(signals), dtype=object)
        for trial_idx, signal in enumerate(signals):
            y = signal[stimulus_onset_idx:]
            do1 = DecOsc(y, Fs, **fit_settings)
            if do1.knee_index == 0:
                do1.knee_index = np.argmax(do1.ll)
            oscillators.append(do1)
            freqs = do1.get_knee_osc().freq
            theta_freq[trial_idx] = freqs[(freqs >= 3.5) & (freqs <= 7.5)]
        result[f"{prefix}_oscillators"] = oscillators
        result[f"{prefix}_theta_freq"] = theta_freq
        result[f"{prefix}_trial_indices"] = epochs.selection[condition_mask]
        n_with_theta = sum(freqs.size > 0 for freqs in theta_freq)
        summary[f"n_{prefix}"] = len(signals)
        summary[f"n_{prefix}_theta"] = n_with_theta
        summary[f"n_{prefix}_without_theta"] = len(signals) - n_with_theta
        summary[f"{prefix}_theta_pct"] = 100 * n_with_theta / len(signals)

    summary["hit_minus_miss_pct_points"] = (
        summary["hit_theta_pct"] - summary["miss_theta_pct"]
    )
    result["summary"] = summary
    temporary_file = cache_file.with_suffix(".tmp")
    with temporary_file.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_file.replace(cache_file)
    return summary


def analyze_subjects(project_dir, cache_dir):
    """Loop over included subjects and return one HIT/MISS rate pair per subject."""
    subject_dirs = sorted(
        p for p in project_dir.glob("sub-*")
        if p.is_dir() and p.name not in EXCLUDED_SUBJECTS
    )
    summaries = []
    for i, subject_dir in enumerate(subject_dirs, start=1):
        row = fit_subject(subject_dir, cache_dir)
        summaries.append(row)
        print(f"[{i}/{len(subject_dirs)}] {row['subject_id']}: "
              f"HIT {row['hit_theta_pct']:.1f}%; MISS {row['miss_theta_pct']:.1f}%",
              flush=True)
    return pd.DataFrame(summaries).sort_values("subject_id")


def compare_subjects(summary, output_dir):
    """Save the paired distribution figure and an exploratory subject-level test."""
    hit_pct = summary["hit_theta_pct"].to_numpy()
    miss_pct = summary["miss_theta_pct"].to_numpy()
    differences = hit_pct - miss_pct
    test = ttest_rel(hit_pct, miss_pct)
    interval = test.confidence_interval()
    report = (
        "Excluded subjects: sub-10, sub-15, sub-29.\n"
        "Selection: if knee_index is 0, use argmax(ll).\n"
        f"Subjects: {len(summary)}\n"
        f"Mean subject theta percentage: HIT {hit_pct.mean():.2f}%; "
        f"MISS {miss_pct.mean():.2f}%\n"
        f"Mean paired HIT - MISS: {differences.mean():+.2f} percentage points "
        f"(95% CI {interval.low:+.2f} to {interval.high:+.2f})\n"
        f"Exploratory paired t-test (two-sided): t({test.df:.0f}) = "
        f"{test.statistic:.3f}, p = {test.pvalue:.4g}\n"
        f"Subjects with HIT higher / equal / lower: "
        f"{np.count_nonzero(differences > 0)} / "
        f"{np.count_nonzero(differences == 0)} / "
        f"{np.count_nonzero(differences < 0)}\n"
    )
    for response_type, prefix in (("HIT", "hit"), ("MISS", "miss")):
        n_with = summary[f"n_{prefix}_theta"].sum()
        n_without = summary[f"n_{prefix}_without_theta"].sum()
        report += (
            f"Pooled {response_type} trials (descriptive): "
            f"{n_with / (n_with + n_without):.1%} "
            f"({n_with}/{n_without} with/without)\n"
        )
    report += "Theta presence: at least one selected-model frequency in 3.5-7.5 Hz.\n"
    (output_dir / "comparison.txt").write_text(report)
    print(report, flush=True)

    fig, (ax_paired, ax_difference) = plt.subplots(
        1, 2, figsize=(11, 4.5), layout="constrained"
    )
    for values, cloud_pos, box_pos, side, color in (
        (hit_pct, -0.15, -0.05, "low", "C0"),
        (miss_pct, 1.15, 1.05, "high", "C1"),
    ):
        cloud = ax_paired.violinplot(
            values, positions=[cloud_pos], widths=0.55, side=side,
            showextrema=False,
        )["bodies"][0]
        cloud.set_facecolor(color)
        cloud.set_edgecolor(color)
        cloud.set_alpha(0.3)
        ax_paired.boxplot(
            values, positions=[box_pos], widths=0.09, showfliers=False,
            patch_artist=True, boxprops={"facecolor": color, "alpha": 0.4},
            medianprops={"color": "black"},
        )
    jitter = np.random.default_rng(0).uniform(-0.04, 0.04, len(hit_pct))
    hit_x, miss_x = 0.10 + jitter, 0.90 + jitter
    for x_hit, x_miss, hit, miss in zip(hit_x, miss_x, hit_pct, miss_pct):
        ax_paired.plot([x_hit, x_miss], [hit, miss], color="0.7", alpha=0.5,
                       linewidth=0.7)
    ax_paired.scatter(hit_x, hit_pct, color="C0", s=28, zorder=3)
    ax_paired.scatter(miss_x, miss_pct, color="C1", s=28, zorder=3)
    ax_paired.set(xticks=[0, 1], xticklabels=["HIT", "MISS"],
                  xlim=(-0.5, 1.5), ylim=(-2, 102),
                  ylabel="Trials with theta oscillator (%)",
                  title=f"Paired subject percentages (n={len(summary)})")
    ax_paired.text(0.5, 0.95, f"Paired t-test: p = {test.pvalue:.4f}",
                   transform=ax_paired.transAxes, ha="center", va="top")
    bin_width = 2.5
    bin_min = bin_width * np.floor(differences.min() / bin_width)
    bin_max = bin_width * (np.floor(differences.max() / bin_width) + 1)
    bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)
    ax_difference.hist(differences, bins=bin_edges, color="0.65", edgecolor="white")
    ax_difference.axvline(0, color="black", linestyle=":", label="Equal percentages")
    ax_difference.axvline(differences.mean(), color="C3",
                          label=f"Mean: {differences.mean():+.2f} pp")
    ax_difference.set(xlabel="HIT - MISS (percentage points)", ylabel="Subjects",
                      title="Within-subject differences")
    ax_difference.legend()
    fig.suptitle("Theta occurrence: 3.5-7.5 Hz, stimulus 0-2.5 s\n"
                 "Excluding sub-10, sub-15, sub-29; index-0 adjustment")
    fig.savefig(output_dir / "theta_distribution.png", dpi=180)
    return fig


if __name__ == "__main__":
    # %% Loop through included subjects and compare theta detection rates
    pipeline_dir = Path(__file__).resolve().parent
    project_dir = Path(__file__).resolve().parents[3]
    cache_dir = pipeline_dir / "reports" / "step7_theta_analysis"
    output_dir = pipeline_dir / "reports" / "step7_theta_occurrence"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_summary = analyze_subjects(project_dir, cache_dir)
    subject_summary.to_csv(output_dir / "subject_summary.csv", index=False)

    # %% Paired HIT/MISS percentages and distribution of within-subject differences
    fig = compare_subjects(subject_summary, output_dir)
    plt.show()
