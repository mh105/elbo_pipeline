"""Exploratory waveform event review; leaves the EEG pipeline and exclusions unchanged."""
from pathlib import Path
import base64
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1]
SAVED = PIPE / 'reports/sub03_step3_ica_assessment_20261016'
COMPONENTS = [1, 2, 3, 4, 5, 6, 7, 19]
LAGS = [1, 5, 10, 20, 40]
SFREQ = 250
THRESHOLD = 5.0
MERGE_GAP_MS = 300


def robust_parameters(a):
    median = np.median(a)
    return median, 1.4826 * np.median(np.abs(a - median))


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)])


def group_runs(intervals, gap_samples):
    groups = []
    for start, end in intervals:
        if groups and start - groups[-1][1] <= gap_samples:
            groups[-1][1] = int(end)
            groups[-1][2] += 1
        else:
            groups.append([int(start), int(end), 1])
    return groups


def waveform_panel(ax, x, support, events, start_s, end_s):
    start = max(0, int(start_s * SFREQ))
    end = min(len(x), int(end_s * SFREQ) + 1)
    t = np.arange(start, end) / SFREQ
    y = x[start:end]
    ax.plot(t, y, color='#315978', linewidth=.85)
    ax.plot(t, np.where(support[start:end], y, np.nan), color='#c74535', linewidth=1.2)
    for event in events:
        if event['end_s'] > start_s and event['start_s'] < end_s:
            ax.axvspan(event['start_s'], event['end_s'], color='#e8a842', alpha=.15)
    ax.set_xlim(start / SFREQ, (end - 1) / SFREQ)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    mne.set_log_level('ERROR')
    ica_path = SAVED / 'sub-03-baseline-ica.fif'
    epochs_path = SAVED / 'sub-03-training-epo.fif'
    ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    assert len(epochs) == 415 and len(epochs.times) == 500
    assert epochs.info['sfreq'] == SFREQ and ica.n_components_ == 119
    assert epochs.info['highpass'] == 1 and epochs.info['lowpass'] == 100
    # Native event starts are contiguous: concatenation adds no differences across gaps.
    assert np.all(np.diff(epochs.events[:, 0]) == 2000)
    sources = ica.get_sources(epochs).get_data()[:, COMPONENTS, :]
    sources = sources.transpose(1, 0, 2).reshape(len(COMPONENTS), -1)
    payload = {'sfreq': SFREQ, 'lags_ms': [4 * n for n in LAGS],
               'threshold': THRESHOLD, 'merge_gap_ms': MERGE_GAP_MS,
               'duration_s': sources.shape[1] / SFREQ, 'components': []}
    all_events, summaries, per_lag = [], [], []
    diagnostic = []
    for component, source in zip(COMPONENTS, sources):
        center, scale = robust_parameters(source)
        x = (source - center) / scale
        deltas = np.full((len(LAGS), len(x)), np.nan)
        delta_medians, delta_scales = [], []
        for k, lag in enumerate(LAGS):
            delta = x[lag:] - x[:-lag]
            delta_median, delta_scale = robust_parameters(delta)
            assert delta_scale > 0
            z = (delta - delta_median) / delta_scale
            # Lags 1 and 5 have half-sample centers; floor aligns within 2 ms.
            start = lag // 2
            deltas[k, start:start + len(z)] = z
            delta_medians.append(float(delta_median))
            delta_scales.append(float(delta_scale))
            lag_support = np.zeros(len(x), dtype=bool)
            lag_support[start:start + len(z)] = np.abs(z) >= THRESHOLD
            per_lag.append({'component': component, 'lag_ms': lag * 4,
                            'threshold_runs': len(runs(lag_support)),
                            'grouped_300ms': len(group_runs(runs(lag_support), 75))})
        absolute = np.nan_to_num(np.abs(deltas), nan=0.)
        score = np.max(absolute, axis=0)
        support = score >= THRESHOLD
        intervals = runs(support)
        groups = group_runs(intervals, round(MERGE_GAP_MS * SFREQ / 1000))
        level = median_filter(x, size=21, mode='reflect')
        contrast = level[30:] - level[:-30]
        contrast_median, contrast_scale = robust_parameters(contrast)
        contrast_z = np.full(len(x), np.nan)
        contrast_z[15:-15] = (contrast - contrast_median) / contrast_scale
        events = []
        for event_id, (start, end, n_runs) in enumerate(groups, 1):
            peak = start + int(np.argmax(score[start:end]))
            winning = int(np.argmax(absolute[:, peak]))
            fast_z = float(np.max(absolute[:2, start:end]))
            offset_z = float(contrast_z[peak]) if np.isfinite(contrast_z[peak]) else None
            pre = x[max(0, peak - 25):max(0, peak - 5)]
            post = x[min(len(x), peak + 5):min(len(x), peak + 26)]
            shape = ('fast edge + level change' if offset_z is not None and abs(offset_z) >= 5
                     else 'fast transient') if fast_z >= 5 else 'broader change'
            event = dict(component=component, event_id=event_id,
                         start_s=start / SFREQ, end_s=end / SFREQ,
                         duration_s=(end - start) / SFREQ, peak_s=peak / SFREQ,
                         peak_score=float(score[peak]), peak_lag_ms=LAGS[winning] * 4,
                         n_runs=n_runs, fast_z=fast_z, median_contrast_z=offset_z,
                         pre_level=float(np.median(pre)) if len(pre) else None,
                         post_level=float(np.median(post)) if len(post) else None,
                         peak_to_peak=float(np.ptp(x[start:end])), shape=shape)
            events.append(event)
        assert sum(event['n_runs'] for event in events) == len(intervals)
        assert all(support[int(round(e['peak_s'] * SFREQ))] for e in events)
        assert all(a['end_s'] <= b['start_s'] for a, b in zip(events, events[1:]))
        durations = [e['duration_s'] for e in events]
        summary = dict(component=component, name=f'ICA{component:03d}',
                       n_runs=len(intervals), n_events=len(events),
                       gap150_events=len(group_runs(intervals, int(.150 * SFREQ))),
                       gap500_events=len(group_runs(intervals, round(.500 * SFREQ))),
                       n_gt1s=int(np.sum(np.array(durations) > 1)),
                       max_duration_s=max(durations),
                       threshold_time_s=float(np.sum(support) / SFREQ),
                       grouped_span_s=float(np.sum(durations)))
        summaries.append(summary)
        all_events.extend(events)
        payload['components'].append(dict(id=component, name=f'ICA{component:03d}',
            signal_b64=base64.b64encode(x.astype('<f4').tobytes()).decode(),
            delta_medians=delta_medians, delta_scales=delta_scales,
            events=events, summary=summary))
        diagnostic.append((x, support, score, deltas, events))
    payload['method'] = (
        'For each IC and lag (4, 20, 40, 80, 160 ms), subtract the temporal median '
        'of the lagged differences and divide by 1.4826 × MAD. A candidate occurs '
        'where any |delta Z| ≥ 5. Group contiguous threshold runs when the intervening '
        'quiet gap is ≤300 ms. Shape labels are descriptive: “fast” means the 4 or 20 ms '
        'delta reaches 5; “level change” means the before/after median contrast at the '
        'strongest edge reaches |Z| ≥5. The median windows are −100 to −20 ms and '
        '+20 to +100 ms. These labels do not remove any candidate. Counts precede '
        'cross-component context. Grouped spans include quiet gaps and can contain '
        'multiple physical artifacts. All samples come from the original 415 annotation-clean '
        '2-second fit epochs, filtered 1–100 Hz and sampled at 250 Hz.')
    pd.DataFrame(all_events).to_csv(HERE / 'events.csv', index=False)
    pd.DataFrame(summaries).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(per_lag).to_csv(HERE / 'per_lag_counts.csv', index=False)
    (HERE / 'review_data.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    template = HERE / 'viewer_template.html'
    if template.exists():
        (HERE / 'index.html').write_text(template.read_text().replace(
            '__PAYLOAD_JSON__', json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)))
    provenance = {'inputs': {p.name: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in [ica_path, epochs_path]},
                  'method': payload['method'], 'threshold': THRESHOLD, 'merge_gap_ms': MERGE_GAP_MS,
                  'source_shape': list(sources.shape), 'epochs_contiguous': True,
                  'delta_center_alignment': 'Nearest earlier native sample; 4/20ms lags shifted earlier by 2ms.',
                  'no_cross_component_context_applied': True, 'no_exclusions_changed': True}
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2))

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
    fig, axes = plt.subplots(8, 2, figsize=(21, 17), sharex=True)
    bin_width = 25
    for row, ((x, support, score, deltas, events), summary) in enumerate(zip(diagnostic, summaries)):
        n = len(x) // bin_width
        t = (np.arange(n) + .5) * bin_width / SFREQ
        blocks = x[:n * bin_width].reshape(n, bin_width)
        axes[row, 0].fill_between(t, blocks.min(axis=1), blocks.max(axis=1), color='#315978', alpha=.8)
        peaks = np.array([e['peak_s'] for e in events])
        axes[row, 0].plot(peaks, np.full(len(peaks), .025), '|', color='#c74535',
                          markersize=8, transform=axes[row, 0].get_xaxis_transform())
        axes[row, 0].set_ylabel(f"{summary['name']}\nrobust amplitude")
        axes[row, 0].set_title(f"{summary['n_events']} grouped candidates | {summary['n_runs']} threshold runs", loc='left')
        sb = score[:n * bin_width].reshape(n, bin_width).max(axis=1)
        axes[row, 1].plot(t, sb, lw=.6, color='#765a93')
        axes[row, 1].axhline(5, color='#c74535', ls='--', lw=.8)
        axes[row, 1].set_yscale('log')
        axes[row, 1].set_ylim(.4, max(15, score.max() * 1.2))
        axes[row, 1].set_ylabel('max |delta Z|')
        axes[row, 1].set_title('Union score across five lags; dashed line = 5', loc='left')
        for ax in axes[row]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_xlim(0, len(x) / SFREQ)
    for ax in axes[-1]:
        ax.set_xlabel('Recording time (s)')
    fig.suptitle('Sub-03: delta Z >=5, all five lags, 300ms grouping; ICA019 is the manual keep comparison', fontsize=15)
    fig.text(.5, .005, 'Exact fit data: 415 x2s,1-100Hz,250Hz. Red ticks=grouped candidates. No cross-component discounting or IC rejection.', ha='center')
    fig.tight_layout(rect=(0, .025, 1, .975))
    fig.savefig(HERE / 'overview.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(7, 3, figsize=(22, 17))
    for row, (x, support, score, deltas, events) in enumerate(diagnostic[:7]):
        ranked = sorted(events, key=lambda e: e['peak_score'])
        selected = [ranked[-1], ranked[len(ranked) // 2], max(events, key=lambda e: e['duration_s'])]
        for col, (event, category) in enumerate(zip(selected, ['Strongest', 'Median score', 'Longest group'])):
            start_s = max(0, min(event['start_s'] - .35, event['peak_s'] - 1))
            end_s = min(len(x) / SFREQ, max(event['end_s'] + .35, event['peak_s'] + 1))
            waveform_panel(axes[row, col], x, support, events, start_s, end_s)
            axes[row, col].set_title(f"ICA{COMPONENTS[row]:03d} {category}: #{event['event_id']} | Z={event['peak_score']:.1f}\n{event['shape']}, {event['n_runs']} runs, span{event['duration_s']:.3f}s")
            if col == 0:
                axes[row, col].set_ylabel('Robust amplitude')
    for ax in axes[-1]:
        ax.set_xlabel('Recording time (s)')
    fig.suptitle('Detected candidate examples: red waveform=threshold support; pale amber=grouped interval', fontsize=15)
    fig.text(.5, .005, 'Shape labels are descriptive; all Z>=5 candidates remain. Long intervals expose chaining by the300ms grouping rule.', ha='center')
    fig.tight_layout(rect=(0, .025, 1, .975))
    fig.savefig(HERE / 'event_examples.png', dpi=150)
    plt.close(fig)

    colors = ['#4677a0', '#4f9d8b', '#aa8142', '#8d6aa5', '#bd655e']
    for component, (x, support, score, deltas, events) in zip(COMPONENTS, diagnostic):
        ranked = sorted(events, key=lambda e: e['peak_score'])
        selected = [ranked[-1], ranked[len(ranked) // 2], ranked[0], max(events, key=lambda e: e['duration_s'])]
        fig, axes = plt.subplots(4, 2, figsize=(17, 11))
        for row, (event, category) in enumerate(zip(selected, ['Strongest', 'Median score', 'Lowest score', 'Longest group'])):
            a = max(0, min(event['start_s'] - .3, event['peak_s'] - 1))
            b = min(len(x) / SFREQ, max(event['end_s'] + .3, event['peak_s'] + 1))
            waveform_panel(axes[row, 0], x, support, events, a, b)
            axes[row, 0].set_ylabel('Robust amplitude')
            contrast_text = f"{event['median_contrast_z']:.1f}" if event['median_contrast_z'] is not None else 'n/a'
            axes[row, 0].set_title(f"{category}: #{event['event_id']}, {event['shape']}\nmax Z={event['peak_score']:.1f}, median-contrast Z={contrast_text}")
            start, end = int(a * SFREQ), min(len(x), int(b * SFREQ) + 1)
            tt = np.arange(start, end) / SFREQ
            for k, lag in enumerate(LAGS):
                axes[row, 1].plot(tt, deltas[k, start:end], color=colors[k], lw=.75, label=f'{lag * 4}ms')
            axes[row, 1].axhline(5, color='#c74535', ls='--', lw=.7)
            axes[row, 1].axhline(-5, color='#c74535', ls='--', lw=.7)
            axes[row, 1].axvspan(event['start_s'], event['end_s'], color='#e8a842', alpha=.15)
            axes[row, 1].set_xlim(a, b)
            axes[row, 1].set_ylabel('Signed delta robust Z')
            axes[row, 1].spines[['top', 'right']].set_visible(False)
            axes[row, 1].set_title(f"{event['n_runs']} threshold runs in {event['duration_s']:.3f}s grouped span")
            axes[row, 1].tick_params(labelsize=8)
        axes[0, 1].legend(ncol=5, fontsize=8)
        axes[-1, 0].set_xlabel('Recording time (s)')
        axes[-1, 1].set_xlabel('Recording time (s)')
        fig.suptitle(f'ICA{component:03d}: Z5 event inspection;300ms grouping', fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(HERE / f'ICA{component:03d}_detail.png', dpi=150)
        plt.close(fig)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
