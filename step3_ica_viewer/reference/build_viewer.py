"""All sub-03 IC diagnostics for MNE component review; no EEG edits."""
from pathlib import Path
import base64
import hashlib
import importlib.util
import json

import mne
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
ORIGINAL = REPORTS / 'sub03_delta_z5_20261018'
DISTRIBUTION = REPORTS / 'sub03_delta_z5_distribution_20261018'
SAVED = REPORTS / 'sub03_step3_ica_assessment_20261016'
spec = importlib.util.spec_from_file_location('detector', ORIGINAL / 'analyze_delta_z5.py')
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


def main():
    mne.set_log_level('ERROR')
    input_provenance = json.loads((DISTRIBUTION / 'provenance.json').read_text())
    for entry in input_provenance['input_provenance'].values():
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    labels_path = Path(input_provenance['corrected_iclabel_table'])
    assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == input_provenance['table_sha256']
    counts = pd.read_csv(DISTRIBUTION / 'all119_counts.csv').set_index('component')
    eligible = counts.index[~counts.corrected_iclabel_auto_exclude].tolist()
    excluded = counts.index[counts.corrected_iclabel_auto_exclude].tolist()
    assert len(eligible) == 111 and excluded == [0, 8, 9, 12, 17, 71, 110, 118]
    epochs = mne.read_epochs(SAVED / 'sub-03-training-epo.fif', preload=True, verbose=False)
    ica = mne.preprocessing.read_ica(SAVED / 'sub-03-baseline-ica.fif', verbose=False)
    assert ica.noise_cov is None
    mixing_uV = ica.pre_whitener_ * ica.get_components() * 1e6
    voltage_channels = np.argmax(np.abs(mixing_uV), axis=0)
    assert (len(epochs), len(epochs.times), epochs.info['sfreq']) == (415, 500, 250)
    assert epochs.info['highpass'] == 1 and epochs.info['lowpass'] == 100
    assert np.all(np.diff(epochs.events[:, 0]) == 2000)
    sources = ica.get_sources(epochs).get_data().transpose(1, 0, 2).reshape(119, -1)
    payload = dict(sfreq=250, duration_s=830, lags_ms=[4, 20, 40, 80, 160],
                   n_epochs=len(epochs), epoch_duration_s=len(epochs.times) / epochs.info['sfreq'],
                   source_units='ICA activation units (AU)', source_display_scale=5.0,
                   threshold=5, merge_gap_ms=300,
                   review_id=hashlib.sha256((SAVED / 'sub-03-baseline-ica.fif').read_bytes()).hexdigest(),
                   components=[],
                   iclabel_excluded=excluded)
    all_events, summaries = [], []
    (HERE / 'data').mkdir(exist_ok=True)
    for component in counts.index:
        source_median, source_scale = detector.robust_parameters(sources[component])
        x = (sources[component] - source_median) / source_scale
        deltas = np.full((5, len(x)), np.nan)
        delta_medians, delta_scales = [], []
        for k, lag in enumerate(detector.LAGS):
            delta = x[lag:] - x[:-lag]
            median, mad = detector.robust_parameters(delta)
            z = (delta - median) / mad
            start = lag // 2
            deltas[k, start:start + len(z)] = z
            delta_medians.append(float(median))
            delta_scales.append(float(mad))
        absolute = np.nan_to_num(np.abs(deltas), nan=0.)
        score = absolute.max(axis=0)
        intervals = detector.runs(score >= 5)
        groups = detector.group_runs(intervals, 75)
        previous = counts.loc[component]
        assert len(intervals) == previous.threshold_runs
        assert len(groups) == previous.grouped_events
        level = median_filter(x, size=21, mode='reflect')
        contrast = level[30:] - level[:-30]
        median, mad = detector.robust_parameters(contrast)
        contrast_z = np.full(len(x), np.nan)
        contrast_z[15:-15] = (contrast - median) / mad
        events = []
        for event_id, (start, end, n_runs) in enumerate(groups, 1):
            peak = start + int(np.argmax(score[start:end]))
            winning = int(np.argmax(absolute[:, peak]))
            fast_z = float(absolute[:2, start:end].max())
            offset_z = float(contrast_z[peak]) if np.isfinite(contrast_z[peak]) else None
            pre = x[max(0, peak - 25):max(0, peak - 5)]
            post = x[min(len(x), peak + 5):min(len(x), peak + 26)]
            shape = ('fast edge + level change' if offset_z is not None and abs(offset_z) >= 5
                     else 'fast transient') if fast_z >= 5 else 'broader change'
            event = dict(component=component, event_id=event_id, start_s=start / 250,
                         end_s=end / 250, duration_s=(end - start) / 250,
                         peak_s=peak / 250, peak_score=float(score[peak]),
                         peak_lag_ms=detector.LAGS[winning] * 4, n_runs=n_runs,
                         fast_z=fast_z, median_contrast_z=offset_z,
                         pre_level=float(np.median(pre)) if len(pre) else None,
                         post_level=float(np.median(post)) if len(post) else None, shape=shape)
            events.append(event)
        summary = dict(n_runs=len(intervals), n_events=len(groups),
                       gap150_events=int(previous.gap150_events),
                       gap500_events=int(previous.gap500_events),
                       n_gt1s=int(previous.n_groups_over1s), max_duration_s=float(previous.max_group_duration_s))
        payload['components'].append(dict(id=component, name=f'ICA{component:03d}',
            source_median=float(source_median), source_scale=float(source_scale),
            voltage_channel=ica.ch_names[voltage_channels[component]],
            voltage_uV_per_source_unit=float(mixing_uV[voltage_channels[component], component]),
            delta_medians=delta_medians, delta_scales=delta_scales, events=events, summary=summary,
            iclabel_auto_exclude=bool(previous.corrected_iclabel_auto_exclude),
            iclabel_label=previous.corrected_iclabel_label, iclabel_score=float(previous.corrected_iclabel_score)))
        encoded = base64.b64encode(x.astype('<f4').tobytes()).decode()
        (HERE / 'data' / f'ICA{component:03d}.js').write_text(f'receiveICSignal({component},"{encoded}");\n')
        summaries.append(dict(component=component, name=f'ICA{component:03d}', **summary,
                              iclabel_auto_exclude=bool(previous.corrected_iclabel_auto_exclude),
                              iclabel_label=previous.corrected_iclabel_label,
                              iclabel_score=float(previous.corrected_iclabel_score)))
        all_events.extend(events)
    payload['method'] = json.loads((ORIGINAL / 'review_data.json').read_text())['method']
    table = pd.DataFrame(summaries)
    table.to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(all_events).to_csv(HERE / 'events.csv', index=False)
    (HERE / 'review_metadata.json').write_text(json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
    template = HERE / 'viewer_template.html'
    if template.exists():
        (HERE / 'index.html').write_text(template.read_text().replace('__PAYLOAD_JSON__',
            json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)))
    (HERE / 'provenance.json').write_text(json.dumps(dict(input_provenance=input_provenance,
        detector_sha256=hashlib.sha256((ORIGINAL / 'analyze_delta_z5.py').read_bytes()).hexdigest(),
        counts_match_previous_all119=True, shape_labels_do_not_gate=True,
        original_fit_epochs_only=True, production_edits=False), indent=2))

    print(f'Wrote{len(payload["components"])} component source files and{len(all_events)} events.')
    from add_shared_context import main as add_shared_context
    add_shared_context()


if __name__ == '__main__':
    main()
