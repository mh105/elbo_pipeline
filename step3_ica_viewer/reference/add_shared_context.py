"""Add cross-IC context to the exploratory report, without changing detections."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'sub03_delta_z5_distribution_20261018'


def main():
    payload = json.loads((HERE / 'review_metadata.json').read_text())
    sfreq = payload['sfreq']
    n_samples = int(round(payload['duration_s'] * sfreq))
    all_events = pd.read_csv(SOURCE / 'all119_grouped_events.csv')
    all_counts = pd.read_csv(SOURCE / 'all119_counts.csv')
    assert np.array_equal(np.sort(all_events.component.unique()), np.arange(119))
    assert [c['id'] for c in payload['components']] == all_counts.component.tolist()

    # Each row is one IC's union of grouped intervals, including bridged quiet gaps.
    # Intervals are half open, matching the detector's [start, end) sample bounds.
    presence = np.zeros((119, n_samples), dtype=bool)
    for component, frame in all_events.groupby('component'):
        starts = np.rint(frame.start_s.to_numpy() * sfreq).astype(int)
        ends = np.rint(frame.end_s.to_numpy() * sfreq).astype(int)
        assert np.all(starts < ends) and np.all(starts[1:] >= ends[:-1])
        for start, end in zip(starts, ends):
            presence[component, start:end] = True
    simultaneous = presence.sum(axis=0)
    peak_samples = np.rint(all_events.peak_s.to_numpy() * sfreq).astype(int)
    components = all_events.component.to_numpy()
    assert presence[components, peak_samples].all()
    all_events['context_count'] = simultaneous[peak_samples]
    all_events['context_fraction'] = all_events.context_count / 119
    all_events['shared'] = all_events.context_fraction > .25
    assert np.array_equal(all_events.shared, all_events.context_count >= 30)
    by_event = all_events.set_index(['component', 'event_id'])

    summaries, exported_events = [], []
    existing_summary = pd.read_csv(HERE / 'summary.csv').set_index('component')
    for component in payload['components']:
        ic = component['id']
        for event in component['events']:
            row = by_event.loc[(ic, event['event_id'])]
            assert np.allclose([event['start_s'], event['end_s'], event['peak_s']],
                               [row.start_s, row.end_s, row.peak_s], atol=1e-10, rtol=0)
            assert event['n_runs'] == row.threshold_runs
            peak = int(round(event['peak_s'] * sfreq))
            voters = np.flatnonzero(presence[:, peak]).tolist()
            event.update(shared=bool(row.shared), context_count=int(row.context_count),
                         context_fraction=float(row.context_fraction), context_components=voters)
            assert len(voters) == event['context_count'] and ic in voters
            exported_events.append(event)
        retained = [e for e in component['events'] if not e['shared']]
        summary = component['summary']
        summary.update(n_events_adjusted=len(retained),
            n_runs_adjusted=sum(e['n_runs'] for e in retained),
            max_duration_s_adjusted=max((e['duration_s'] for e in retained), default=0),
            n_gt1s_adjusted=sum(e['duration_s'] > 1 for e in retained),
            n_shared=summary['n_events'] - len(retained))
        assert summary['n_events'] == len(component['events'])
        assert summary['n_runs'] == sum(e['n_runs'] for e in component['events'])
        assert 0 <= summary['n_events_adjusted'] <= summary['n_events']
        assert 0 <= summary['n_runs_adjusted'] <= summary['n_runs']
        record = existing_summary.loc[ic].to_dict()
        record.update(component=ic, **summary)
        summaries.append(record)
    payload['context'] = dict(n_components=119, fraction_threshold=.25, min_components=30,
        rule='interval_overlap_at_event_peak', interval_bounds='start <= target peak < end',
        includes_iclabel_excluded=True, count_target_component=True,
        adjusted_runs_definition='Omit all threshold runs in shared grouped events.',
        rate_denominator_seconds=payload['duration_s'])
    payload['method'] = payload['method'].replace('Counts precede cross-component context.',
        'Raw counts precede cross-component context; parenthesized counts omit shared groups.')
    payload.pop('count_metric', None)
    payload.pop('count_cutoff', None)
    ica_path = HERE.parent / 'sub03_step3_ica_assessment_20261016' / 'sub-03-baseline-ica.fif'
    payload['review_id'] = hashlib.sha256(ica_path.read_bytes()).hexdigest()
    table = pd.DataFrame(summaries).drop(columns=['count_flag_gt100', 'run_review_flag'], errors='ignore')
    table.to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(exported_events).to_csv(HERE / 'events.csv', index=False)
    (HERE / 'review_metadata.json').write_text(json.dumps(payload, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False))
    template = (HERE / 'viewer_template.html').read_text()
    (HERE / 'index.html').write_text(template.replace('__PAYLOAD_JSON__',
        json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)))
    summary = dict(**payload['context'], n_displayed_components=len(table),
        raw_groups=int(table.n_events.sum()), adjusted_groups=int(table.n_events_adjusted.sum()),
        raw_runs=int(table.n_runs.sum()), adjusted_runs=int(table.n_runs_adjusted.sum()),
        per_component=table[['component','name','n_events','n_events_adjusted','n_runs','n_runs_adjusted']].to_dict('records'))
    (HERE / 'shared_context_summary.json').write_text(json.dumps(summary, indent=2))
    initial_rejected = table.loc[table.iclabel_auto_exclude, 'component'].tolist()
    (HERE / 'initial_review_summary.json').write_text(json.dumps(dict(
        review_id=payload['review_id'], initial_rejected=initial_rejected,
        initial_kept=len(table) - len(initial_rejected),
        viewer_mode='Manual component review across three shared-state tabs',
        manual_decisions_in='Browser local storage scoped to this saved ICA; Copy ica.exclude exports decisions',
        count_cutoff_used=False, saved_ica_modified=False), indent=2))

    # Independent sweep validates exact half-open membership at every target peak.
    sweep = np.zeros(n_samples + 1, dtype=int)
    np.add.at(sweep, np.rint(all_events.start_s.to_numpy() * sfreq).astype(int), 1)
    np.add.at(sweep, np.rint(all_events.end_s.to_numpy() * sfreq).astype(int), -1)
    assert np.array_equal(sweep[:-1].cumsum(), simultaneous)
    # Exact boundary checks: 30/119 is shared; 29/119 is below the threshold.
    assert by_event.loc[(1,176),'context_count'] == 30
    assert by_event.loc[(3,155),'context_count'] == 29
    eligible = ~table.iclabel_auto_exclude
    assert table.loc[eligible,'n_events_adjusted'].sum() == 6831
    assert table.loc[eligible,'n_runs_adjusted'].sum() == 28961
    assert summary['raw_groups'] == len(all_events)
    assert summary['adjusted_groups'] == int((~all_events.shared).sum())
    provenance = dict(source_events=str(SOURCE / 'all119_grouped_events.csv'),
        source_sha256=hashlib.sha256((SOURCE / 'all119_grouped_events.csv').read_bytes()).hexdigest(),
        context=payload['context'], unique_ic_presence_matches_independent_sweep=True,
        exact_threshold_boundaries_checked=True, all_original_counts_preserved=True,
        production_changes=False, detections_changed=False,
        review_policy='ICLabel baseline with shared manual overrides in the HTML viewer', count_cutoff_used=False)
    (HERE / 'shared_context_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(table.loc[table.component.isin([1,2,3,4,5,6,7,19]),
        ['name','n_events','n_events_adjusted','n_runs','n_runs_adjusted']].to_string(index=False))
    print(f'Groups: {summary["raw_groups"]} ({summary["adjusted_groups"]}); runs: {summary["raw_runs"]} ({summary["adjusted_runs"]}).')


if __name__ == '__main__':
    main()
