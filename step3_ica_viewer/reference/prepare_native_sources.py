"""Recover native ICA activation scale without duplicating stored source arrays."""
from pathlib import Path
import base64
import hashlib
import json
import re

import mne
import numpy as np

HERE = Path(__file__).resolve().parent
SAVED = HERE.parent / 'sub03_step3_ica_assessment_20261016'
DISTRIBUTION = HERE.parent / 'sub03_delta_z5_distribution_20261018'


def main():
    mne.set_log_level('ERROR')
    inputs = json.loads((DISTRIBUTION / 'provenance.json').read_text())['input_provenance']
    for entry in inputs.values():
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    epochs = mne.read_epochs(SAVED / 'sub-03-training-epo.fif', preload=True, verbose=False)
    ica = mne.preprocessing.read_ica(SAVED / 'sub-03-baseline-ica.fif', verbose=False)
    assert ica.noise_cov is None
    mixing_uV = ica.pre_whitener_ * ica.get_components() * 1e6
    voltage_channels = np.argmax(np.abs(mixing_uV), axis=0)
    assert (len(epochs), len(epochs.times), epochs.info['sfreq']) == (415, 500, 250)
    assert (epochs.info['highpass'], epochs.info['lowpass']) == (1, 100)
    assert np.all(np.diff(epochs.events[:, 0]) == 2000)
    assert ica.n_components_ == 119 and epochs.ch_names == ica.ch_names
    sources = ica.get_sources(epochs).get_data().transpose(1, 0, 2).reshape(119, -1)
    transformed = ica._transform_epochs(epochs, concatenate=True)
    assert np.array_equal(sources, transformed)
    del transformed
    payload = dict(sfreq=250, duration_s=830, n_epochs=415, epoch_duration_s=2.0,
                   source_units='ICA activation units (AU)', source_display_scale=5.0,
                   components=[])
    validation = []
    for component, source in enumerate(sources):
        center = float(np.median(source))
        scale = float(1.4826 * np.median(np.abs(source - center)))
        assert scale > 0
        source_file = HERE / 'data' / f'ICA{component:03d}.js'
        match = re.fullmatch(r'receiveICSignal\((\d+),"([A-Za-z0-9+/=]+)"\);\s*', source_file.read_text())
        assert match and int(match[1]) == component
        normalized = np.frombuffer(base64.b64decode(match[2]), dtype='<f4')
        assert normalized.size == source.size == 207500
        expected = ((source - center) / scale).astype('<f4')
        assert np.array_equal(normalized, expected)
        recovered = normalized.astype(float) * scale + center
        error = recovered - source
        relative_error = float(np.max(np.abs(error)) / np.max(np.abs(source)))
        assert relative_error < np.finfo(np.float32).eps
        correlation = float(np.corrcoef(recovered, source)[0, 1])
        assert correlation > 1 - 1e-12
        payload['components'].append(dict(id=component, source_median=center, source_scale=scale,
            voltage_channel=ica.ch_names[voltage_channels[component]],
            voltage_uV_per_source_unit=float(mixing_uV[voltage_channels[component], component])))
        validation.append(dict(component=component, n_samples=source.size,
                               max_absolute_error=float(np.max(np.abs(error))),
                               max_error_relative_to_peak=relative_error,
                               correlation=correlation,
                               normalized_float32_exact=True))
    (HERE / 'native_source_parameters.json').write_text(json.dumps(payload, indent=2))
    provenance = dict(input_provenance=inputs, mne_version=mne.__version__,
        channel_order=epochs.ch_names, n_components=119, n_epochs=415, n_samples_per_epoch=500,
        sfreq=250, highpass=1, lowpass=100, event_start_difference_native_samples=2000,
        ica_get_sources_equals_plot_sources_transform=True,
        source_units='ICA activation units (AU); not sensor voltages in microvolts',
        reconstruction='native_source = normalized_float32 * source_scale + source_median',
        voltage_scale='Signed contribution at the electrode with the largest absolute physical mixing weight; relative to the source median.',
        voltage_conversion='uV = robust_value * source_scale * voltage_uV_per_source_unit',
        display='Subtract the displayed-window mean and divide every IC by 2 * source_display_scale',
        mne_plot_sources_scaling=5.0, per_component=validation,
        maximum_absolute_roundtrip_error=max(v['max_absolute_error'] for v in validation),
        maximum_relative_roundtrip_error=max(v['max_error_relative_to_peak'] for v in validation),
        minimum_correlation=min(v['correlation'] for v in validation),
        decomposition_changed=False, detections_changed=False)
    (HERE / 'native_source_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({key: value for key, value in provenance.items()
                      if key not in ('per_component', 'channel_order', 'input_provenance')}, indent=2))


if __name__ == '__main__':
    main()
