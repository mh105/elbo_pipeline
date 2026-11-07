# Synthetic ERP preservation

How much would a particular, already fitted ELBO correction change a known event-locked response?
The assessment places artificial signals at the recording's event samples and replays the frozen corrections
on those signals alone. It uses no recorded EEG voltages, makes no channel or component decisions, and
performs no calibration or fitting.

## Use the auditory defaults

```python
import numpy as np
from erp_preservation import Assessment, write_assessment
from step2_capture import Step2Capture
from step3_capture import Step3Capture

step2 = Step2Capture.load("step2-capture.npz")
ica = Step3Capture.load("ica_transform.npz")
# event_samples is a mapping such as {"standard": [...], "target": [...]}.
# Values are integer sample numbers on the original MNE grid, including first_samp.
# xyz is a channels-by-3 array of head coordinates in capture channel order.
# bad_intervals contains half-open sample offsets from the start of this recording.
labels = ("standard", "target")
for directory, assay in (
    ("step2-preservation", Assessment(step2, event_samples, bad_intervals=bad_intervals,
                                      contrast=("target", "standard"))),
    ("ica-preservation", Assessment(ica, event_samples, bad_intervals=bad_intervals,
                                    contrast=("target", "standard"), upstream_step2=step2)),
):
    models = assay.auditory_models(xyz, conditions=labels)
    write_assessment(directory, assay, models)
```

Step 2 compares the filter, the local repair and reference operations, ASR alone, and the entire stage.
Each comparison uses the same final reference on both sides. The report records which stages
were observed. If capture starts after filtering, the filter and whole-stage comparisons are
unavailable. Observed local repairs can still be tested on a synthetic input to the repair
stage. ASR alone can be tested directly on a synthetic input in the effective pre-ASR
reference; the context records when this standalone input was used. A completed repair pass
with zero repairs is distinguishable from a pass that was never observed. If ASR was skipped, the whole-stage result
ends at the pre-ASR recording. If the ASR capture is unavailable, the filter and repair comparisons
remain available and the report identifies the missing comparisons.

Step 3 measures one saved exclusion choice per assessment. The fixed ICA matrix acts only on the
synthetic contribution. Repeat with another capture from the same fit to compare branches.
The optional upstream Step 2 capture supplies the original map normalization reference and repair/ASR
intervals for coverage. Step 3 never reapplies its corrections. Without that capture, the auditory
defaults use the Step 3 reference for map normalization. An original model reference and EEG channel
indices can also be supplied explicitly to `assay.auditory_models`.

The default catalog contains six 5 µV pulse shapes on three maps, five isolated components
(P1, N1, P2, N2, P3) with varying latency and width, and 72 complete-waveform scenarios.
The two labels give 36 pulse configurations, 312 isolated-component configurations and 144
complete-waveform condition configurations. Complete responses sum both event trains on the full timeline
before epoching. Even identical single-event responses can yield different condition averages through overlap.
The profiles include absent P3 and identical-condition controls.

## Supply a known waveform

```python
from erp_preservation import Model, Term, Windows

assay = Assessment(ica, event_samples, Windows(epoch=(-0.2, 1.0), baseline=(-0.2, 0.0),
                                               wide_mean=(0.2, 0.9)),
                   contrast=("target", "standard"))
known = Model(name="response", family="custom",
              terms=(Term(pulse, weights, {"standard": 2.0, "target": 4.0}),),
              focus=weights, polarity=1, peak_window=(0.25, 0.85), mean_window=(0.3, 0.6))
write_assessment("custom-preservation", assay, [known], full_events=True)
```

`pulse` is a sampled, dimensionless waveform at the capture rate. Its first sample defaults to stimulus
onset; use a signed integer `offset_samples` on a term to shift it. `weights` supplies one dimensionless
scalp weight per channel in capture order. Amplitudes are in µV and custom inputs are not renormalized.
Terms can have different maps, lengths and condition amplitudes. The focus map selects the reporting
electrode by largest absolute weight among eligible electrodes in the comparison reference.
Polarity orients the peak search; window means retain signed electrode voltage.
Custom maps need no electrode geometry for numerical assessment.

## Review the output

- `context.json` gives the grid, reference, channel order, included electrodes, windows, stage status,
  and event-by-event coverage. An event crossing an epoch boundary is counted and excluded from measurement;
  its in-bounds synthetic signal can still overlap another event. A BAD, repair, or ASR interval overlapping
  an otherwise complete epoch is a coverage flag. An unknown interval list is reported as null.
- `summary.json` indexes the cases and gives distributions across individual events and across averaged
  waveforms separately. `preservation.pdf` plots event-level peak loss. These distributions describe
  model scenarios and stimulus times within the recording.
- `case_*.npz` contains the sampled inputs, model definition, eligible event samples, individual-event
  metrics, metrics measured on the average, before/after average waveforms and selected individual examples.
  Both unbaselined and baseline-corrected measurements are saved. `full_events=True` also saves every
  baseline-corrected representative-electrode trace. For a specified contrast, the report includes the
  difference wave and signed mean changes for complete models with both conditions.
- The measurements include signed window means, peak amplitude and timing, projection gain, correlation,
  RMS error and relative multichannel error. Expected oriented peaks below 0.1 µV get no peak-loss
  percentage; low expected RMS gets no normalized error or gain. Weak output peaks and undefined
  correlations are marked. The earlier whole-window peak and shape conventions are retained in the
  `legacy_*` fields. A normalized metric can be undefined while an absolute voltage remains measurable.

```python
from erp_preservation import load_case, plot_model, plot_waveforms

case = load_case("custom-preservation/case_0000.npz")
fig = plot_waveforms([("Reviewed ICA", case, "ica")], condition="target")
fig.savefig("target_waveforms.pdf")
plot_model(known, assay, xyz, condition="target").savefig("model.pdf")
```

For a branch comparison, supply a tuple for each branch. Use `event_sample=` to inspect a saved
individual waveform. Waveform arrays use volts, plotted amplitudes use µV, and epoch windows are half-open.
The default epoch spans −200 to +1,400 ms and the baseline spans −200 to 0 ms. The default component
peak window is the modeled peak ± one FWHM; the complete-waveform peak is measured over 250–1,000 ms,
with means over 300–600 ms and 200–900 ms. The measured complete-waveform peak includes contributions
from all overlapping components.

Each case records the requested measurement windows and the actual half-open sample spans.
A partially covered window is flagged and uses its available samples; a window with no
samples yields an unavailable measurement. Without baseline samples only the unbaselined
waveforms are scored. Raw peak shifts remain in the case file for reproducibility. Timing
summaries and largest-shift examples exclude events whose input or output peak is too weak
to time; the distribution's `undefined` count records these exclusions.

Reports are assembled in a temporary sibling directory and published when complete.
Rerun the same command after interruption. A hard stop can leave an unpublished
`.<name>.incomplete-*` directory; it does not block a fresh run. A completed destination
is preserved.

Run `python -m pytest tests/test_erp_preservation.py` for the synthetic checks.
