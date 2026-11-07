# Replay Step 2 on a fixed signal

Create one capture on the Step 1 recording, before the Step 2 high-pass filter.
Supply the original reference and the channel and flatline decisions produced
before Step 2. The ordinary return values of the cleaning functions still apply.

```python
from step2_bad_ch_time_detection import bad_ch_time_detection_inplace
from step2_capture import Step2Capture, filter_step2_inplace

capture = Step2Capture(raw, flatlines["reference_channel"])
filter_step2_inplace(raw, capture=capture)
details, effective_ref = bad_ch_time_detection_inplace(
    raw, bad_ch_report, flatlines, capture=capture,
)
capture.save("step2-capture.npz")
```

The archive records whether the filter, local repairs and reference check were observed.
A completed check can legitimately apply no change. Starting capture after a stage leaves
that stage unverified. Older archives still support replay of their saved operations.

The archive contains the sample rate, first sample, channel order, EEG picks,
references and bad-channel names, filter taps and annotation-delimited bounds,
production repair projections and fades, and ASR transitions. Bounds are
half-open sample offsets from the beginning of the captured recording. Donor
estimates from temporary detection copies do not enter the capture. Files use
numeric arrays and JSON and can be loaded with pickling disabled.

```python
capture = Step2Capture.load("step2-capture.npz")
# signal: channel × time array in volts, in the original order and reference
after_step2 = capture.replay(signal)
before_asr = capture.replay(signal, stage="pre_asr")
after_asr = capture.replay(before_asr, stage="asr_only")
```

Use `stage="filtered"` for the filter alone. ASR-only replay expects data in
the effective pre-ASR reference. When ASR was deliberately omitted, call
`capture.skip_asr(reason)`; full replay then applies just the earlier corrections.
The Step 2 workflow records a skipped ASR if calibration support is insufficient.
Direct callers of the lower-level ASR function receive a ValueError with the
available and required durations. An applied ASR correction with no corrected samples is
an identity transformation. If its matrices cannot be verified, the EEG keeps
the ordinary correction, the archive records `asr_status="unavailable"`, and
ASR replay reports the reason.

## Recovery and handoff

During initial preparation, call `assess_persistent_flatness(raw, flatlines=flatlines)`
after bridging local flat intervals and annotating global dropouts. The default
uses the existing BAD annotations. With the optional `voltage_mask=True`, the
voltage screen subtracts each channel's median in a temporary calculation.
Thresholds, voltage data, and reference-channel identification are unchanged.
If no samples qualify, the original intervals remain available and
`flatlines["persistent_flatness"]` records the warning.

If a native-reference channel check has no usable samples or windows, Step 2
records which check was unanswered. Once the robust average-reference assessment
succeeds, ELBO retries those checks on a repaired and reference-corrected copy.
Any new bad channel triggers a fresh repair from the filtered samples, with
that channel excluded from the donor pool. The final repairs are applied once
to the working EEG and checked against that settled copy. The capture excludes
all provisional operations. The report lists any check that is still unanswered.

To save checkpoints and a report, pass a **new directory** as `output_dir` to
`bad_ch_time_detection_inplace`. It writes double-precision FIFs for completed
stages and a `step2_report.json`. The report is also returned in
`details["step2_report"]` when running in memory. On completion,
`report["handoff"]` identifies the stage and `report["checkpoints"][stage]`
identifies the saved EEG file. Use `report["reference_channel"]` for Step 3.
With less than 60 seconds of candidate or automatically selected calibration
data, the handoff is the full pre-ASR recording. The report states which
calibration step fell short and by how much. If ASR applied a correction but
its replay could not be verified, the handoff is the actual post-ASR recording
and replay is marked unavailable.

A missing diagnostic figure gets its own warning. A failed numerical operation
outside the documented recovery cases raises an exception. The report retains
the last completed checkpoint and identifies the failed stage; such a run has
no completed Step 3 handoff. Repair bounds in the report are half-open sample
offsets from the start of the recording.

## Timing at the ends of the recording

Capture uses the public `process_asr(..., store_reconstruction_matrices=True)`
diagnostics for standard ASR. Capture was tested with MNE 1.13.2 and mne-denoise
0.0.3. If the installed processor has no matrix-export option, ordinary ASR still
runs once and the report marks ASR replay unavailable. Each blend uses the original update width before
clipping to the output sample grid. If a correction is active at the first
visible sample, a preceding update can fall entirely in the library's lookahead
padding. In that case only, a second diagnostic pass on the same real signal
and calibrated state exposes that matrix: the reflected tail is made explicit
and diagnostic lookahead is set to zero. The visible matrices must agree exactly
with the ordinary pass, and replay must reproduce the corrected EEG
(`atol=1e-12` V, `rtol=1e-10`). Only applied transitions are retained.
Checkpoint replay returns only endpoints observed continuously from the Step 2 input.
Standalone local and ASR replay accept a known signal at the corresponding stage input.
Loading the archive for a synthetic signal does not invoke ASR fitting or
processing. FIR replay uses MNE's overlap-add implementation; MNE and
mne-denoise versions are saved with the archive.

Run the artificial checks from this checkout with
`python -m pytest tests/test_step2_capture.py tests/test_step2_recovery.py`.

The [synthetic ERP assessment](erp_preservation/README.md) uses this capture to
measure the filter, repairs, ASR and whole-stage effects on independent model waveforms.
