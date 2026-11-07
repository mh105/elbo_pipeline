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
Insufficient calibration keeps the ordinary ELBO exception and records the
reason in the capture. An applied ASR correction with no corrected samples is
an identity transformation. If its matrices cannot be verified, the EEG keeps
the ordinary correction, the archive records `asr_status="unavailable"`, and
ASR replay reports the reason.

## Timing at the ends of the recording

Capture uses the public `process_asr(..., store_reconstruction_matrices=True)`
diagnostics for standard ASR. Each blend uses the original update width before
clipping to the output sample grid. If a correction is active at the first
visible sample, a preceding update can fall entirely in the library's lookahead
padding. In that case only, a second diagnostic pass on the same real signal
and calibrated state exposes that matrix: the reflected tail is made explicit
and diagnostic lookahead is set to zero. The visible matrices must agree exactly
with the ordinary pass, and replay must reproduce the corrected EEG
(`atol=1e-12` V, `rtol=1e-10`). Only applied transitions are retained.
Loading the archive for a synthetic signal does not invoke ASR fitting or
processing. FIR replay uses MNE's overlap-add implementation; MNE and
mne-denoise versions are saved with the archive.

Run the artificial checks from this checkout with
`python -m pytest tests/test_step2_capture.py`.
