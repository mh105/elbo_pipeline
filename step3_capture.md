# Replay a reviewed Step 3 ICA reconstruction

After reviewing a fit, the Step 3 example loads the selected labeling, reconstructs
the EEG with MNE, and writes a new folder under the viewer's `reconstructions/`
directory. The Step 4 example uses the same function when preparing its Step 3 input.
Each report contains the spatial transformation, numerical EEG-change measurements
and a figure. The returned cleaned Raw has the input sample grid, channel order
and annotations. If the optional variance or spectral summary cannot be drawn,
its warning is saved alongside the completed reconstruction report. Save that recording explicitly when needed.

## Use a saved labeling or compare exclusion lists

```python
from step3_capture import reconstruct_ica, save_ica_report

# Use this viewer's designated final labeling, or specify version="v2".
after, capture = reconstruct_ica(before, ica, reference_channel=reference,
                                 viewer_dir=viewer_dir)
folder = save_ica_report(output_parent, before, after, capture)

# On the same fit and input, compare an explicit list of zero-based components.
alternative, alternative_capture = reconstruct_ica(
    before, ica, reference_channel=reference, viewer_dir=viewer_dir, exclude=[1, 3])
alternative_folder = save_ica_report(output_parent, before, alternative, alternative_capture)
```

New viewers save a digest of the processed samples. The report identifies an exact
match, grid-only verification for an older viewer, or the absence of a saved recording
identity. Deliberately transferring a fit to another recording requires
`allow_other_recording=True` when a known mismatch is detected. The numerical MNE
check runs independently. Use `reconstruct_ica_report` for the combined reconstruction
and report with one full verification pass.

Each explicit list replaces the complete exclusion list on a copy of the ICA
fit. An empty list keeps all components. MNE still uses the selected PCA reconstruction
rank, so an empty list can change the EEG when residual PCA dimensions are omitted.
A saved decision carries the viewer fit ID, labeling version, review revision, and
the exact component list. Explicit choices are labeled as such and leave the
viewer untouched. For an ELBO fit without a viewer, supply the explicit list;
the model fingerprint then identifies the transform. The archive also records
MNE's version, the effective reference, channel orders, and the PCA setting and
resolved component count.

## Apply a frozen choice to a known waveform

```python
from step3_capture import Step3Capture

capture = Step3Capture.load(folder / "ica_transform.npz")
response_after = capture.replay(response_before, ch_names=channel_names)
```

Provide a channels-by-time array in volts, using the recording's reference and
the channel order saved in the archive. A fixed ICA reconstruction is affine.
If MNE applies `F(X) = L @ X + b`, the change contributed by a separate response
`S` is `F(X + S) − F(X) = L @ S`. Replay returns this contribution. To reconstruct
absolute voltages using the matrix, add the saved `offset_v[:, None]`.
Channels outside the ICA fit pass through unchanged. The same capturing code
checks the reconstructed samples against MNE over the entire recording
(`atol=1e-15` V, `rtol=1e-10`). No additional ICA fit is needed.

## Interpret the EEG-change report

`eeg_change.json` records before, after, and difference RMS amplitudes in µV.
Each time point summarizes all fitted electrodes over one second by default;
the final window can be shorter. Each electrode also gets a recording-wide
measure, `100 × RMS(after − before) / RMS(before)`. This percentage measures
the size of a voltage change. It can exceed 100%. Its interpretation as artifact
or neural signal requires other evidence.

The default uses every sample, including BAD annotations, to show how much
of the stored recording changed. Use `bad_samples="omit"` in `save_ica_report`
to exclude marked samples from the measurements. Both options reconstruct
the entire timeline. The report records BAD intervals as half-open offsets from
the first sample, the first sample number, and contributing sample counts per
window and overall. A window with no eligible samples and a relative change
with a zero denominator are reported as undefined. The original viewer's
variance-reduction summary retains its own unannotated-sample convention.

Run the synthetic checks with `python -m pytest tests/test_step3_capture.py`.
