# Inputs, bundles, and persistent exclusions

This document specifies the reusable product implemented by `storage.py`, `server.py`, and `build.py`. The frozen sub-03 scripts in `reference/` retain their historical behavior.

## Supported inputs and initial capture

The initial preparation call receives the exact fitted ICA, the exact fitting epochs, the unrounded ICLabel table, the effective reference name, the explicit original EEG path, and the Raw metadata needed for provenance/time mapping. Accepting `raw_bads` is the convenient way to obtain that metadata; retain no extra full Raw signal copy just for the viewer.

Validate:

- ICA component indices, ordered fitted channel names, transforms, and epoch channels agree; all required electrode positions are finite.
- Epoch data really are the fitting data. Do not reconstruct a different set from the raw recording simply because the filter/sampling metadata look similar.
- The v1 contract is EEG, standard whitening (`noise_cov=None`), 1–100 Hz at 250 Hz, equal nonoverlapping 2-second epochs. Determine sizes from the objects and validate them.
- The ICLabel table has exactly one row per zero-based component and preserves the seven unrounded probabilities, winner, winning probability, and automatic-exclusion flag. Capture the original ICLabel threshold/model/version too.
- The original source filename, source sample rate, source `first_samp`, crop origin/reference, and per-epoch original event samples are known. A missing original-time mapping must not be silently fabricated from decimated event indices.

Current `fit_ica(..., return_details=True)` supplies the table. Its default two-value return does not attach probabilities to ICA. Missing labels are a missing input, not permission to refit ICA or silently display guessed ICLabel results. Recomputing labels with the current private feature path needs the effective reference's geometry, which is absent from `fit_epochs` after the reference channel is dropped.

Preserve the actual automatic exclusion rule as provenance: current Step 3 rejects when **any** of the muscle, eye, heart, line-noise, or channel-noise probabilities is strictly greater than `iclabel_threshold` (default 0.75). Brain and other do not trigger that rule. A winning category/probability alone does not fully specify the automatic decisions.

## Folder layout

Place `.ica_viewer` next to the original EEG file, under that subject's `eeg` directory. Separate recordings/runs and separate fits. Recommended layout:

```text
sub-04/eeg/
  sub-04_task-ColorPrecMemStudy_run-1_eeg.set
  sub-04_task-ColorPrecMemStudy_run-1_eeg.fdt       # if externally stored by EEGLAB
  .ica_viewer/
    sub-04_task-ColorPrecMemStudy_run-1_eeg.set/
      <fit-id>/
        manifest.json
        sub-04_task-ColorPrecMemStudy_run-1_eeg_ica_exclude.json
        inputs/
          fitted-ica.fif
          mixing-matrix.npy                      # only if FIF recomputes different mixing values
          training-epo.fif
          training-data.npy                      # only if FIF needs exact-sample supplementation
          iclabel.csv
          epochs.csv
        cache/
          <cache-id>/
            index.html
            review_metadata.json
            summary.csv
            events.csv
            data/
              ICA000.js
              ...
            topomaps/
              ICA000.png
              ...
```

The recording directory includes the exact original basename **including its extension**, avoiding `.set`/`.fif` collisions. The decision filename visibly includes that recording's stem and is a UTF-8 text file. JSON is the recommended text representation because it carries both the exclusion list and the identity needed to interpret its indices safely.

Create missing viewer directories when preparing a bundle; do not modify or rename the original EEG or its companion data files. If several fits exist, require an explicit fit selection or exact match rather than picking the newest directory and assuming its component numbers still apply.

## Durable records versus rebuildable cache

| Artifact | Purpose | Retention |
| --- | --- | --- |
| Decision JSON | Authoritative final `ica.exclude` and save revision. | Durable; never delete/reset during cache regeneration. |
| Manifest | Recording identity, fit identity, preprocessing/timing metadata, software versions, files, and checksums. | Durable. |
| Fitted ICA | Immutable transform snapshot for this fit; its original exclusions are only an initial snapshot. | Durable by default. |
| Exact fitting epochs | Rebuild sources, metrics, and plots without rerunning preprocessing. | Durable by default. |
| ICLabel table | Preserve unrounded class results and the automatic baseline. | Durable. |
| Epoch mapping CSV | Reconstruct retained-time/original-time relationship and discontinuities. | Durable. |
| HTML/JSON, waveform JS, topomap PNGs, event/summary CSVs | Browser assets and derived diagnostics. | Rebuildable cache. |

The hidden directory is not entirely temporary: it contains the manual review record. Deleting a cache is different from deleting a review. Rebuildable files may be removed only when their saved inputs remain available.

V1 saves ICA and exact fitting epochs inside the bundle. Referencing existing external files by path and checksum is a possible future storage extension, not a current preparation option. Save epoch samples at double precision when using FIF (`fmt="double"`) if exact regeneration from the in-memory fit is required, and verify the round-trip. Avoid relying on default single-precision epoch serialization for exact data identity. Record any FIF split-file parts.

The implemented v1 always archives a copy of the ICA and fitting epochs. MNE's epoch writer can temporarily rescale the supplied buffer, so save an epochs copy. Even double-precision FIF can round calibrated samples; when its round-trip is not bit-exact, save `inputs/training-data.npy` with the original samples. Rebuilding overlays that array only onto newly loaded epochs and verifies its canonical sample fingerprint. Caller objects remain unchanged, and the manifest/checksums cover the optional array. Captured electrode XYZ coordinates also restore geometry rounded by FIF metadata serialization.

MNE saves the unmixing matrix in ICA FIF files and recomputes the mixing matrix on read. After component sorting, that recomputation can differ from the original mixing matrix by floating-point round-off. When needed, archive the original matrix in `inputs/mixing-matrix.npy` and restore it only onto newly loaded ICA objects. Verify the full model fingerprint after restoration, retaining exact transform identity and existing review decisions. The FIF and optional array together form the exact ICA archive; a bare `mne.read_ica()` does not restore the supplement.

The initial source Raw need not be saved again. For EEGLAB source identity, account for both `.set` and its external `.fdt` where present; checking only the small header file is insufficient if samples live elsewhere.

## Identity: recording, fit, and cache

Source filename determines where the bundle belongs, but filename alone is not enough to reuse exclusions. Component indices are meaningful only for the corresponding fitted decomposition.

Use separate, versioned identities:

1. **Recording identity:** original basename plus source-file/companion-file checksums and relevant import/reference provenance. Keep the source path as useful metadata; moving the same files need not invalidate the numerical fit.
2. **Model identity:** a canonical fingerprint of ordered fitted channel names/types, effective reference and projection/whitening configuration, component count, `pre_whitener_`, `pca_components_`, `pca_mean_`, `mixing_matrix_`, and `unmixing_matrix_`. Include the exact transform semantics. Exclude `ica.exclude`, display state, timestamps, and incidental filesystem paths.
3. **Training identity:** exact fitting samples, ordered channel metadata, original event/sample map, sampling/filter information, and retained support. Include geometry used for the displayed maps/scales in the bundle metadata.
4. **Fit ID:** combine the recording, model, and training identities. Store full fingerprints in the manifest; a short directory label is allowed only if full identities are still checked and collisions are handled.
5. **Cache ID:** fit identity plus detector, rendering, and schema versions/settings. Changing display styling or detector settings can regenerate the cache without discarding this fit's manual exclusions.

Define canonical fingerprint encoding once: stable field order, explicit array shape/dtype/endianness, and normalized serialization. Do not use Python's process-randomized `hash()`. Preserve saved numerical values exactly enough for the identity contract and verify load/save round-trips.

**Do not use the complete ICA FIF file hash as the durable model identity.** Saving a changed `ica.exclude` can change that file hash without changing the decomposition. File hashes still verify archived inputs, but they are different from the immutable numerical model fingerprint.

A changed fit identity creates a separate review even if subject, channel count, component count, or filenames match. A deterministic rerun with exactly identical recorded inputs and transforms may correctly match the same content fingerprint. Do not automatically transfer rejected indices across different fits. Updated ICLabel results or detector versions do not silently reset an existing final review of the same fit; retain the original automatic baseline with its provenance.

## Decision text format

Recommended v1 file, `<recording-stem>_ica_exclude.json`:

```json
{
  "schema_version": 1,
  "recording_filename": "sub-04_task-ColorPrecMemStudy_run-1_eeg.set",
  "fit_id": "sha256:<full-fit-fingerprint>",
  "n_components": 119,
  "revision": 12,
  "updated_at_utc": "2026-10-19T23:00:00Z",
  "initial_auto_exclude": [0, 8],
  "exclude": [0, 3, 10]
}
```

Numbers and exclusions above are illustrative, not defaults for another subject. The `exclude` field is the **complete current final list**, sorted and unique, using zero-based integer indices. `[]` is a valid saved decision to keep every component; it must not be mistaken for a missing review. Reject booleans, noninteger values, duplicate/out-of-range indices, corrupt files, and identity mismatches rather than guessing.

One authoritative structured text file avoids an exclusion-only `.txt` and separate identity sidecar drifting apart. A plain list or Python assignment can still be copied/exported for convenience, but must not become a competing saved state.

Record the original automatic list separately from the final list so manual acceptance is representable. A saved final list **replaces**, never unions with, the automatic or in-memory `ica.exclude` list. Do not write manual choices into the ICLabel probability table.

## Autosave and resume

Plain static HTML and the current read-only file server cannot reliably write arbitrary subject-local files. The reusable viewer needs a small local HTTP service configured with one bundle directory. Serve browser assets and read/write that bundle's decision file; the browser must not supply arbitrary output paths.

On open:

1. Validate the bundle manifest and fit identity.
2. If a valid decision file exists, load its final list. If no decision file exists, initialize it from the captured current `ica.exclude`, recording the distinct initial automatic baseline from the ICLabel table. The first launch may already contain deliberate manual decisions made in Python.
3. A corrupt/mismatched existing file is an error requiring resolution, not an invitation to overwrite it with ICLabel defaults.

On every toggle:

1. Update shared UI state immediately and show **Saving**.
2. Send the complete desired exclusion set with the expected save revision. Serialize/queue writes so rapid toggles cannot make an older request overwrite the latest decision.
3. Validate component indices/fit identity, write a temporary file in the same directory, and atomically replace the authoritative JSON. Acknowledge only after a durable write; return the new revision.
4. Show **Saved** with the acknowledged revision/time. On failure, show **Unsaved changes** and keep a retryable pending decision; never claim the file was saved.

Use one active writer per bundle in v1. A stale revision from another tab/process must trigger a reload/conflict message instead of silently overwriting newer decisions. Reopening the browser or restarting the service restores the last acknowledged on-disk state.

Do not depend on closing the page, downloads, or a final button to save decisions. Python does not wait for browser closure. Browser storage may hold view preferences or clearly identified pending recovery data, but must not silently override a newer disk decision file. Existing prototype `localStorage` decisions are not a new subject's baseline and are not automatically migrated.

A minimal server contract is: fetch manifest/decision state; replace decisions with an expected revision; serve the immutable cache. The implemented routes are `GET /api/review` and `PUT /api/review` with the full `exclude` list and `expected_revision`. No database, remote service, user accounts, or multi-reviewer merging is required for v1.

## Explicit loading into Python

The user selected **explicit loading when ready**, rather than blocking Python until review finishes.

The implemented loader accepts an explicitly selected bundle and target ICA. It checks the bundle's manifest/decision identity internally, verifies the target ICA's numerical model/component fingerprint, validates the saved list, and returns the exclusions. The caller assigns that result to `ica.exclude`. It must not silently choose among multiple fits, union in ICLabel exclusions, apply ICA, or modify the archived transform snapshot.

An ICA object alone cannot independently prove which recording or training samples produced it. The bundle records that provenance; independently checking a live recording or training set requires passing its corresponding metadata/epochs too. Do not promise that `load_ica_exclusions(bundle, ica=ica)` can establish provenance absent from its inputs.

Updating a live Python ICA is separate from the browser's autosave. Reopening the viewer does not require a live kernel. Applying ICA to `raw_bads` remains a separate downstream decision. If the caller saves an ICA file with the final exclusions, this does not change the bundle's model identity and must not replace the authoritative review record.

## Timing metadata

Store original source sample rate, `raw_bads.first_samp`, crop origin, epoch `tmin`, decimated sampling rate, and samples per epoch. For each epoch, `epochs.csv` records:

- Retained zero-based epoch index and, where available, original `fit_epochs.selection` index.
- Original `events[:, 0]` sample and event code.
- Original first/last represented sample or an equivalent precisely defined half-open support interval in original-sample units.
- Retained concatenated start/end samples and seconds.
- Whether this epoch is contiguous with the previous one in original recording time.

For the present pipeline, decimating 1000-Hz Raw epochs to 250 Hz does **not** rescale event sample indices. Event times relative to the cropped Raw use `(event_sample - raw.first_samp) / original_sfreq`, with epoch `tmin` and decimation offsets handled explicitly. Do not divide original event samples by 250 Hz. Preserve the distinction between epoch duration (`n_samples / sfreq`, 2 s) and the last sample timestamp (`(n_samples - 1) / sfreq`, 1.996 s).

Use this mapping to prevent differences/grouping across removed-time gaps, to label discontinuities, and to recover where a reviewed event occurred in the original recording.

## Browser assets and CSV exports

Retain the prototype's simple runtime representation unless there is a demonstrated need to change it:

- `index.html`: accepted template with embedded recording/component metadata.
- `review_metadata.json`: the same machine-readable payload for inspection/rebuild.
- `data/ICA###.js`: lazy `receiveICSignal(id, base64)` calls containing little-endian Float32 robust-normalized source arrays, with native affine parameters in metadata.
- `topomaps/ICA###.png`: per-IC images and a rendering-version cache key.
- `events.csv`: grouped bounds, peaks, shape/context fields, constituent run counts, and sharing classification.
- `summary.csv`: original and adjusted component metrics and initial ICLabel fields. Manual final exclusions live in the decision file, not a stale CSV column.

Save the detector configuration, versions, timing semantics, native source scaling, physical electrode calibration, and input fingerprints in metadata/manifest. Do not rely on a long chain of subject-specific report directories to locate inputs. A builder for a new subject must work from its bundle and the checked-in viewer code.
