# Step 3 ICA viewer

This folder implements the accepted sub-03 review interface for other recordings and documents its data and persistence contracts.

**Status:** reusable v1 implemented. Public helpers live in `../step3_ica_cleaning.py`; `storage.py`, `build.py`, `server.py`, and `viewer_template.html` implement preparation, rendering, disk autosaving, and validated loading. The original sub-03 prototype is frozen in `reference/`. Its old browser-only decisions are not automatically migrated: a new bundle starts from the supplied current `ica.exclude` unless a valid disk review already exists.

## Read these documents

| Document | Purpose |
| --- | --- |
| [SPECIFICATION.md](SPECIFICATION.md) | Accepted tabs, layout, controls, waveform units, topomaps, event detector, and generalization requirements. |
| [STORAGE.md](STORAGE.md) | Required inputs, `.ica_viewer` layout, stable fit identity, durable exclusions, saving/resuming, and explicit Python loading. |
| [reference/README.md](reference/README.md) | Frozen prototype files, their limitations, provenance, and sub-03 acceptance values. |

The specification preserves the current visual review workflow. Event counts are diagnostic context; they do not automatically reject components. The browser is the manual review interface.

## Inputs for a new recording

Use the existing Step 3 option that retains the unrounded ICLabel results:

```python
ica, fit_epochs, iclabel_table = fit_ica(
    raw_bads,
    reference_channel,
    return_details=True,
)
```

The default two-value call remains valid for fitting, but its returned ICA does not contain the ICLabel probability table. Do not refit ICA just to recover that table. Capture it initially, reuse an existing matching table, or explicitly arrange label recomputation with the required reference geometry.

| Input | Why it is needed |
| --- | --- |
| Exact fitted `ica` | Unmixing/mixing transforms, component ordering, fitted channels, geometry, and the current `ica.exclude`. |
| Exact `fit_epochs` | The actual annotation-clean fitting data, retained epoch order, source activations, detector input, and sampling information. |
| Unrounded `iclabel_table` | All seven class probabilities, assigned category/probability, and the original automatic exclusions. |
| Explicit original EEG path | Locates `subject/eeg/.ica_viewer` and links the review to the correct recording, run, and source files. |
| Effective `reference_channel` | Records the actual reference used by Step 3. |
| Metadata from `raw_bads` | Original sample rate, `first_samp`/crop origin, source filenames, and epoch-to-recording time mapping. |

Passing `raw_bads` at initial preparation is convenient for capturing this metadata. Its full signal array is not required by the accepted viewer: all waveform views are derived from `fit_epochs`. Do not save another full preprocessed Raw solely for this viewer. Additional future sensor-channel views would have a separate data requirement.

Pass the explicit original `.set`/EEG path. MNE's `raw_bads.filenames[0]` can refer to an external `.fdt` file rather than the `.set` file you opened; the Step 3 example constructs `eeg_path` explicitly.

CSV files are useful exports, but they are not substitutes for the fitted ICA and exact epochs. The current browser loads embedded JSON (including median component spectra), lazy per-IC Float32 waveform files, and topomap PNGs. It does not read `summary.csv` to draw its waveforms.

The Full time traces tab also shows a synchronized reconstructed EEG envelope in µV across all fitted channels, using 20 ms extrema. It starts as one black **Current** band. After a manual reject/keep toggle in either review tab, it compares black **Before** and green **After** bands for that last change. Reopening or loading a labeling version resets this temporary comparison to Current. Exact native source and reconstruction arrays are rebuildable cache files; no additional Raw recording is needed.

## Intended workflow

1. Run preprocessing and `fit_ica(..., return_details=True)`. Preparation must not silently rerun preprocessing, refit ICA, change the reference, or apply ICA.
2. Prepare a recording-specific, fit-specific bundle under the original EEG file's sibling `.ica_viewer` folder. Preserve the exact fit inputs and ICLabel table, then build the derived viewer cache.
3. Start the local viewer service for that bundle. It serves the HTML and atomically saves every exclusion change to the bundle's human-readable decision file.
4. Review through the three tabs. The top-right **Labeling version** menu switches between saved versions, creates a new version from the original ICLabel decisions, or marks the selected version as final. Reopening always restores final, including manual acceptance of ICLabel-rejected components in that version. Existing reviews become Version 1 without losing decisions.
5. **Explicitly load the saved list when ready.** This was the user's selected workflow. The open/load helpers do not block for review, automatically alter an ICA object in another Python process, or apply ICA to `raw_bads`.
6. Assign the validated list to `ica.exclude`, replacing its existing contents. Apply ICA separately when ready.

After fitting, use these implemented helpers. `viewer_dir` is a `Path` to the recording-and-fit directory; it is not a separate MNE object:

```python
from step3_ica_cleaning import (
    prepare_ica_viewer, open_ica_viewer, load_ica_exclusions,
    summarize_ica_exclusions,
)

viewer_dir = prepare_ica_viewer(
    ica,
    fit_epochs,
    raw_bads,
    eeg_path=eeg_path,
    reference_channel=reference_channel,
    iclabel_table=iclabel_table,
)
viewer_url = open_ica_viewer(viewer_dir)

# Later, after review, validate the target ICA and replace its exclusions.
ica.exclude = load_ica_exclusions(viewer_dir, ica=ica)
# Defaults to final. To explicitly use another saved labeling:
# ica.exclude = load_ica_exclusions(viewer_dir, ica=ica, version="v2")
exclusion_figure = summarize_ica_exclusions(ica, fit_epochs, raw_bads)
# Applying ICA remains a separate analysis step:
# ica.apply(raw_bads)
```

The summary prints the excluded indices/count and the joint percentage reduction in summed temporal channel variance after excluding them. It uses the original-rate `raw_bads` samples outside BAD annotations for both variance estimates, and only fitted channels in the original reference. The printed support gives retained and total sample counts. Call it before modifying `raw_bads` with ICA. Its figure shows three topomap/median-PSD pairs per row, using the same exact fitting-epoch spectra as the viewer. It leaves inputs unchanged and returns the figure (or `None` for no exclusions).

A new session can reopen an existing bundle without an active Python kernel or full Raw. Rebuilding a missing cache uses the stored ICA, exact epochs, labels, and manifest. Objects that existed only in a closed kernel cannot be recovered without first saving them.

The Step 3 example separates preparation/opening and explicit loading into `# %%` cells. The final cell asks for Enter before loading, so running the entire file also allows time for review. The server stays alive while its Python process is running.

To reopen a viewer without rerunning preprocessing, run this from the pipeline directory:

```bash
python -m step3_ica_viewer /absolute/path/to/the/recording/fit-directory
```

This prints the URL and serves until Ctrl+C. Add `--no-browser` to print the URL without opening a browser. In a running kernel, `open_ica_viewer(viewer_dir)` starts a background service and returns immediately.

After updating viewer code, reopen the same `viewer_dir`; the code-versioned display cache rebuilds from archived inputs and preserves the saved exclusion JSON. Do not delete `.ica_viewer`, rerun preprocessing, or refit ICA. If the builder was already imported in a running kernel, refresh it before reopening and follow the returned URL:

```python
import importlib
import step3_ica_viewer.build as viewer_build

importlib.reload(viewer_build)
open_ica_viewer(viewer_dir)
```

Do not reload `server.py`, because its module state tracks running local services. A fresh command-line process uses the updated builder directly.

The labeling-version and reconstructed-envelope updates also change the server code. For an already-running older service, launch a fresh process with `python -m step3_ica_viewer /absolute/path/to/viewer_dir` and use its new URL. This reads the same saved fit and decisions; no preprocessing, ICA refit, or deletion of `.ica_viewer` is needed.

## Implementation decisions and boundaries

Implemented requirements are the three-tab interface, subject-local hidden storage, persistent final exclusions, explicit Python loading, separate folders for separate fits, immutable fit snapshots, and a small local write service.

Keep v1 focused on this pipeline's EEG input: 1–100 Hz fitting data, 250 Hz sampling, equal 2-second nonoverlapping retained epochs, and standard MNE whitening without `noise_cov`. Derive sizes and durations from the actual objects, validate this support contract, and replace sub-03 constants. Broader recording types or whitening models require a deliberate extension.

V1 archives inputs inside the bundle. Additional NumPy arrays preserve exact fitting samples or the original mixing matrix when MNE's FIF round-trip changes their numerical values; rebuilding uses them automatically. Referencing external fit snapshots instead of archiving them is not implemented.

## Request template for a future session

> Read `code/eeg_analysis/elbo_pipeline/step3_ica_viewer/README.md`, `SPECIFICATION.md`, and `STORAGE.md`. Prepare or reopen the ICA viewer for `<original EEG path>`, using `<saved fit bundle or available ICA, fit_epochs, ICLabel table, and raw metadata>`. Preserve the accepted interface, save decisions under this recording's `.ica_viewer` directory, and do not refit or apply ICA. Use the implemented helpers rather than the frozen sub-03 scripts.
