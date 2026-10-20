# Frozen sub-03 reference

These files preserve the accepted implementation as of 2026-10-19, including the three-tab interface, manual review controls, vertical physical-voltage endpoint labels, and black-center/green-ring topomap markers.

They are **reference snapshots**, not the reusable product. In particular:

- `viewer_template.html` has the accepted visual and interaction implementation but still contains sub-03 constants and browser-only decision persistence.
- The Python scripts contain historical report paths/imports and subject-specific assertions. **Do not run them from this folder as a generic builder.** Their preserved functions/formulas provide an implementation reference while extracting the reusable code.
- No EEG arrays, ICA/epoch FIF files, actual browser decisions, or large generated viewer assets are copied here.
- [../SPECIFICATION.md](../SPECIFICATION.md) and [../STORAGE.md](../STORAGE.md) govern the reusable implementation where they deliberately extend the prototype, especially disk autosaving, stable numerical identity, and discontinuous retained epochs.

`source_manifest.json` identifies each copied source, its original repository-relative location, and SHA256. `sub03_baseline.json` contains a small numerical reference extracted from the current metadata, not a policy for other subjects.

## Source roles

| Snapshot | Role |
| --- | --- |
| `viewer_template.html` | Accepted tabs, layouts, canvas rendering, navigation, and manual review state. |
| `build_viewer.py` | All-component sources, multiscale deltas, original runs/groups, shape descriptors, payload and CSV exports. |
| `add_shared_context.py` | Unique-IC interval coverage at event peaks, adjusted counts, and context validation. |
| `generate_topomaps.py` | Native-reference maps, normalization, electrode geometry, black/green markers. |
| `prepare_native_sources.py` | Exact ICA source scaling, physical mixing calibration, and Float32 reconstruction checks. |
| `analyze_delta_z5.py` | Original robust-parameter, run-boundary, grouping, and event-characterization implementation. |

## Expected sub-03 behavior

The reference fit has 119 ICs, 415 retained epochs ×500 samples at 250 Hz, 830 s of 1–100 Hz fitting data. Initial corrected ICLabel exclusions are `[0, 8, 9, 12, 17, 71, 110, 118]`; these are not defaults for a new recording. Histogram reference population is 111 ICs.

| IC | Original runs | Adjusted runs | Groups | Adjusted groups |
| --- | ---: | ---: | ---: | ---: |
| ICA001 | 2950 | 2406 | 542 | 524 |
| ICA010 | 1179 | 664 | 148 | 114 |
| ICA019 | 255 | 71 | 52 | 21 |

ICA010's strongest physical loading is E14, +4.092285681690222 µV per native ICA AU; its robust source scale is 0.4092668527707323 AU per robust unit. Thus one robust unit corresponds to +1.6748368815840882 µV at E14, relative to the component median. ICA006 provides a negative-polarity check: E21 has −29.340184630432077 µV per native ICA AU.

The existing shared-context boundary checks are ICA001 event 176 with 30 covering ICs (shared) and ICA003 event 155 with 29 (not shared). The fixed initial-ICLabel-retained population has 28,961 adjusted original runs and 6,831 adjusted groups. These checks can be reproduced when the original saved sub-03 inputs are available; they do not justify hardcoding any counts.

The accepted live artifacts currently reside in `../../reports/sub03_delta_z5_all_retained_20261018/`, with fit inputs in `../../reports/sub03_step3_ica_assessment_20261016/`. Those report directories are ignored by Git. The snapshots here preserve the code and specification without depending on those ignored directories being present in a fresh checkout.
