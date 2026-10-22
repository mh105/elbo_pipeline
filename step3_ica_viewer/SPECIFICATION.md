# ICA viewer specification

Normative visual and numerical behavior, based on the accepted sub-03 interface on 2026-10-19. [reference/viewer_template.html](reference/viewer_template.html) is the visual starting point, not a generalized implementation. Storage and lifecycle rules are in [STORAGE.md](STORAGE.md).

## Shared interface and decisions

- Show the recording identity, actual fitting epoch count/duration, filter range, sampling rate, component count, and initial automatic exclusion count.
- Tabs are ordered **1. Metric distributions**, **2. Full time traces**, **3. IC waveforms**. Open on Metric distributions.
- Keep all fitted components available, including ICLabel-rejected components. Component labels use zero-based `ICA000`, `ICA001`, etc., in MNE component order. Do not sort the navigation list by event count or special-case ICA019.
- All tabs use the rejection set of the selected labeling version. On first launch, Version 1 starts from captured current `ica.exclude`, preserving any deliberate Python manual changes; keep the captured automatic ICLabel decisions as a distinct baseline. An existing single-list review becomes Version 1 with its decisions preserved.
- Replace the static top-right review badge with a **Labeling version** dropdown. List saved versions and identify the stable version. Always open stable, regardless of which version was last edited. **Create new version (ICLabel defaults)** creates and selects the next numbered version from the original automatic exclusions; it does not copy human decisions or change stable. **Mark current version as stable** changes the persistent designation without altering any version's exclusions.
- Autosave each version independently in the same fit's decision JSON. Disable version switching while changes are being saved or unresolved, and prevent decisions during a version change. Version selection preserves the selected IC and trace positions; every tab updates its rejection colors and counts. Python loads stable by default; a caller may explicitly request a version ID.
- Rejecting or accepting an IC is idempotent. Accepting an ICLabel-rejected IC must work. Do not infer rejection from any event-count cutoff.
- Show the current number rejected/kept and saving status. The Python exclusion assignment may remain copyable, but disk saving is authoritative in the reusable product.
- In individual views, current rejections control red dropdown entries, all four metric values and captions, ICLabel probability/category, status, and summary-table rows. Kept entries use the normal dark font. Initial ICLabel status remains separately identifiable.
- Do not add back a manual numeric rejection threshold or the former same-time ICA019 comparison panel.

Preserve component deep links such as `?ic=10` and `?ic=ICA010`. They select that IC without changing the default opening tab. The prototype does not encode tab, event, or inspection time in its URL; extending those deep links is not required for v1.

## Tab 1: Metric distributions

Show two histograms: adjusted original threshold-run count, and adjusted grouped waveform-event count. "Adjusted" means removing shared groups as defined below.

The reference population is fixed to the components **not initially excluded by ICLabel**. Manual decisions do not change this population or reshape the histograms while reviewing. Label the population explicitly, including when the selected IC lies outside it. Use counts derived from the actual recording, not sub-03's 111-component population.

If every component was initially excluded, show an empty reference-population message rather than inventing a histogram or rank.

Retain the selected component picker/navigation and its current decision color. Each distribution shows the selected component's value with a teal indicator, its descending rank where applicable, and the number of reference components at or below that value. The current histogram bin widths are 100 original runs and 25 grouped events. These are display choices, not rejection criteria.

Switching tabs must preserve the selected IC and individual waveform position. Returning to the stacked view preserves its scrolled range unless another selected IC needs to be brought into view.

## Tab 2: Full time traces

This is the multi-component, MNE-like review surface.

Place a reconstructed EEG envelope panel between the shared exclusion summary and the IC-trace panel. It follows the stacked view's exact time window, with matching horizontal plot margins and no independent time navigation. Use all ICA-fitted EEG channels from the retained fitting epochs (1–100 Hz, 250 Hz), rather than the original broadband Raw or dropped channels. Reconstruct in physical µV with MNE's mean, pre-whitening and configured retained PCA residuals restored. Within each 20 ms bin, take the minimum and maximum over all fitted channels and samples; shade between them. Clip bins at the visible window and omitted-time boundaries, and do not connect bands across removed-time gaps.

On initial load, reopening, or selecting/creating a labeling version, show one **Current** envelope in 50%-transparent black. After an actual manual rejection or acceptance, show **Before** in 50%-transparent black and **After** in 50%-transparent green. Before is the complete exclusion set immediately before that one toggle; After is the current complete set. Each subsequent toggle replaces the comparison, including toggles made in Tab 3. No-op accept/reject commands, save acknowledgments and marking the current version stable do not replace it. This comparison exists only in browser-session memory; it is not a saved labeling history. Auto-scale both bands on one shared symmetric µV axis using their visible extrema.

Load full-recording envelope bins once for each exclusion comparison and retain them in browser memory. Horizontal panning, resizing and changing the displayed duration redraw the visible bins locally in the same animation frame as the IC traces. These actions must not clear the envelope, wait for scrolling to stop, or request another reconstruction. Reconstruct the full envelope in bounded chunks on the server; do not materialize a full channel-by-time recording. Only a new decision comparison or version load invalidates it.

| Control or behavior | Contract |
| --- | --- |
| ICs displayed | Default `n_components` (all ICs); positive integer, clamped to available components. |
| First IC | Default 0 (ICA000); zero-based component index, clamped so the requested number of traces fits. |
| Seconds displayed | Default 60 s, limited to retained duration. Round up to whole fitting epochs; display the effective seconds and epoch count. |
| Amplitude | One common scale for all ICs. Initial MNE-style scale is 5 native ICA AU per half-row. Buttons multiply/divide display gain by 1.25. |
| Vertical trackpad scroll | Move up/down through ICs, rather than scrolling the page while the pointer is over the traces. Prototype sensitivity is 30 wheel pixels per IC. |
| Horizontal swipe | Pan retained-data time; use the dominant horizontal wheel direction, or Shift with vertical wheel input. |
| Bottom scrollbar | Always-visible track and proportional thumb; click, drag, wheel, Home/End, and left/right keys move through the full retained duration. |
| IC-label click | Toggle that component's rejection using the shared current decision set. |
| Trace-row click | Open Tab 3 with that row's IC selected and free time inspection centered at the clicked time. Leave rejection decisions and the stacked view's scrolled range unchanged. |

Plot native ICA activations, not per-component robust-normalized traces. Subtract each displayed trace's visible-window mean and use a shared amplitude scale. Native ICA activation units are arbitrary units (AU), not sensor µV. Undo the stored robust scaling before plotting; do not use a separate display SD or MAD per IC.

Use min–max rendering when downsampling to screen pixels so narrow spikes are retained. Lazy-load only the necessary IC waveforms and avoid stale asynchronous draws after rapid navigation.

Accepted traces and labels are near black. Rejected traces use faint pink `rgba(213,132,151,.38)` and labels `rgba(193,103,125,.48)`. Keep the component label visible and clickable. Epoch boundaries are subtle: 0.5 px gray `rgba(120,130,140,.50)`, dotted `[1,4]`. This tab uses the full available window width and grows the canvas with the available viewport height, without the other tabs' page-width cap or a maximum canvas height. Scale trace spacing and amplitude with the resulting row height. Reserve room for the bottom time scrollbar, status and page padding; retain a 320 px minimum canvas height on small windows, where the page can scroll. Page scrolling must not change the calculated canvas height.

## Tab 3: IC waveforms

Preserve this panel order and layout:

1. Component picker labeled with **original threshold runs**, e.g. `ICA010 · 1179 (664) runs`, and component navigation. Color follows the current rejection decision.
2. A single aligned control row: grouped waveform-event selector, view-width selector (2/5/10/30 s), then Previous/Next event buttons.
3. IC mixing-weight topomap in the upper-right corner of the first panel, with the selected IC's median PSD directly beneath it in the same column.
4. Four large metrics: original threshold runs; original runs/minute; grouped waveform events; longest grouped interval. For each, show raw value followed by adjusted value in parentheses. The rate denominator stays the full retained fitting duration.
5. A separate second metric row: ICLabel winning probability and category in the same large font, with a smaller ICLabel caption; next to it, an obvious Kept/Rejected toggle and the corresponding shortcut.
6. Whole-recording overview, zoomed source waveform, five signed delta traces, detection/grouping explanation, and clickable counts-by-component table.

The overview is a min–max source envelope with event intervals and the current inspection window. Clicking it allows free time inspection. From free inspection, Left/Previous selects the closest event whose peak precedes the clicked time; Right/Next selects the closest event whose peak follows it. If none exists in that direction, keep the free view and disable that navigation button. Once an event is selected, navigation proceeds sequentially. Selecting an IC initially focuses its strongest detected group when one exists. Event navigation uses grouped events, even though original runs are the primary diagnostic count. Selecting a long group expands the view enough to include the whole interval and padding. Zero-event ICs must still support free inspection.

Below-sharing-threshold groups are shaded pale red; shared groups are yellow. Short red ticks mark original threshold-run peaks before grouping. A dashed vertical line marks the selected group's strongest delta peak. Event details include bounds, duration, strongest absolute delta Z and lag, original-run count, descriptive shape, pre/post levels, and sharing count/fraction. Expandable details list all covering ICs.

Whole-recording and zoomed source axes show robust units. Keep their existing dimensions (prototype heights 145 and 220 CSS px) and margins. Add secondary µV endpoint values vertically in the existing right margin, and name the corresponding electrode above each plot. These endpoints represent the IC's contribution relative to its median at its strongest physical-loading electrode. Preserve sign; the upper robust endpoint can correspond to negative µV. Signed delta plots retain their own Z units, separate ranges, and dashed ±5 lines.

## Keyboard behavior

| Scope | Keys | Action |
| --- | --- | --- |
| All tabs | Command+1 / +2 / +3 | Select distributions / full traces / individual waveforms. |
| Focused tab strip | Left/Right, Home/End | Accessible tab navigation. |
| Individual waveforms | Up/Down | Previous/next IC. |
| Individual waveforms | Left/Right | Previous/next grouped event. |
| Individual waveforms | `-` / `=` | Reject / accept the selected IC. |
| Full traces | Up/Down | Move first displayed IC by one. |
| Full traces | Left/Right | Pan by one fitting epoch. |

Do not steal ordinary navigation/editing keys from inputs, selects, editable content, or the open component picker. Command tab shortcuts should also work after editing a numeric control. Browser-reserved shortcut delivery must be checked in the browser used for review. Reject/accept shortcuts are confined to the individual waveform view.

## Topomaps

Use saved, native-reference `ica.get_components()` maps in `ica.ch_names` order. Do not restore the omitted zero-reference row or average-reference these display maps. ICLabel's private reference/scaling adjustments are a separate feature-extraction path.

Normalize each displayed map by its own maximum absolute fitted-electrode weight; use `RdBu_r`, symmetric limits ±1, six contours, MNE head outline, nose up, ears left/right. Colors describe within-map weights and do not compare absolute component strengths. Preserve the saved polarity.

Electrode markers have a **black center (`#000000`) and bright green surrounding ring (`#39ff14`)**. Current Matplotlib scatter size is 10 pt² with a 0.65 pt edge. Keep the actual fitted electrode coordinates. Render 360×360 PNGs and display responsively in the first panel; the usual desktop width is 210 px. Version image URLs when marker styling changes to avoid stale cached images.

## Component spectra

Below the selected IC's topomap, show its median power spectral density with a linear frequency axis from 0 to `min(50, sfreq / 2)` Hz and a dB power axis. Use native ICA activation units: dB relative to 1 AU²/Hz, not sensor µV²/Hz or robust-normalized source units.

Compute a density-scaled Hamming periodogram for each exact retained fitting epoch, removing that epoch's mean. Do not concatenate epochs for spectral estimation or use the Float32 waveform display cache. Take the pointwise median of linear PSD values across epochs, then convert with `10 * log10(power)`, flooring zero densities at the smallest positive normal float64 value. For current 2-second epochs the frequency spacing is 0.5 Hz. These are spectra of the 1–100 Hz fitting data, including the corresponding filter effects near the low-frequency end.

Store shared frequency bins and method metadata in the viewer payload, with one dB spectrum per component. Redraw when the selected IC, tab, or viewport changes. Preserve existing topomap size and review controls. Use positive-frequency bins for vertical autoscaling so a numerical DC floor cannot flatten the rest of the plot; draw the full spectrum with clipping to the axes.

Spectra are derived display assets. Updating this calculation or its layout changes the display cache, not the fit identity or durable rejection JSON. Reopening an existing bundle must rebuild from its archived exact ICA and fitting epochs without refitting or resetting manual decisions.

## Numerical units

For native IC source `s`, store `m = median(s)` and `q = 1.4826 × median(abs(s-m))` across retained fitting samples. Store the source trace as little-endian Float32 `r = (s-m)/q`. Native reconstruction is `s = r*q + m`; the stacked plot's visible-mean removal cancels `m`.

Compute source/delta statistics, detection, grouping, and counts at the original fitting-data precision before casting the display cache to Float32. The compressed waveform files are display assets, not the authoritative input for recomputing scientific event counts. Browser-rendered delta curves and ticks can use their stored trace/normalization parameters without replacing the prepared event records.

For this pipeline's EEG fit with `ica.noise_cov is None`, define the physical mixing matrix:

```python
B_uV = ica.pre_whitener_ * ica.get_components() * 1e6
j = argmax(abs(B_uV[:, component]))
endpoint_uV = robust_endpoint * source_scale * B_uV[j, component]
```

Use the signed weight and label electrode `ica.ch_names[j]`. The result is a scalp-channel contribution in the existing recording reference, not the voltage of a neural generator. Do not add `source_median` to the median-relative endpoint. `get_components()` alone has not undone pre-whitening and is not a V/AU matrix.

V1 must reject unsupported covariance-whitened fits rather than display an incorrect µV scale. A future extension must apply the appropriate inverse whitening and verify it against MNE component removal.

## Delta event detector

Compute independently for **all** ICs, including initially rejected ICs. Use the exact fitting epochs: current Step 3 supplies 1–100 Hz, 250 Hz data with BAD-prefixed annotation epochs already removed. Do not train another ICA, use the 0.3-Hz Raw instead, or add transient masking as part of viewer preparation.

1. Use lags 4, 20, 40, 80, 160 ms. At 250 Hz these are 1, 5, 10, 20, 40 samples.
2. For each IC and lag `L`, form `d[t] = r[t+L] - r[t]`. Subtract that lag's temporal median and divide by `1.4826 × MAD(d)` over valid lag pairs. Store the unrounded median and scale.
3. Align delta values at `t + floor(L/2)`. Odd lags therefore have a half-sample timing offset (2 ms at 250 Hz) relative to the exact midpoint; preserve/document this prototype convention.
4. A candidate sample has any absolute lag Z ≥5. Contiguous true samples in the union mask are **original threshold runs**, with half-open `[start, end)` sample bounds. Each run has its own strongest peak.
5. Merge successive runs when the intervening quiet gap is ≤300 ms (75 samples at 250 Hz). A grouped event spans the first run's start to the last run's exclusive end, including bridged quiet gaps. Store the constituent run count. Its peak is the first strongest union-Z sample within the interval.
6. Compute sharing and adjusted metrics below. Grouped intervals remain the waveform navigation unit; original runs remain the primary count.

Runs are detected on the union mask, not counted separately and summed across five lags. Grouping is not pulse-width estimation, amplitude recovery, or evidence that a group contains only one physical artifact. There is no additional arbitrary padding in the stored grouped bounds; view padding is a display operation.

For zero/undefined source or delta MAD, report the degenerate IC/lag explicitly and avoid division by zero or nonfinite serialized values. Do not silently generate spurious threshold violations. No automatic exclusion follows from this condition.

Shape labels describe candidates and never remove them. The prototype calls a group "fast" if its 4- or 20-ms delta reaches 5. Its level-change descriptor uses a 21-sample median-filtered source and the 30-sample difference of that series, robustly normalized and evaluated at the group peak. Separate displayed pre/post medians use offsets `[-25:-5]` and `[+5:+26]` samples at 250 Hz. These are not the same calculation as the level-change contrast. See the frozen detector/build scripts for exact boundary conventions; do not replace them with a new shape classifier while generalizing.

## Shared-event context and adjusted counts

For a target grouped event, count unique ICs whose union of grouped intervals covers **that target event's peak**: `start <= peak < end`. Include the target IC and all initially/manual-rejected ICs. Each IC votes at most once.

Shared means strictly more than 25% of **all fitted ICs**:

```python
minimum_shared_count = floor(0.25 * n_components) + 1
```

For 119 ICs, 30 is shared and 29 is not. Do not use the number of currently kept ICs as the denominator. This is interval coverage at the target peak, not a symmetric overlap or peak-distance test; different groups can consequently have different sharing labels.

Adjusted metrics remove entire shared groups. Adjusted original-run count sums the original runs in the remaining groups; it does **not** classify each original run's peak separately. Adjusted groups, longest interval, and groups longer than 1 s use the remaining groups. Runs/minute use the same full retained-data duration before and after adjustment. Sharing is diagnostic context and does not automatically reject epochs or ICs.

## Time mapping and missing fitting epochs

The viewer's primary timeline remains concatenated **retained fitting time**, as in the prototype. Record the mapping back to original recording time; see [STORAGE.md](STORAGE.md). A displayed 216 s is not necessarily original recording second 216 after rejected epochs are removed.

Adjacent fitting epochs may be concatenated for display, but do not draw a connecting waveform segment or compute lag differences across an actual removed-time gap. Mark such joins subtly and make their original times available. Legitimately contiguous epochs may share valid lag pairs across their boundary.

Exclude lag pairs spanning a discontinuity from robust delta normalization and detection. Runs and groups cannot bridge discontinuities even when concatenated-time gaps appear small. Shape windows cannot borrow samples across a discontinuity; unavailable context must be reported as unavailable. Sharing intervals use the same retained-time coordinate system and cannot include invented gap support.

This is a necessary generalization of the prototype's contiguous-epoch assumption. Preserve sub-03 results when its epochs are contiguous; version detector/cache semantics when introducing gap handling. Do not interpret `fit_epochs.events[:, 0]` using the downsampled epoch rate: those event indices retain the original Raw sampling units.

## Generalization and acceptance

Derive component/channel counts, ICLabel flags, histogram population, epoch count, epoch sample count, durations, available navigation limits, and sharing threshold from inputs. Do not embed sub-03, 119, 8, 111, 415, 500, 830 s, its exclusion list, or its file hashes as product constants. Keep millisecond detector settings separate from their derived sample counts. V1 validates the current 250-Hz/2-second EEG contract rather than pretending to support arbitrary rates.

Before calling the reusable implementation complete:

- Reproduce the frozen sub-03 counts and physical endpoint conversion with the same exact inputs and detector semantics.
- Exercise a second recording with different component/epoch counts and an actual rejected-epoch gap. Verify the gap adds no artificial event.
- Review all three tabs, lazy loading, navigation limits, keyboard focus, trackpad directions, scrollbar, zero-event components, long groups, and negative µV polarity.
- Confirm automatic rejection can be overridden in either review view and is restored after browser/server restart.
- Verify disk-save failure is visible, cache rebuild preserves exclusions, changed fits do not inherit component indices, and explicit loading validates identity and replaces `ica.exclude`.
- Verify metadata and event exports match displayed counts, and source data, annotations, reference, fitted ICA transform, and Raw are not modified by viewer preparation/review.
