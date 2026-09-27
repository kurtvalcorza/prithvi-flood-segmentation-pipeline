# DIMER Philippines Flood Mapping Capstone — Specification

Prepared: 2026-09-27.
Status: Design only. Dataset catalogue and metadata inspected; raster pixels, model execution and capstone qualification remain unverified.

## 1. Learning artifact and scope

**Title:** Mapping Water After Typhoon Ompong: Can Prithvi Generalise to Ilocos Sur?

**Driving question:** Does a flood-trained foundation model improve on a simple spectral baseline in Philippine scenes, and what can the comparison actually establish?

- Artifact: `tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb`.
- Proposed host: `kurtvalcorza/prithvi-flood-segmentation-pipeline`.
- Preserve the existing model tutorial; this is a separate scientific capstone using its verified model semantics.
- Normative basis: DIMER Notebook Specification 2.2; profile `TASK-INFERENCE`, mode `WORKSHOP`, standalone. This is an end-to-end scientific investigation, not an E2E adaptation-profile claim.
- Placement: optional capstone after Earth Observation; basic Python/Colab familiarity. Explain geospatial concepts before use.
- Default: public pinned data, frozen model, no credentials/uploads, no manual restart, fresh Colab T4 Run all.
- GPU execution only on authorised hosted compute; local verification is CPU-only.
- No fine-tuning in version 1. An adaptation extension requires separate training geography, a frozen selection protocol and new hosted evidence.

## 2. Learning outcomes

Learners will explain satellite bands, pixels, spatial resolution, clouds and label uncertainty; compare a simple baseline with Prithvi; distinguish water from newly inundated land; interpret IoU, precision, recall and F1; run one controlled experiment; inspect spatial failures; and write a bounded scientific conclusion.

Use the shared sequence: orient -> explain Input / Model / Output -> predict -> run -> notice -> interpret -> change one thing -> compare -> conclude.

Completion notes are optional learning aids, not required submissions.

## 3. Verified dataset lead

Dataset: `isp-uv-es/WorldFloodsv2`.
Pinned candidate revision: `1f3faa2989e69930ac31d5c0a79fd224461123f8`.
Event: Copernicus EMSR312, Typhoon Ompong/Mangkhut, September 2018.

| Proposed role | Scene ID | Upstream directory | Sentinel-2 time, UTC | Reference time, UTC |
|---|---|---|---|---|
| Development / threshold selection | `EMSR312_08CANDON_DEL_MONIT01_v1` | `val` | 2018-09-18 02:31:20.980 | 2018-09-17 10:06:08 |
| Held-out geographic evaluation | `EMSR312_07VIGAN_DEL_MONIT01_v1` | `train` | 2018-09-18 02:31:20.980 | 2018-09-17 10:06:08 |

These are **capstone roles**, not a claim to follow the WorldFloods benchmark split. Vigan being in upstream `train` means it must not be called an unseen test for a model trained on WorldFloods. Audit the selected Prithvi checkpoint's task-training provenance and document unresolved pretraining overlap.

Published metadata records Vigan at 2,843 x 2,331 pixels, Candon at 2,969 x 2,434 pixels, EPSG:32651, 10 m output grid. Sentinel-2 files are approximately 77 MB and 86 MB compressed. Metadata counts imply about 0.3% cloud-labelled pixels in Vigan and 7.1% in Candon over their full raster rectangles; these are not independently measured cloud-free guarantees.

For each selected ID, retrieve only its pinned `S2/<id>.tif`, `gt/<id>.tif`, `PERMANENTWATERJRC/<id>.tif`, `meta/<id>.json`, and associated flood-vector provenance where required. Include the pinned dataset metadata and attribution. Freeze exact paths, byte sizes and SHA-256 values after acquisition; never download the complete dataset by default.

Dataset licence: CC BY-NC 4.0. Preserve attribution and the non-commercial restriction separately from code/model licences. Public availability does not remove that restriction; confirm course/distribution use is compatible before release.

## 4. Mandatory data feasibility gate

Before notebook implementation is considered ready:

1. Read the actual rasters and confirm availability, checksums, band count/order, dtype, units, nodata, CRS, transform, bounds and alignment. Inspect representative RGB/false-colour and mask overlays.
2. Verify optical processing level against the checkpoint's expected Sentinel-2 L1C input. Do not silently substitute L2A or HLS. If the published input differs, document and resolve the mismatch before treating results as comparable.
3. Confirm the six input bands: B02 blue, B03 green, B04 red, B8A narrow NIR, B11 SWIR1, B12 SWIR2. Derive indices from verified metadata/provider documentation, not guessed array positions.
4. Establish explicit reflectance conversion from the product contract. Apply it exactly once; validation and repeated inference must not rescale already-normalised values. Test bright valid pixels and repeated validation. Record ranges before/after conversion.
5. Confirm WorldFloods v2 mask channels: cloud/clear and land/water. Published metadata encodes cloud channel 0 invalid / 1 clear / 2 cloud and water channel 0 invalid / 1 land / 2 water. Verify actual values and map clear land to 0, clear water to 1, all other pixels to ignore.
6. Measure effective clear/valid pixels, water support and candidate flood support. Identify unlabelled areas; never treat missing coverage as dry land.
7. Audit duplicate scene versions and overlapping footprints. Exclude the older DEL versions of Vigan and Candon. Do not mix different masks for the same optical pixels across roles.
8. Require at least 50% clear valid coverage and at least 10,000 water and 10,000 land pixels per selected scene, plus at least four nonoverlapping 512-pixel windows with 80% valid coverage for interpretation. These are feasibility criteria, not statistical independence guarantees. Failure triggers documented scene/design revision, never quiet filtering for better performance.
9. Verify the permanent-water product's values, year and nodata before using it. Do not assume any nonzero value denotes permanent water.

The reference precedes the optical image by approximately 16 hours 25 minutes. Flood movement during that interval can create apparent model errors. Treat reference agreement as imperfect evidence, not exact contemporaneous ground truth.

## 5. Systems compared

### A. No-water baseline

Predict land at every valid pixel. Report it as a diagnostic baseline illustrating why overall accuracy can conceal failure to detect water.

### B. MNDWI baseline

Compute `(green - SWIR1) / (green + SWIR1)` on the same reflectance grid. Exclude nonfinite values and near-zero denominators using a declared epsilon in reflectance units. Log excluded counts and use a shared evaluation mask across all systems.

On Candon only, evaluate thresholds from -0.50 to +0.50 inclusive in steps of 0.05, using `MNDWI > threshold`. Select maximum unrounded water IoU; ties choose the threshold closest to zero, then the numerically smaller threshold. Record every candidate and freeze the selected configuration before Vigan evaluation. Explain that threshold fitting is a form of model selection.

### C. Frozen Prithvi

Model: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`.
Candidate model revision verified in current source: `91ce9d38086a80b078a192b374df758b8855b732`.
Reuse verified architecture, preprocessing and audited safe-loading semantics, with immutable asset hashes. No remote custom code or unrestricted pickle fallback.

Use water softmax probability > 0.5 for the canonical prediction, with exact tie behaviour documented. Freeze all parameters. Do not describe inference as training.

Primary comparison: three systems on one shared clear/valid mask. Preserve results even if MNDWI wins.

## 6. Spatial processing and output meaning

- Process the complete selected areas with bounded 512 x 512 windows, batch size 1 by default. Never resize the entire geographic scene into one model chip.
- Preserve and validate georeferencing. Specify deterministic edge padding and crop predictions back to real pixels. Padded pixels never enter metrics.
- Use a fixed nonoverlapping window grid in version 1. Inspect seams as a limitation; do not count overlapping pixels multiple times.
- Report the nominal output grid and native band-resolution differences; resampling to 10 m does not make every band a 10 m measurement.
- Primary scientific endpoint: **clear-sky surface-water segmentation**.
- A secondary map may show predicted water outside the verified permanent-water mask. Name it **candidate inundation**, not confirmed flood extent. Seasonal water, irrigation, reference dates and permanent-water errors remain confounders.
- Existing permanent-water data alone are not an event-specific pre-flood observation. Do not claim a before/after change experiment unless a verified matched pre-event image is actually added.

## 7. Guided cell sequence

1. Orient: event, scientific question, intended use, limits, runtime and access requirements.
2. Preflight/bootstrap: isolated pinned environment, hardware/disk checks, anonymous downloads and integrity verification.
3. Data inspection: two AOIs, dates, bands, true/false-colour maps, label provenance and valid-pixel counts.
4. Predict: ask where water indices and Prithvi might disagree and why.
5. Declare protocol: Candon development, Vigan held-out; lock scene IDs, selection rule and scoring conventions.
6. Baselines: explain the no-water and MNDWI rules; select MNDWI threshold on Candon.
7. Frozen model: run tiled Prithvi inference; show Candon probabilities, masks and uncertainty without calling probabilities calibrated.
8. Controlled activity on Candon, then freeze the experiment record.
9. Held-out evaluation: run all canonical systems on Vigan once configurations are frozen.
10. Interpretation: metrics, disagreement map, error overlays and reference-time limitations.
11. Candidate inundation: cautiously distinguish persistent water from potentially inundated land, if permanent-water decoding passes validation.
12. Export, reload, integrity verification and scientific conclusion.

Each substantial code cell has explanatory markdown before it and a specific interpretation prompt after its output. Provide captions and legends; colour alone must not carry meaning.

## 8. Controlled learner experiment

Change only Prithvi's water decision threshold: 0.3, 0.5, 0.7, using cached Candon probabilities. Keep weights, imagery, masks and preprocessing fixed.

Ask learners to predict the precision/recall trade-off first. Show water area, precision, recall, F1, IoU and a paired disagreement map. Demonstrate that a threshold changes the decision, not the underlying model probabilities.

Canonical Vigan reporting remains at 0.5. The activity does not select a new test threshold. If participants later explore Vigan thresholds, label that work exploratory and do not reuse it as independent confirmation.

## 9. Metrics and evidence boundaries

- Aggregate TP/FP/FN/TN over unique valid pixels; derive water IoU, precision, recall, F1 and overall accuracy. Include land IoU and mean IoU as secondary metrics.
- Undefined ratios are null with a reason, not silently zero or one. No-water baseline precision is undefined when it predicts no water; recall/F1 are zero when reference water exists.
- Report valid, cloud, nodata and denominator-excluded counts and area fractions beside metrics. No conclusions about masked/cloudy regions.
- Export per-window metrics for spatial diagnosis, but do not treat pixels or adjacent windows as independent experimental replicates.
- This is one event with two geographic areas. No national, cross-event or operational generalisation claim and no naive pixel-level significance test.
- Example selection: fixed location overview, deterministic highest-disagreement window, and deterministic highest false-positive/false-negative windows with support counts. Label error-selected examples as such; allow absence of an error category.
- Compute area in the verified projected CRS from pixel dimensions. Label predicted water area and candidate inundation area separately; reference disagreement is not equivalent to disaster impact.
- Audit possible task-training dataset overlap, including event/scene identity, before calling the data out of distribution. Foundation-model pretraining overlap may remain unknown.

## 10. Artifacts and reproducibility

Export a bounded `results.zip` containing:

- `run_summary.json`: candidate status, exact configuration, hardware/software, timings, peak memory, counts and metrics.
- `data_manifest.json`: immutable data/model IDs, file hashes, scene roles, observation times and attribution.
- `preprocessing.json`: band order, units/scaling, mask mapping, grids, padding and exclusion counts.
- `selection.json`: MNDWI development sweep, selection rule and chosen threshold.
- `metrics.csv`, `per_window_metrics.csv`, `activity.csv`.
- Georeferenced water probability and prediction GeoTIFFs; prediction values 0 land / 1 water / 255 invalid, probability nodata explicitly declared.
- Candidate-inundation GeoTIFF only if its input layer passes validation, with an explicit meaning/limitations sidecar.
- Accessible comparison figures and `conclusion.md` scaffold.
- `checksums.json`, `verification.json` and relevant licence/attribution text.

Do not include base model weights or full source imagery in the results bundle by default. No trained adapter is produced in version 1.

Reopen exported rasters and verify dimensions, CRS, transform, nodata, masks and values. Recompute metrics from exported predictions against the canonical effective reference. Verify ZIP member hashes and CSV round trips. In a fresh process, reload the frozen model and reproduce a deterministic probe window within declared probability tolerance, with identical binary predictions or an explicit failure.

Cache identity must bind source, model, data, preprocessing and configuration. Changing upstream inputs invalidates downstream receipts; stale exports must not be presented as current.

## 11. Runtime and scope controls

Target: fresh Colab T4, batch size 1, float32 initial reference path. Measure actual runtime and peak RAM/VRAM before publishing estimates. Stream scene windows where practical and avoid retaining full model activations or all chip copies.

Embed required reference implementation at notebook generation time. Do not clone/download DIMER source or invoke workers during execution. Verify source hashes and generator parity. Download public data/weights through pinned manifests with bounded retries and resumable caches.

Use only the two selected optical scenes by default; expected imagery total is about 164 MB plus masks/provenance and separate model/runtime downloads. Establish hard byte limits from actual frozen manifests. No entire-WorldFloods download.

BYOD is deferred from version 1; link to the existing model tutorial's documented contract as an optional continuation. Do not add upload complexity to the capstone's core question.

Include the agreed AI Assistance Disclosure: generative AI assisted code/documentation under maintainer direction; the maintainer owns review, validation and release; assistance is not independent verification, endorsement or approval.

## 12. Acceptance and handoff

Required CPU checks: bands/units/scaling idempotence, mask mapping, denominator exclusions, duplicate-role rejection, spatial transforms, padding, threshold ties, metric edge cases, GeoTIFF/CSV/ZIP round trips, stale-cache invalidation, notebook schema/code compilation and generator parity. Synthetic tests are not model-performance evidence.

Required data checks: actual pixel inspection and counts for both areas, immutable hashes, input-level compatibility and attribution. Record any discrepancy from the metadata-derived estimates in this proposal.

Required hosted check: exact generated notebook, fresh T4 default Run all without manual repair; real maps, baseline/model comparisons, threshold activity, exports and fresh-process probe verification. Preserve notebook hash, execution outputs and `run_summary.json`.

Status stays Candidate until dataset feasibility, licensing review and hosted execution are complete. Do not inherit the older tutorial's release evidence. No performance improvement is an acceptance requirement; honest complete evidence is.

Build deliverables: notebook, generator, carried implementation, pinned sample manifest, relevant CPU tests, registry/README updates and a concise validation record. Work on a feature branch. This specification does not authorise commit, push, PR creation, merge or deployment.

## 13. Conclusion scaffold

On the clear, valid pixels of [area/date], frozen Prithvi achieved [IoU/F1/precision/recall] compared with [MNDWI and no-water baseline]. Changing the decision threshold on Candon produced [trade-off]. Inspection of [examples] suggests [bounded finding]. The observation-time gap, reference uncertainty, masked clouds and single-event design limit [claims]. We recommend [further testing / revision / stopping] and would next collect [specific evidence].

## Sources

- Notebook standard: https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md
- Host pipeline: https://github.com/kurtvalcorza/prithvi-flood-segmentation-pipeline
- Model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
- Dataset: https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2
- Dataset documentation: https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html
- Event: https://mapping.emergency.copernicus.eu/activations/EMSR312/
- Vigan metadata: https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2/blob/1f3faa2989e69930ac31d5c0a79fd224461123f8/train/meta/EMSR312_07VIGAN_DEL_MONIT01_v1.json
- Candon metadata: https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2/blob/1f3faa2989e69930ac31d5c0a79fd224461123f8/val/meta/EMSR312_08CANDON_DEL_MONIT01_v1.json
