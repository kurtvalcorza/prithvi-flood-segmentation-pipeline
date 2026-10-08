# Release verification

`tutorials/prithvi_flood_segmentation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the generator /2.2 kernel cell that builds the isolated
  uv environment from the hash lock (no in-kernel install, no restart); `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `PrithviFloodPipeline.from_pretrained(weights_dir=..., device=..., report=print)` so the pickle audit and the
  conversion are printed before the model loads, `fetch_sample_dataset` from the pinned cache path,
  `load_byod_dataset`, `dataset_manifest`, `write_sample_pair`, `validate_dataset` with the refusal probes,
  `pipe.evaluate` on the frozen model and after adaptation with the procedural assertions, `pipe.predict`,
  `pipe.adapt` with its explicit hyperparameters, `pipe.save_artifact`, `PrithviFloodPipeline.from_artifact` and
  the reload-parity assertion, and the provenance fields `served_from_pickle: False`, `remote_code_executed: False`
  and the data base URL), the seven expected `outputs/` paths, the learner-facing statements (the asset is a pickle
  unpickled once, the model already trained on the dataset, the no-water baseline, uncalibrated scores, frozen
  BatchNorm, split by scene or event, CC BY 4.0) and the gated-off BYOD default; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` / `terratorch` / `Unpickler` use or
  `torch.load(` / `pickle.load` **outside the carried module cells**, `trust_remote_code=True`, `pickle.load` or
  `torch.load(` without `weights_only=True` anywhere, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub model, skipped without torch),
`tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; crafted pickles,
temporary manifests, synthetic chips and an injected fetcher, no weights and no `terratorch`). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/prithvi-eo-2.0-300m-tl-sen1floods11/` or the data cache `weights/sen1floods11/` (the standalone
   path writes the manifest itself, stages all seven listed files from the Hub, audits and converts the checkpoint,
   and fetches the 88 pinned Sen1Floods11 objects, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `EPOCHS = 4`, `LEARNING_RATE = 1e-5`, `BATCH_SIZE = 2`, `TRAINABLE = 'decoder'`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `terratorch==1.2.13`, `tifffile==2026.9.15`,
   `numpy==2.5.3`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`. Since generator /2.2 the pins come from the
   hash lock `tutorials/requirements-colab.lock.txt` and are installed into an isolated uv environment, never into
   the kernel: `Run all` must complete in **one pass with no restart** (RUN1, RUN10, ENV6); a run that needed a
   restart is not release evidence (review FL-M1);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `PrithviFloodPipeline`, `audit_pickle`, `convert_model`,
     `build_model`, `verify_snapshot`, `verify_converted`, `stage_missing_files`, `validate_inputs`,
     `validate_dataset`, `read_chip`, `read_mask`, `fetch_object`, `fetch_sample_dataset`, `load_byod_dataset`,
     `write_sample_pair`, `write_dataset_csv`, `dataset_manifest`, `segmentation_metrics`, `majority_baseline`)
     with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 7 entries fetched from `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` at the immutable
     revision and `verify_snapshot` reporting 7 verified files;
   - the model cell printing the **conversion record** with the static audit (globals `collections.OrderedDict` /
     `torch.FloatStorage` / `torch.LongStorage` / `torch._utils._rebuild_tensor_v2`, 0 violations, audit digest
     `5b9f0ba0…`), the checkpoint record (epoch 41, global step 630, Lightning 2.4.0) and the converted file
     (`65e4377f…`, 1,276,749,320 bytes), then the load report on `cuda` with source "converted from the
     manifest-verified source checkpoint";
   - the dataset manifest with 24 / 8 / 12 chips, water fractions about 0.21 / 0.36 / 0.20, the written sample pair
     and `outputs/prithvi_flood_segmentation_sample_pairs.csv`, and three refusals (five-band chip, unknown label
     class, reflectance out of range);
   - the no-water baseline and the frozen model on the test chips (on the sample: baseline accuracy ≈ 0.80, frozen
     water IoU ≈ 0.73, F1 ≈ 0.84, precision above recall) and the validation chips (water IoU ≈ 0.86);
   - `pipe.adapt` printing epoch 0 as the frozen model, 15,082,242 trainable of 318,968,580 parameters, 48 steps,
     frozen BatchNorm statistics, and a four-epoch history with validation loss ≈ 0.13 → ≈ 0.12 at the kept epoch;
   - `pipe.evaluate` on the test chips with the three-way comparison and
     `outputs/prithvi_flood_segmentation_evaluation_report.json` written (the cell asserts the kept epoch's validation
     loss is no higher than the frozen model's and that the validation metrics match the history);
   - the three upstream example chips segmented with `outputs/prithvi_flood_segmentation_predictions.json` and one
     mask GeoTIFF per chip written;
   - `pipe.save_artifact` writing `outputs/prithvi_flood_segmentation_adapter/{adapter.safetensors,manifest.json}`
     (46 tensors, about 60 MB), and `PrithviFloodPipeline.from_artifact` reloading it with identical held-out
     metrics and score maps (the cell asserts both);
   - `outputs/prithvi_flood_segmentation_result.json` written with `NOTEBOOK_SOURCE`, the model identity, the
     provenance block (`served_from_pickle: false`, `remote_code_executed: false`, the audit digest, the converted
     digest, the data base URL), the runtime versions, the comparison and the reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the data cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `prithvi_flood_segmentation_colab.ipynb` (`E2E`) | `eb79368` / `e04ef87b` | 2026-10-08 | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 VM (session `suite-prithvi-eb79368-14ae`); kernel Python 3.13.15, isolated uv environment Python 3.12.12 with the 131-package hash lock; not a browser `Run all`, no execution counts (order from `exec.log`) | **PASSED** — one pass, no restart, 0 errors, 11/11 code cells, 287.2 s (session wall, includes the 91 s isolated-environment build); 24 / 8 / 12 chips from the official splits, the three refusal probes each rejected for its own rule; test water IoU 0 (no-water baseline, accuracy 0.8049) / frozen 0.7301 (F1 0.8440, precision 0.9251, recall 0.776, accuracy 0.9440) → adapted 0.7458 (F1 0.8544, precision 0.9165, recall 0.8002, accuracy 0.9468; verdict *improved*, delta 0.0157); validation loss 0.1306 → 0.1178, kept epoch 2 (validation water IoU 0.8614 → 0.8748); Section 7 figure rendered; Section 8 prints `also_a_sample_chip: 'test'` for `USA_430764` only; reload parity positive_iou_diff 0.0, metrics_identical True, max_abs_score_diff 0.0; evidence in `docs/execution-evidence/2026-10-08-eb79368/` (executed notebook SHA-256 `a5a4443519700e881285495e47d0da240ba7671484019df166b1f18f339920de`, `exec.log` `255c8ff94e1518230961480614eed95767b98779d3042c200d8f24953b692f1a`, `run_summary.json` `f4c2019768af6a88d422b39a1d6274a992c55bb41cc20abb5cb76a85d07140be`) |
| `prithvi_flood_segmentation_colab.ipynb` (`E2E`) | `b6240ae` / `a61580e4` | 2026-09-25 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-flood-segmentation` v3; image `torch 2.10.0+cu128` before the pinned install, pinned `torch 2.14.0+cu130` / `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | **PASSED** — 10/10 code cells ok (1 restart after install cell); 106 files, 2664 MB fetched into a clean runtime; comparison test water IoU / F1 (no-water baseline 0 / 0): frozen 0.7301 / 0.8440 → adapted 0.7458 / 0.8544 (best epoch 2, validation loss 0.1306 → 0.1178, validation water IoU 0.8614 → 0.8748), accuracy 0.9440 → 0.9468 vs baseline 0.8049; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; the first clean run after the idempotent-scaling fix; run summary and executed notebook archived under `.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-flood-segmentation/v3/evidence/` in the workspace |
| `prithvi_flood_segmentation_colab.ipynb` (`E2E`) | `d43975f` / `6d648d70` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-flood-segmentation` v2; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `timm 1.0.26`, `lightning 2.6.6`, `tifffile 2026.9.15`, `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | **PASSED** — 10/10 code cells ok (1 restart after install cell); 106 files, 2664 MB fetched into a clean runtime (Hub snapshot + the pinned data); comparison test water IoU / F1 (no-water baseline 0 / 0): frozen 0.5857 / 0.7387 → adapted 0.6005 / 0.7504 (best epoch 2, validation loss 0.1306 → 0.1178), accuracy 0.8916 → 0.8952 vs baseline 0.8049; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-prithvi-flood-segmentation/v2/evidence/` in the workspace |
| `prithvi_flood_segmentation_colab.ipynb` | generated, pre-commit | 2026-09-19 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/prithvi_flood_segmentation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/prithvi_flood_segmentation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-08 | `eb79368` / `e04ef87b` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 VM (session `suite-prithvi-eb79368-14ae`); kernel Python 3.13.15, isolated uv environment Python 3.12.12 with the 131-package hash lock; not a browser `Run all`, no execution counts (order from `exec.log`) | Default sample path only (blob SHA-1 verified against GitHub before execution; the optional BYOD path, the upload dialog and the closing activity not exercised) | 287.2 s | **PASSED** — one pass, no restart, 0 errors, 11/11 code cells, 287.2 s (session wall, includes the 91 s isolated-environment build); 24 / 8 / 12 chips from the official splits, the three refusal probes each rejected for its own rule; test water IoU 0 (no-water baseline, accuracy 0.8049) / frozen 0.7301 (F1 0.8440, precision 0.9251, recall 0.776, accuracy 0.9440) → adapted 0.7458 (F1 0.8544, precision 0.9165, recall 0.8002, accuracy 0.9468; verdict *improved*, delta 0.0157); validation loss 0.1306 → 0.1178, kept epoch 2 (validation water IoU 0.8614 → 0.8748); Section 7 figure rendered; Section 8 prints `also_a_sample_chip: 'test'` for `USA_430764` only; reload parity positive_iou_diff 0.0, metrics_identical True, max_abs_score_diff 0.0 |
| 2026-09-25 | `b6240ae` / `a61580e4` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-flood-segmentation` v3; image `torch 2.10.0+cu128` before the pinned install, pinned `torch 2.14.0+cu130` / `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 392.9 s | **PASSED** — 10/10 code cells ok (1 restart after install cell); 106 files, 2664 MB fetched into a clean runtime; comparison test water IoU / F1 (no-water baseline 0 / 0): frozen 0.7301 / 0.8440 → adapted 0.7458 / 0.8544 (best epoch 2, validation loss 0.1306 → 0.1178, validation water IoU 0.8614 → 0.8748), accuracy 0.9440 → 0.9468 vs baseline 0.8049; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; the first clean run after the idempotent-scaling fix; run summary and executed notebook archived under `.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-flood-segmentation/v3/evidence/` in the workspace |
| 2026-09-19 | `d43975f` / `6d648d70` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-flood-segmentation` v2; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `timm 1.0.26`, `lightning 2.6.6`, `tifffile 2026.9.15`, `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the checkpoint audited and converted in the notebook, the pinned data fetched by the notebook | 466.5 s | **PASSED** — 10/10 code cells ok (1 restart after install cell); 106 files, 2664 MB fetched into a clean runtime (Hub snapshot + the pinned data); comparison test water IoU / F1 (no-water baseline 0 / 0): frozen 0.5857 / 0.7387 → adapted 0.6005 / 0.7504 (best epoch 2, validation loss 0.1306 → 0.1178), accuracy 0.8916 → 0.8952 vs baseline 0.8049; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-prithvi-flood-segmentation/v2/evidence/` in the workspace |
| 2026-09-19 | generated, pre-commit | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti, `terratorch 1.2.13`) | Default sample path (stage → verify → load the already-converted file → pinned-data assembly from the cache → validate → refusal probes → baseline + frozen evaluation → decoder adapt → evaluate → example-chip inference → export → reload); the Hub files, the converted safetensors and the 88 data objects were pre-staged, so `stage_missing_files` fetched 0 of 7 entries and every data object was served from the cache after its digest check | 60.1 s | **PASSED** — 10/10 code cells; probes refused; test water IoU 0.5857 (frozen) → 0.6003 (adapted, best epoch 2), validation 0.8614 → 0.8750; adapter 46 tensors; reload parity identical. Pre-flight; hosted clean-runtime run still required |
| 2026-09-25 | branch `fix/idempotent-reflectance-scaling`, uncommitted | Local CPU pre-flight (Windows, CPython 3.12.10, `torch 2.14.0+cpu`, `terratorch 1.2.13`, fp32) | `samples.fetch_sample_dataset` from the digest-verified cache → frozen `evaluate` on test and validation → default `adapt` (4 epochs, lr 10⁻⁵, batch 2, `decoder`) → `evaluate`; the same run on unfixed `main` as a control | 333.1 s (adapt) | **PASSED** — fixed: test water IoU 0.7301 (frozen) → 0.7459 (adapted, best epoch 2), validation 0.8614 → 0.8747; control on unfixed code: 0.5857 → 0.5990. Pre-flight on CPU; **not** promotion evidence; the notebook was not executed |

**Correction, 2026-09-25.** The test metrics in the 2026-09-19 rows above were produced by code with a scaling defect: `_check_record` rescaled checked chips whose reflectance maximum exceeded 1.0, so test chip `Paraguay_868895` and training chip `Spain_2938657` reached the model at ≈ 10⁻⁴ of their reflectance. The rows remain the record of what those runs executed and are not rewritten. With the fix, the frozen model's test water IoU / F1 is 0.7301 / 0.8440 (accuracy 0.9440, no-water baseline 0.8049) and the adapted model 0.7458 / 0.8544, confirmed by the clean-runtime Kaggle T4 run of the fixed blob `a61580e4` recorded above (a CPU pre-flight had given 0.7459 / 0.8545); see `MODEL_CARD.md` §Runtime.

## Current status

**Candidate.** The tutorial was regenerated (generator `build_notebook.py/2.2`: isolated uv environment from a hash lock, no in-kernel install; the 2026-10-05 sweep fixes; the 2026-10-02 review fixes FL-M1..M5 / FL-m1..m7; the `google.colab` stub spec fix), and is now blob `e04ef87b` (committed at `eb79368`). That exact blob passed a Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 on 2026-10-08 (default path only) in one pass, no restart, 0 errors, 11/11 code cells, 287.2 s, recorded above with byte-exact evidence under `docs/execution-evidence/2026-10-08-eb79368/`; every worked answer the notebook quotes (0.7301 → 0.7458, 0.8049, precision 0.9251 → 0.9165, recall 0.776 → 0.8002, 0.1306 → 0.1178 at epoch 2, 0.8614 → 0.8748, 10 test regions) matches it (REL13). It is not a browser `Run all`, and the optional BYOD path (REL12), the upload dialog and the closing activity were not exercised, so the registry stays **Candidate** pending review for promotion (REL14). History: the blob `a61580e4` (committed at `b6240ae`, the idempotent-scaling fix) executed in a clean Kaggle Tesla T4 runtime on 2026-09-25 (10/10 ok only after 1 restart after the install cell, 392.9 s, 106 files, 2664 MB fetched and digest-verified inside the notebook, the checkpoint converted in the notebook) with no repository checkout and was then marked Release-grade; a two-pass run does not meet RUN1/RUN10/ENV6 (review FL-M1), so that record covers only that blob. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

## Philippines flood-mapping capstone (`WORKSHOP`, Notebook Spec 2.2)

`tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb` (`TASK-INFERENCE`, `WORKSHOP`, standalone) is a **Candidate**, recorded separately from the `E2E` tutorial above. That tutorial's release evidence does not qualify it.

| Requirement | Evidence | Status |
|---|---|---|
| Carried files match `CARRIED_HASHES`; carried `prithvi_reference/`, manifest, licence and weight provenance equal this repository's; `source.json` agrees with the metadata; the data manifest pins one development and one held-out scene by dataset revision and SHA-256; every in-text citation resolves to a reference entry and every entry is cited | `tools/validate_release_assets.py`, `tests/test_capstone_notebook.py` | automatic, every pull request |
| CPU checks from the capstone specification §12: label mapping, reflectance conversion and bounds, MNDWI denominator exclusion, threshold tie rule, undefined-metric reporting, window tiling and padding, duplicate-role rejection, reload parity, CSV and GeoTIFF round trips, stale-receipt invalidation | `tests/test_capstone_notebook.py` (synthetic arrays; not model-performance evidence) | automatic, every pull request |
| 2026-09-28 review fixes: answer keys do not promise monotonic precision; rising and falling precision fixtures; activity leaves scores, masks and selection unchanged; diagnostic figures in a fixed order; error windows read the optical image; candidate-inundation classes, sidecar, figure and display; nested reload details displayed | `tests/test_capstone_notebook.py` (synthetic arrays and the notebook's own display helpers; the three figure-rendering tests skip without matplotlib, which CI does not install) | automatic, every pull request |
| Data feasibility gate (§4) on the actual rasters | CPU pre-flight, 2026-09-27 (below) | passed |
| Fresh Colab T4 `Run all` of the committed blob | Colab T4 run, 2026-09-27 (first row below) | **recorded 2026-09-27**: blob `9ca6d8d02110` passed; stage times recorded; peak GPU memory and disk were not transcribed from this run. **recorded 2026-09-28 for blob `b7f586e3cea0`** (2026-09-28 review fixes, below): passed on a fresh Colab T4; the executed file is kept in `docs/execution-evidence/2026-09-28/` |
| Dataset licensing and use review (WorldFloods v2 is CC BY-NC 4.0) | — | maintainer review pending |
| Task-training overlap audit (Sen1Floods11 versus EMSR312 scenes) | — | pending |

**Data feasibility gate, measured on the pinned rasters (2026-09-27).** Both 15-band Sentinel-2 L1C scenes are on the EPSG:32651 10 m grid, with masks aligned exactly and footprints that do not overlap. The files match their pinned sizes and SHA-256 values. Reflectance on valid pixels (×1e-4) spans 0.0028–1.2823 on Candon and 0.0027–1.5388 on Vigan.

| Scene (role) | Size | Clear valid | Water px | Land px | Cloud px | Reference missing px | Optical invalid px | 512 px windows ≥ 80% valid |
|---|---|---|---|---|---|---|---|---|
| Candon `EMSR312_08CANDON_DEL_MONIT01_v1` (development) | 2,969 × 2,434 | 90.3% | 1,580,828 | 4,948,243 | 510,862 | 536,743 | 186,613 | 16 |
| Vigan `EMSR312_07VIGAN_DEL_MONIT01_v1` (held-out) | 2,843 × 2,331 | 97.1% | 1,485,700 | 4,946,309 | 21,420 | 183,395 | 173,604 | 20 |

The measured cloud fractions (7.1% of Candon, 0.3% of Vigan) agree with the specification's metadata-derived estimates.

**Defects found in review and fixed before the first hosted run:**

1. **Locked install.** `--only-binary :all:` could not install the lock, because `antlr4-python3-runtime==4.9.3` (required by `hydra-core` and `omegaconf`) is published on PyPI only as a source archive. The install cell now works in three steps: install the locked `setuptools` wheel, build `antlr4` from its hash-pinned archive with `--no-build-isolation`, then install the rest of the lock from wheels only. Every artefact is still hash-verified, and no unpinned build dependency is fetched.
2. **Plotting backend.** Colab exports `MPLBACKEND=module://matplotlib_inline.backend_inline`, which the isolated environment cannot import. The sound-event workshop's first Colab run failed on this. The install cell now sets `MPLBACKEND=Agg` for every stage.
3. **Reflectance record.** The reflectance range in `prepare.json` included masked pixels that had been zero-filled. It is now recorded over valid pixels only, both before and after the 1e-4 conversion.
4. **Metadata keys.** The metadata keys were renamed to `notebook_profile` / `notebook_mode`.

**Review fixes, 2026-09-28 (blob `b7f586e3cea0`).** A notebook review of blob `9ca6d8d02110` at `edbfda9` (Notebook Review Framework v1) found one major and four minor findings; none was a default-path failure or a defect in the recorded metrics. The capstone generator is not in this repository, so the notebook was patched in place with anchor-checked replacements; the carried hashes, `source.json` and `generated_from` were recomputed, and each change is listed in `generated_from.post_generation_revisions`.

1. **F01 (major): threshold answer keys.** Sections 3 and 5 stated that precision rises when the threshold rises. On a fixed mask, a higher threshold cannot increase predicted water area or recall, but precision may rise, fall or stay unchanged. Both answers now say so, and label the Candon rise recorded below as an observation about this scene and model, not a rule. The prediction prompts stay open-ended.
2. **F02: error windows.** Each error-window figure now shows, for the same window, a locator box on the whole area, true- and false-colour crops, the reference, the MNDWI and Prithvi decisions, and both systems' error maps. Its title states that the imagery suggests explanations to test but does not establish a cause. The deterministic window-selection policy is unchanged.
3. **F03: figure order.** The diagnostic figures were shown in filename order, which put the error windows before the whole-area score maps the text described as "the first map". `show_diagnostics` now shows them in a fixed order under captions, and reports an absent error category instead of failing.
4. **F04: candidate inundation and reload details.** The report stage now also writes a categorical candidate-inundation figure per area, and the sidecar records scene and role. After the report, the notebook shows an area table, the meaning and limitations, and both maps. The verification display now includes the nested reload record (largest score difference, decision parity, tolerances).
5. **F05: evidence-boundary wording.** Section 2 now declares that both areas' references are inspected for feasibility; what stays protected is Vigan *performance*. Section 5's answer says that Vigan performance has not been used to select any threshold, not that Vigan "has not been touched". An interior optimum is described as "not at a boundary" of the finite candidate set, not as proof that the range was wide enough.

The review's BYOD applicability item is recorded as a scope disposition in the capstone specification §11: it is proposed for maintainer confirmation.

User-visible changes: the error-window figures change from three panels to eight; the results bundle gains `development_candidate_inundation.png` and `heldout_candidate_inundation.png`; each `*_candidate_inundation.json` sidecar gains `scene` and `role`. Metric computation, thresholds, selection, rasters, CSVs and receipts are unchanged.

Offline verification, 2026-09-28 (CPU, Python 3.12, not clean-runtime evidence): the validator, `tools/build_notebook.py --check` and `ruff check src tests tools` pass. `pytest` passes 62 with 4 skipped under CI's dependencies (torch and three matplotlib tests skipped) and 65 with 1 skipped with matplotlib installed; six of the new tests fail on blob `9ca6d8d02110`. A harness ran all seven carried stages in order on two synthetic 1100 × 1150 15-band scenes, with a stand-in probability function in place of Prithvi (no torch, no weights). Every stage and receipt completed, the bundle carried the new figures and sidecars, and the rendered figures were inspected. It exercised both the absent-category path and the full five-figure path. This is plumbing and layout evidence, not model or hosted-runtime evidence. A fresh Colab T4 `Run all` of blob `b7f586e3cea0` followed on the same day (section below).

**Maintainer-supplied Colab execution of blob `b7f586e3cea0` — 2026-09-28.**

- **File:** `docs/execution-evidence/2026-09-28/DIMER_Philippines_Flood_Mapping_Capstone_b7f586e3cea0_colab-t4.ipynb`, byte-identical to the upload (SHA-256 `c9cea09c656a1f7333142f697fca86162159205a4ffd46caa26f79e27dbfd967`, 22,445,813 bytes).
- **Source match:** all 26 cell sources and types equal blob `b7f586e3cea0` exactly; no form or title edits.
- **Runtime:** Google Colab, **Tesla T4** (reported by the runtime check; notebook metadata `gpuType: T4`). The isolated managed Python 3.12.12 environment was installed by the notebook's locked install. Bootstrap time, peak GPU memory and disk were not transcribed.
- **Execution:** default `Run all`: 9/9 code cells, execution counts 1–9 in order, no errors, 15 figures (2 overviews, 5 diagnostics per area, 1 activity, 2 candidate-inundation maps). Stage time 192.9 s: prepare 21.7, baseline 3.4, Candon inference 94.3, activity 1.6, Vigan inference 40.3, reload 21.1, report 10.4.
- **Results:** identical to the 2026-09-27 run of `9ca6d8d02110` to the reported precision. Selected MNDWI threshold 0.15 (Candon IoU 0.9259). **Candon** water IoU / precision / recall: MNDWI 0.9259 / 0.9814 / 0.9425; Prithvi 0.9377 / 0.9936 / 0.9433. **Vigan**: MNDWI **0.7522** / 0.9802 / 0.7638; Prithvi **0.6405** / 0.9766 / 0.6505. The no-water rule scores 0 IoU with undefined precision on both. Activity at 0.3 / 0.5 / 0.7: IoU 0.9461 / 0.9377 / 0.9245, precision 0.9858 / 0.9936 / 0.9972, recall 0.9592 / 0.9433 / 0.9269. Candidate inundation 1.6871 km² (Candon) and 14.6965 km² (Vigan). Reload: max absolute error 0, identical decisions, fresh process.
- **Evidence boundary:** the saved outputs were inspected, including the figures; execution was not independently repeated. This run does not decide licensing, training overlap or the BYOD disposition.

| Journey / finding | Verdict in this run |
|---|---|
| Default `Run all` on a fresh T4 | **passed** |
| F01 answer keys | text present as committed; the observed Candon precision rose with the threshold, matching the answer's labelled observation |
| F02 error windows | **passed**: all 10 windows render the locator box, true- and false-colour crops, reference, both decisions and both error maps; they are legible at full scene scale |
| F03 figure order | **passed**: whole-area figure first, then fixed location, most errors, most false positives and most false negatives, each under its caption, in both areas |
| F04 candidate inundation and reload display | **passed**: area table, meaning and limitations, both maps and the nested reload rows are shown after the report |
| F05 wording | text present as committed |
| Optional journeys | none exist in this notebook; not applicable |

**Observations, not fixed here.**

1. **JRC class 0 dominates land.** The new candidate-inundation figure shows that the permanent-water raster is class 0 ("no data", excluded) across almost all land in both areas. Class 1 ("not water") appears only near rivers and the coast. The provider exports JRC GSW `YearlyHistory` from Earth Engine (`ml4floods` `download_permanent_water`). If never-water land exports as 0 (unverified here), the capstone's exclusion of class 0 removes most dry land, so candidate inundation is computed only within the historical water extent. This behaviour predates this change (the areas equal the 2026-09-27 pre-flight). It needs a data check and a maintainer decision on the class-0 rule before the secondary product is interpreted.
2. **DeprecationWarning noise.** Each stage echoes NumPy 2.5 `DeprecationWarning`s from `rasterio` reads (132 lines in this run), which clutters learner output. It is cosmetic, comes from the locked `numpy`/`rasterio` pair and does not affect results.

**Notebook source layout change (2026-10-02).** The `CARRIED_FILES` literal in the infrastructure cell was one line of 421,086 characters. `tools/split_capstone_carrier.py` now writes it as parenthesised runs of short string pieces (one per carried source line, cut every 1,000 characters), so no cell line exceeds 2,000 characters; the longest is now 1,219. Python joins the pieces back into the same text: the 11 carried files, `CARRIED_HASHES`, the carried `source.json` and `generated_from.files` are unchanged, and only one `post_generation_revisions` entry was added. The notebook blob changes from `b7f586e3cea0` to `4ace5596ae3c`. The hosted runs recorded above are of blob `b7f586e3cea0` (and earlier); the new blob was re-run on 2026-10-03 through the Colab CLI (first row of the table below). Status stays Candidate.

Remaining before promotion: the dataset licensing and use review, the task-training overlap audit, confirmation of the BYOD applicability disposition, and a decision on observation 1.

| Date (UTC) | Notebook source | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-03 | `298c1ea` / blob `4ace5596ae3c` (downloaded from GitHub at the commit and blob-verified before the session; all 26 cells equal the PR head; executed file `docs/execution-evidence/2026-10-03/DIMER_Philippines_Flood_Mapping_Capstone_298c1ea_colab-cli-t4.ipynb`, sha256 `388b9cbae5c6…`) | Google Colab CLI 0.7.4 on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); cells run in order in one kernel, not a browser Run all; order evidenced by the CLI's `Executing cell k/9` log (no execution counts recorded) | Default path, no edits | 302.4 s wall | **PASSED**: 9/9 code cells, no errors. Every printed line equals the 2026-09-28 run of blob `b7f586e3cea0` except the per-stage `seconds` values; product counts and metrics are identical. Saved outputs inspected |
| 2026-09-28 | blob `b7f586e3cea0` (review fixes; the executed file's 26 cells equal this blob; sha256 `c9cea09c656a…`, kept in `docs/execution-evidence/2026-09-28/`) | Google Colab, fresh **Tesla T4** runtime; isolated managed Python 3.12.12 environment from the notebook's locked install | Default `Run all`, no restart and no edits | 192.9 s of stage time; bootstrap not transcribed | **PASSED**: 9/9 code cells, execution counts 1–9, no errors, 15 figures; metrics identical to the 2026-09-27 run; the new diagnostic, candidate-inundation and reload displays render as designed (section above) |
| 2026-09-27 | `3039c4a` / blob `9ca6d8d02110` (the executed file's 26 cells equal this blob; uploaded file sha256 `db297e42f537…`) | Google Colab, fresh **Tesla T4** runtime (reported by the runtime check); isolated managed Python 3.12.12 environment installed by the notebook's three-step locked install | Default `Run all`, no restart and no edits: every stage in its own process, 1.28 GB checkpoint fetched, audited and converted, both scenes fetched and verified | 228 s of stage time (prepare 37 s, baseline 4 s, Candon inference 119 s including the model download and conversion, activity 2 s, Vigan inference 36 s, reload 22 s, report 8 s); bootstrap time not transcribed | **PASSED** — 9/9 code cells, execution counts 1–9, no errors, 13 figures. The feasibility counts are identical to the pre-flight. Selected MNDWI threshold 0.15 (IoU 0.9259). **Candon (development)** water IoU / precision / recall / accuracy: no-water 0 / undefined / 0 / 0.758; MNDWI 0.926 / 0.981 / 0.943 / 0.982; Prithvi 0.938 / 0.994 / 0.943 / 0.985 (150.08 km² predicted water). Threshold activity: IoU 0.946 / 0.938 / 0.925 at 0.3 / 0.5 / 0.7. **Vigan (held-out)**: no-water 0 / undefined / 0 / 0.769; **MNDWI 0.752** / 0.980 / 0.764 / 0.942; **Prithvi 0.641** / 0.977 / 0.651 / 0.916 — the frozen index beats the frozen model on the held-out area, as in the CPU pre-flight (T4 and CPU agree to within 0.0001 IoU). Verification record: grids, masks, metrics and CSV round trips passed; fresh-process reload completed. Each stage's completion line is printed twice (the live stream, then the stage-log tail), which is cosmetic |
| 2026-09-27 | This branch, blob `9ca6d8d02110` (sha256 `b890af31a81d…`) | Builder pre-flight in a real Jupyter kernel (`nbclient` + `ipykernel`), Linux container, 4-core CPU, no GPU. `MPLBACKEND=module://matplotlib_inline.backend_inline` was exported in the kernel, as on Colab. The notebook's own cells downloaded `uv` 0.12.15, created the managed Python 3.12.12 environment and installed the hash-locked set (`torch 2.11.0+cu130`, `terratorch 1.2.13`, `rasterio 1.4.4`); the package cache was warm from the first pre-flight. Two harness-only patches were used: a stub `nvidia-smi` for the runtime check, and CPU execution of the model stages (the device check, `device="cuda"` and the CUDA memory counters in the runner); the install check dropped its `torch.cuda.is_available()` assertion | Default `Run all`, all seven stages, each in a fresh process. The 1.28 GB checkpoint was fetched from the Hub and audited, then converted with `weights_only=True`, and the model loaded from safetensors | 248 s kernel wall (prepare 20 s, baseline 3 s, Candon inference 118 s, activity 2 s, Vigan inference 70 s, reload 14 s, report 6 s) | PASS — pre-flight only, **not** promotion evidence. 10/10 executed code cells, execution counts 1–10, no errors, 13 figures. Selected MNDWI threshold on Candon: 0.15 (water IoU 0.9259; 21 candidates from −0.50 to +0.50, winner inside the range). **Candon (development)** water IoU / precision / recall / accuracy: no-water 0 / undefined / 0 / 0.758; MNDWI 0.926 / 0.981 / 0.943 / 0.982; Prithvi 0.938 / 0.994 / 0.943 / 0.985. Threshold activity on Candon: IoU 0.946 at 0.3, 0.938 at 0.5, 0.925 at 0.7, with precision rising and recall falling. **Vigan (held-out)**: no-water 0 / undefined / 0 / 0.769; **MNDWI 0.752** / 0.980 / 0.764 / 0.942; **Prithvi 0.641** / 0.977 / 0.651 / 0.916 — the frozen index beats the frozen model on the held-out area. Candidate inundation outside JRC 2018 permanent water: 1.69 km² (Candon), 14.70 km² (Vigan). Fresh-process reload parity: exact (max absolute error 0.0, identical decisions). Exported rasters, CSVs and ZIP member hashes all verified; `results.zip` is 94.6 MB with 50 members. Metrics are identical to the first pre-flight. CPU float32 figures may differ slightly from a T4 run |
| 2026-09-27 | Uploaded notebook (sha256 `7f9047ddf3a9…`), stages run directly from its carried files | Builder pre-flight, same container, CPU. The notebook's original install command failed on the `antlr4` source-only pin (defect 1), so the environment was installed with the three-step procedure; stages ran from a shell with `MPLBACKEND=Agg` and the CPU patch | All seven stages | 233 s of stage time | PASS after the install workaround — pre-flight only. Same metrics as the row above; exposed defects 1 and 3 |
