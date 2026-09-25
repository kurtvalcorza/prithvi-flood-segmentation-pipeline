# Release verification

`tutorials/prithvi_flood_segmentation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
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
   `numpy==2.5.3`, `safetensors==0.8.0`, `huggingface-hub==1.32.0` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
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
     water IoU ≈ 0.59, F1 ≈ 0.74) and the validation chips (water IoU ≈ 0.86);
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
| 2026-09-25 | `b6240ae` / `a61580e4` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-flood-segmentation` v3; image `torch 2.10.0+cu128` before the pinned install, pinned `torch 2.14.0+cu130` / `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 392.9 s | **PASSED** — 10/10 code cells ok (1 restart after install cell); 106 files, 2664 MB fetched into a clean runtime; comparison test water IoU / F1 (no-water baseline 0 / 0): frozen 0.7301 / 0.8440 → adapted 0.7458 / 0.8544 (best epoch 2, validation loss 0.1306 → 0.1178, validation water IoU 0.8614 → 0.8748), accuracy 0.9440 → 0.9468 vs baseline 0.8049; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; the first clean run after the idempotent-scaling fix; run summary and executed notebook archived under `.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-flood-segmentation/v3/evidence/` in the workspace |
| 2026-09-19 | `d43975f` / `6d648d70` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-flood-segmentation` v2; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `timm 1.0.26`, `lightning 2.6.6`, `tifffile 2026.9.15`, `terratorch 1.2.13` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the checkpoint audited and converted in the notebook, the pinned data fetched by the notebook | 466.5 s | **PASSED** — 10/10 code cells ok (1 restart after install cell); 106 files, 2664 MB fetched into a clean runtime (Hub snapshot + the pinned data); comparison test water IoU / F1 (no-water baseline 0 / 0): frozen 0.5857 / 0.7387 → adapted 0.6005 / 0.7504 (best epoch 2, validation loss 0.1306 → 0.1178), accuracy 0.8916 → 0.8952 vs baseline 0.8049; reload parity positive_iou_diff: 0.0, metrics_identical: True, max_abs_score_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-prithvi-flood-segmentation/v2/evidence/` in the workspace |
| 2026-09-19 | generated, pre-commit | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti, `terratorch 1.2.13`) | Default sample path (stage → verify → load the already-converted file → pinned-data assembly from the cache → validate → refusal probes → baseline + frozen evaluation → decoder adapt → evaluate → example-chip inference → export → reload); the Hub files, the converted safetensors and the 88 data objects were pre-staged, so `stage_missing_files` fetched 0 of 7 entries and every data object was served from the cache after its digest check | 60.1 s | **PASSED** — 10/10 code cells; probes refused; test water IoU 0.5857 (frozen) → 0.6003 (adapted, best epoch 2), validation 0.8614 → 0.8750; adapter 46 tensors; reload parity identical. Pre-flight; hosted clean-runtime run still required |
| 2026-09-25 | branch `fix/idempotent-reflectance-scaling`, uncommitted | Local CPU pre-flight (Windows, CPython 3.12.10, `torch 2.14.0+cpu`, `terratorch 1.2.13`, fp32) | `samples.fetch_sample_dataset` from the digest-verified cache → frozen `evaluate` on test and validation → default `adapt` (4 epochs, lr 10⁻⁵, batch 2, `decoder`) → `evaluate`; the same run on unfixed `main` as a control | 333.1 s (adapt) | **PASSED** — fixed: test water IoU 0.7301 (frozen) → 0.7459 (adapted, best epoch 2), validation 0.8614 → 0.8747; control on unfixed code: 0.5857 → 0.5990. Pre-flight on CPU; **not** promotion evidence; the notebook was not executed |

**Correction, 2026-09-25.** The test metrics in the 2026-09-19 rows above were produced by code with a scaling defect: `_check_record` rescaled checked chips whose reflectance maximum exceeded 1.0, so test chip `Paraguay_868895` and training chip `Spain_2938657` reached the model at ≈ 10⁻⁴ of their reflectance. The rows remain the record of what those runs executed and are not rewritten. With the fix, the frozen model's test water IoU / F1 is 0.7301 / 0.8440 (accuracy 0.9440, no-water baseline 0.8049) and the adapted model 0.7458 / 0.8544, confirmed by the clean-runtime Kaggle T4 run of the fixed blob `a61580e4` recorded above (a CPU pre-flight had given 0.7459 / 0.8545); see `MODEL_CARD.md` §Runtime.

## Current status

**Release-grade.** The `E2E` notebook blob `a61580e4` (committed at `b6240ae`, the idempotent-scaling fix) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-25 (10/10 ok (1 restart after install cell), 392.9 s, 106 files, 2664 MB fetched and digest-verified inside the notebook, the checkpoint converted in the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.
