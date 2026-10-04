# Prithvi-EO-2.0 Flood Segmentation E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/prithvi-flood-segmentation-pipeline`  
**Notebook:** `tutorials/prithvi_flood_segmentation_colab.ipynb`  
**Reviewed commit:** `8e49acc134be42574edfd99ee16c43124b6dfc44` (`main`, confirmed with `gh api repos/kurtvalcorza/prithvi-flood-segmentation-pipeline/commits/main`)  
**Notebook Git blob:** `a61580e4a82e570d21ba4560b455ee9df1f6d13f`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-25 (commit `b6240ae`, the idempotent-scaling fix). No later commit touches the notebook, the carried modules or the generator.  
**Finding prefix:** `FL`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The engineering is careful. The notebook statically audits the pickled Lightning checkpoint against a four-global allow-list, unpickles it once through torch's weights-only loader into a digest-pinned safetensors file, fetches 88 digest-pinned Sen1Floods11 objects, scores every model number against a no-water baseline on the same pixels, labels the softmax scores uncalibrated, says plainly that the model already trained on this dataset, selects the adapted epoch on validation loss with the frozen model as epoch 0, and reloads the exported adapter into a fresh pipeline with exact parity. The carried code includes the 2026-09-25 scaling fix (`REFLECTANCE_MAX = 2.0`), and the recorded run of this blob shows the corrected metrics.

The prose did not follow the fix. The notebook still describes the run made with the scaling defect.

| Measure | Notebook text (blob `a61580e4`) | Kaggle T4 record of the same blob (2026-09-25) |
|---|---|---|
| Frozen test water IoU | "about 0.59" (Section 5); "near 0.6" (Interpretation) | **0.7301** (0.5857 was the pre-fix run of 2026-09-19) |
| Error direction of the frozen model | "recall above precision — the model over-predicts water" | **precision 0.9251 > recall 0.776**: the model *under*-predicts water |
| Frozen → adapted test water IoU | "from about 0.59 to 0.60" (Section 7) | **0.7301 → 0.7458** |
| Regions in the 12 test chips | "seven regions" (Section 7, Interpretation) | **10 regions** (the record's own print and `SAMPLE_RECORDS`) |
| BYOD run as documented | flows "through the same contract … export and reload parity" | **`KeyError: 'source_id'` in Section 5** (actual cells, stub model, CPU) |
| Model state when BYOD is re-run as instructed | "frozen" | **not reset**: Section 5 "frozen" and epoch 0 "frozen model" are the sample-adapted model (stub: 0.1109 reported vs 0.0 true) |
| Section 8 "new chips" that are new | 3 of 3 implied | **2 of 3**: `USA_430764` is byte-identical to sample test chip `test-011` |
| BYOD smallest dataset accepted | "at least four chips" | **7 chips** |

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (FL-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3) and passed only after a restart. The repository marks the blob `Release-grade` on that run.
2. **The notebook's expected results are the defective run's results (FL-M2).** The "Look for" notes, the Section 7 note and the Interpretation quote 0.59 / 0.60 and "recall above precision". A learner who runs the notebook sees 0.73 / 0.75 and the opposite error pattern. This is the 0.5857-vs-0.7301 discrepancy recorded in the workspace, still live in the learner-facing text.
3. **BYOD breaks in Section 5 (FL-M3).** Section 5 prints `record['source_id']`, which only the sample loader sets. User data never reaches adaptation, evaluation, export or reload.
4. **A rerun compares against a model that is not frozen (FL-M4).** `adapt()` trains from whatever state the model is in and always labels epoch 0 "frozen model". The documented BYOD route never reloads the base model.
5. **Guided layer largely absent (FL-M5).** The notebook declares `GUIDED` but has no audience statement, roadmap, glossary, prediction, checkpoint, troubleshooting or conclusion template, and 1,638 lines of carried modules sit in three unlabelled, uncollapsed cells.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. The Prerequisites ("Knowledge") assume the reader knows multispectral reflectance chips (bands, scaling, no-data), pixel-wise masks with an ignore class, and how IoU / precision / recall are read against a majority baseline |
| Supported runtime | "a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12)"; about 3 GB of disk; "about 2.5 GB of GPU memory" for the default adaptation |
| Promised outcomes | Pinned install; carried package; snapshot staged and digest-verified; pickle audited and converted once; 44 pinned chips fetched, validated and assigned the official roles (24 / 8 / 12) with three refusals; frozen model vs no-water baseline; bounded fine-tuning of neck, decoder and head (15.1 M parameters); paired held-out comparison; segmentation of "new chips"; safetensors adapter export and fresh reload with parity; BYOD zip "through the same contract" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `a961b97` (see FL-m7) |
| Release status | **`Release-grade`** (`tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 23 cells (10 code). Cells 5, 7 and 9 carry `pipeline.py` (957 lines), `metrics.py` (76) and `samples.py` (605). Also read: `pipeline.py` (`_check_record`, `read_chip`, `validate_dataset`, `adapt`, `evaluate`), `samples.py` (`SAMPLE_RECORDS`, `fetch_object`, `read_corpus`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `dataset_manifest`), `tools/notebook_template.py`, `tools/build_notebook.py`, `README.md`, `MODEL_CARD.md` (runtime section and its 2026-09-25 correction), `tutorials/README.md`, `docs/release-verification.md` and `STATUS.md`. `docs/execution-evidence/` holds only the separate capstone notebook's runs.
- **Documented execution evidence.** `docs/release-verification.md`, row 2026-09-25, and the workspace archive it cites (`.agent/backups/scaling-fix-kaggle-2026-09-26/out/dimer-nb2-prithvi-flood-segmentation/v3/evidence/`: `run_summary.json`, `executed.ipynb`, `executed-pass1.ipynb`). The run was on a Kaggle Tesla T4 with **the reviewed blob** (`fetched_blob_verified: true`) and a clean Hugging Face cache. Pass 1 (221.8 s) raised `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.3; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Pass 2 (171.1 s) completed 10/10. For comparison, the pre-fix archive `.agent/backups/kaggle-e2e-2026-09-19/…/v2/evidence/executed.ipynb` (blob `6d648d70`) holds the 0.5857 / precision 0.6973 / recall 0.7854 outputs that the current prose describes.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), the repository's own `.venv` (Python 3.12.12, torch 2.11.0+cpu, numpy 2.5.3, tifffile 2026.9.15; no usable `terratorch`; nothing installed). No Prithvi model was built. Model cells were driven with the repository's test stub (`tests/test_adaptation.py::_StubModel`) and are labelled as stub runs. The 88 pinned Sen1Floods11 objects came from a scratch copy of a local cache, each digest-checked by `fetch_object`.
  - **Probes (about 70 s in total):**
    - P1: notebook parse, every code cell compiles, guided-layer markers, display calls, brace typo.
    - P2: `tools/build_notebook.py --check` (OK, byte-identical), `tools/validate_release_assets.py` (PASS), offline suite with `PYTHONPATH=src` (exit 0, 73 passed).
    - P3: every result the prose states, against the recorded run of this blob and the pre-fix run.
    - P4: the three Section 8 example chips against `SAMPLE_RECORDS` digests and the upstream Sen1Floods11 hand-labelled split lists (fetched from the public bucket).
    - P5: the carried `load_byod_dataset` + `split_dataset(seed=0)`, as Section 4 calls them, on 7 synthetic zips.
    - P6: the notebook's own cells 13, 15, 17 and 19 executed in order on the stub model — default sample path first, then `USE_BYOD = True` re-run from Section 4 with a simulated Colab upload of an 8-chip zip; plus a `TRAINABLE` switch.
    - P7: whether the exported `…_sample_pairs.csv` plus the upstream objects form a loadable BYOD zip.
- **Not verified:** any cell run with the real Prithvi weights here; any Colab run (the stated runtime has no record for this notebook); the real upload dialog; BYOD beyond Section 5 on real weights; the optional experiments on real weights; the "about 2.5 GB of GPU memory" figure.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| **Technical correctness** | Strong on the default path: provenance, conversion, data fetch, metrics, adaptation selection and reload are sound and recorded. Three state defects: the install needs a restart (FL-M1), BYOD crashes on a sample-only key (FL-M3), and a rerun adapts from the already-adapted model while labelling it frozen (FL-M4). BYOD error handling is partly unactionable (FL-m1). |
| **Promise fulfilment** | Default promises are delivered on the recorded run. The BYOD promise is not (FL-M3, FL-M4), and one of the three "new chips" is a test chip already scored (FL-m2). |
| **Learner experience** | The "Look for" notes are the right device but quote the wrong run (FL-M2); there is no guided layer (FL-M5), no image or mask is shown (FL-m5), and the interpretation glosses over a recall/precision trade (FL-m4). |
| **Spec conformance** | Open `MUST`s: RUN1, RUN10, ENV6, REL2/REL11 (FL-M1); DAT14, REL12 (FL-M3, FL-M4); DAT12, DAT19 (FL-m1); SPL5 (FL-m3, BYOD); UNC4 (FL-m6); ST5 provenance (FL-m7). Open `SHOULD`s: GDL1–GDL14, UX8 (FL-M5); GDL8 (FL-M2); UX3, UX11 (FL-m5); SPL10 (FL-m3); EXE5 (FL-S3). |

## 3. Findings

### FL-M1 — Major: `Run all` needs a manual restart after the install cell

**Location:** Section 1, install cell (cell 3); `docs/release-verification.md` Current status; generator `tools/notebook_template.py` (install cell).

**Observed issue:** The install cell `pip install`s the pins into the running kernel, then raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')` whenever a pin replaced an already-imported distribution. In a fresh Kaggle T4 image that always happens (`numpy` 2.0.2 preloaded vs pin 2.5.3, `cuda-bindings` 12.9.4 vs 13.4.3).

**Consequence:** A learner who chooses `Run all` stops at cell 3 after about 3.7 minutes of installation and must restart and run again. The release record marks the blob `Release-grade` on a two-pass run ("10/10 code cells ok (1 restart after install cell)"), which RUN10 and ENV6 forbid.

**Evidence:** Documented execution evidence: `run_summary.json` passes `[{attempt 1, ok false, 221.8 s}, {attempt 2, ok true, 171.1 s}]` and the pass-1 error text above. Source inspection: the guard in cell 3.

**Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass. The setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. This repository's own capstone (`tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb`) already uses the pattern with the same model semantics. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/notebook_template.py`, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS; return the status to `Candidate` until then.

**Acceptance check:** On a fresh Colab or Kaggle GPU runtime, `Run all` on the regenerated blob completes every code cell in one pass with no error output and no restart; the release record shows one pass for that blob.

**Spec:** RUN1, RUN10, ENV6, REL2, REL11.

### FL-M2 — Major: the expected results in the notebook are those of the pre-fix, defective run

**Location:** Section 5 markdown (cell 14) "Look for"; Section 7 markdown (cell 18); Interpretation (cell 22); generator `tools/notebook_template.py` lines 183–184, 245–246, 348 and 354; also `README.md` line 19.

**Observed issue:** The notebook says the frozen test water IoU is "about 0.59, with recall above precision — the model over-predicts water on some regions", that adaptation moved it "from about 0.59 to 0.60", and that the model "finds water with an IoU near 0.6 and a recall above its precision". These are the 2026-09-19 numbers (frozen 0.5857, precision 0.6973, recall 0.7854; adapted 0.6005). That run was produced by the scaling defect that rescaled `Paraguay_868895` and `Spain_2938657` to about 10⁻⁴ of their reflectance; `docs/release-verification.md` and `MODEL_CARD.md` carry a correction for it. The recorded run of **this** blob prints frozen 0.7301 (precision 0.9251, recall 0.776) and adapted 0.7458 (precision 0.9165, recall 0.8002). Section 7 and the Interpretation also say the 12 test chips come from "seven regions"; the record prints 10 (`Ghana` … `USA`), which `SAMPLE_RECORDS` confirms.

**Consequence:** A learner checks their output against the "Look for" note and finds a different number and the opposite error pattern. They are told the model over-predicts water when it misses water, which is the conclusion the lesson asks them to draw about flood maps. The interpretation's limitation statement understates the regional spread of the test sample. The repository's headline (README line 19) still quotes the defective figure.

**Evidence:** Documented execution evidence: `executed.ipynb` of blob `a61580e4`, cells 15 and 19, against the pre-fix `executed.ipynb` of blob `6d648d70`. Direct execution (P3): all six prose checks true; record test regions = 10. Source inspection of the template lines above.

**Recommended correction:** In `tools/notebook_template.py`, replace the stated values with ones that match the current record, or better, state the shape of a normal result without hard-coding it (GDL8): for example "a water IoU well above the baseline's 0, with precision above recall on this sample — the model misses some water rather than inventing it". Correct "seven regions" to ten. Regenerate, and correct README line 19 in the same change. Add a check (in `tests/` or the validator) that numbers the template quotes as "in the build record" match the release-record row for the current blob.

**Acceptance check:** No markdown cell of the regenerated notebook, and no line of README, states 0.59, 0.60 or "recall above precision" for the frozen model; any quoted numbers match the release-record row for that blob (water IoU 0.7301 → 0.7458, precision > recall, 10 test regions, or the newer run's values).

**Spec:** GDL8 (and promise fulfilment / explanation dimensions of the framework).

### FL-M3 — Major: the BYOD branch crashes in Section 5 with `KeyError: 'source_id'`

**Location:** Section 5 (cell 15), the per-chip loop `print({'chip': record['source_id'], …})`; `load_byod_dataset` in `samples.py` (records carry only `id`, `image`, `label`); opening cell and `tutorials/README.md` ("user chips flow through the same validation, baseline, adaptation, evaluation, inference, export and reload cells").

**Observed issue:** Only the sample loader (`read_corpus`) sets `source_id` and `region`. With `USE_BYOD = True`, Section 5 prints the baseline and frozen metrics, then fails on the first per-chip line. The `except` clauses elsewhere do not catch it; Sections 6–8 are never reached.

**Consequence:** The BYOD promise fails at its first model stage. User data gets no adaptation, no paired comparison, no adapter and no reload parity, which is exactly the local sequence DAT14 requires for an adaptation-capable `E2E` notebook. The Section 4 dataset summary also prints empty `regions` for user data.

**Evidence:** Direct execution (P6): the notebook's cells 13 then 15, run with `USE_BYOD = True` and a simulated Colab upload of an 8-chip synthetic zip on the repository stub model: cell 13 completes, cell 15 raises `KeyError: 'source_id'` after printing `frozen_test`. P5: BYOD records have keys `['id', 'image', 'label']`.

**Recommended correction:** Use `record.get('source_id', record['id'])` in the template (as `write_dataset_csv` already does), or have `load_byod_dataset` set `source_id` from `id` and an optional `region` column from `pairs.csv`. Add an offline test that runs the generated Section 4–7 cells on a stub model with BYOD records.

**Acceptance check:** With `USE_BYOD = True` and a valid 8-chip zip, Sections 4–8 complete on a hosted GPU runtime and write `…_result.json` with `data_source` naming the zip and a passing reload-parity block; the offline stub test passes.

**Spec:** DAT14, REL12.

### FL-M4 — Major: a rerun adapts from the already-adapted model and labels it "frozen"

**Location:** Opening cell ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); Section 5 (cell 15); Section 6 (cell 17) and `PrithviFloodPipeline.adapt` in `pipeline.py` (epoch 0 note `"frozen model"`); Section 7 (cell 18) "the gap between frozen and adapted is the number to watch"; Interpretation's optional experiments.

**Observed issue:** `adapt()` trains the selected tensors from their current values and keeps the best epoch in place; `initial_state` is restored only on an exception, and Section 4 does not reload the model. After the default run, the BYOD rerun therefore scores the sample-adapted model in Section 5 as `frozen_test` / `frozen_val`, starts Section 6 from it and labels its epoch 0 "frozen model". The same holds for re-running Section 6 for the optional experiments. Switching `TRAINABLE` from `'decoder+last_block'` back to `'decoder'` also leaves the last encoder block modified while the exported adapter lists only the decoder tensors, so the artifact no longer reproduces the in-memory model.

**Consequence:** On the learner's own chips, the comparison the notebook calls "the number to watch" is between two adapted models. After an optional experiment, the "frozen" baseline can differ substantially from the packaged model, and the learner cannot tell.

**Evidence:** Direct execution (P6, stub model, CPU): after the default path, the BYOD rerun's Section 5 reports frozen test water IoU 0.1109 where a fresh stub scores 0.0; Section 6 epoch 0 is noted `"frozen model"` with val loss 0.6921, against 0.6931 for a fresh stub on the same split. The `TRAINABLE` switch leaves `encoder.blocks.23.w` at 1.002445 (base 1.0) while `trainable_names` lists only `decoder.weight`, `head.bias`, `neck.scale`. Source inspection of `adapt()`. The effect on the reload-parity assertion with real weights is inferred, not executed.

**Recommended correction:** Snapshot the trainable tensors right after `from_pretrained` and restore them at the start of the BYOD branch and of Section 6 (or give `adapt()` a `reset_to_base=True` default that reloads them from the verified safetensors), and write epoch 0's note as "frozen base" only when the tensors equal the base. Alternatively instruct "Runtime → Restart and run all with `USE_BYOD = True`" and give the same rerun scope for each optional experiment. Fix in `pipeline.py` and `tools/notebook_template.py`, then regenerate.

**Acceptance check:** After a complete default run, setting `USE_BYOD = True` and re-running from Section 4 as documented gives Section 5 frozen metrics identical to a fresh-runtime BYOD run, and Section 6 epoch 0's validation loss equals that fresh run's epoch 0; running `decoder+last_block` then `decoder` leaves the in-memory model equal to base + exported adapter.

**Spec:** DAT14, REL12 (and GDL10 for the rerun instruction).

### FL-M5 — Major (learner-facing): the declared `GUIDED` layer is largely absent

**Location:** Whole notebook; carried module cells 5, 7, 9; generator `tools/notebook_template.py`.

**Observed issue:** The notebook declares mode `GUIDED`, but has no intended-audience statement (GDL1), no **How to use this notebook** (GDL2), no roadmap (GDL3), no Input → Model → Output contract near the opening (GDL4; the Prerequisites' data contract is the closest), no glossary although IoU, ignore class, L1C, UPerNet, pyramid neck, BatchNorm statistics, float16 autocast, GradScaler and safetensors appear (GDL6), no prediction before any principal result (GDL7), no interpretation checkpoints or sample answers (GDL9), no Predict → Change → Run → Observe → Explain activity (GDL10; the optional experiments are one sentence), no Infrastructure labelling or collapsed carrier cells (GDL11; 1,638 lines in three cells, `cellView` unset on all 10 code cells), no troubleshooting (GDL13) and no conclusion template (GDL14). Sections end without a synthesis (UX8). "Look for" notes exist for Sections 1, 4 and 5, and a "Watch the validation loss" note for Section 6 (GDL8 partly met, but see FL-M2).

**Consequence:** A self-paced learner meets 1,600 lines of carrier code before any lesson, gets no help separating essentials from infrastructure, and is never asked to predict, check or explain, so the objectives "read pixel IoU … against a no-water baseline" and "compare the adapted and frozen models" are exercised only by reading printed dictionaries.

**Evidence:** Source inspection; P1 markers (`How to use`, `Glossary`, `Troubleshoot`, `Infrastructure`, `Check your reasoning`, `audience`, `Roadmap` all absent; `cellView_set: []`).

**Recommended correction:** Add the NOTEBOOK_SPEC 2.2 guided layer in `tools/notebook_template.py`: audience, how-to-use, roadmap, task contract, collapsible glossary, a prediction before Sections 5 and 7 (for example "will the adapted model beat a frozen model that already trained on this dataset?"), "What to notice" notes, collapsible "Check your reasoning" answers, one bounded PCROE activity (for example `TRAINABLE = 'decoder+last_block'`, with the reset from FL-M4), `# @title Infrastructure: …` with `cellView: form` on the carrier cells, a troubleshooting section (install/restart, Hub download, disk, GPU memory, BYOD errors), and a conclusion template. The repository's capstone notebook already implements GDL1–GDL15 and can serve as the model.

**Acceptance check:** The GDL1–GDL14 checklist in NOTEBOOK_SPEC 2.2 passes item by item on the regenerated notebook, and the three carrier cells open collapsed in Colab.

**Spec:** GDL1–GDL14, UX8.

### FL-m1 — Minor: BYOD stated minimum is wrong and several failures are not actionable

**Location:** Opening cell ("at least four chips with some water"); Section 4 (cell 13) upload branch and refusal probes; `split_dataset` and `load_byod_dataset` in `samples.py`.

**Observed issue:** `split_dataset(seed=0)` needs at least 4 training chips after taking 25 % for test and 20 % for validation, so 4, 5 and 6 chips are refused and 7 is the true minimum. A `pairs.csv` row naming a file missing from the zip raises a bare `KeyError: 'c0.tif'`; a non-TIFF file raises `TiffFileError: not a TIFF file`; cancelling the upload raises `StopIteration` from `next(iter(uploaded.items()))` (source). On BYOD data with 2 test chips, Section 4's three refusal probes are rejected for record count (`2 records; 4..2000 are required`) rather than for the contract each probe demonstrates.

**Consequence:** A learner following the stated contract with 4–6 chips is refused after uploading; three common mistakes give errors that do not name the failed contract or the fix; and the refusal demonstration teaches nothing on user data.

**Evidence:** Direct execution: P5 (`valid_n4/5/6` → `split leaves 2/3/3 training chips; at least 4 are required`; `valid_n7` → 4/1/2; `missing_member` → `KeyError`; `non_tiff` → `TiffFileError`); P6 `byod_c13` probe lines. Source inspection for the cancel path.

**Recommended correction:** State "at least 7 labelled chips" (or derive and print the minimum from the fractions); wrap member lookup and TIFF decode in `load_byod_dataset` with `ValueError`s naming the row id, file and expected format; handle an empty upload with a message; build the refusal probes from enough records (or call `validate_inputs` per record) so each is rejected for its own reason.

**Acceptance check:** A 7-chip zip is accepted and a 6-chip zip is refused with a message giving the minimum; a missing member, a non-TIFF image and a cancelled upload each produce a `ValueError` naming the row/file and the fix; on an 8-chip BYOD zip the three probes print their band-count, label-class and range reasons.

**Spec:** DAT12, DAT19, UX10.

### FL-m2 — Minor: one of the three "new chips" in Section 8 is a test chip already scored

**Location:** Section 8 markdown (cell 20), cell 21; objective "segment new chips" (cell 0).

**Observed issue:** The three upstream example chips are presented as chips "which carry no labels here". `examples/USA_430764_S2Hand.tif` has the same size and SHA-256 (`385418e1…`) as sample test chip `USA_430764` (`test-011`), which is labelled and scored in Sections 5 and 7. All three are in the upstream Sen1Floods11 hand-labelled **test** split, so all three have hand labels upstream and none is in the checkpoint's training split. `India_900498` and `Spain_7370579` are not in the 44 sample chips, so INF2 is met.

**Consequence:** The learner reads one "sanity check" that is a chip already evaluated, and is told the chips have no labels when they could be scored.

**Evidence:** Direct execution (P4): digest comparison with `SAMPLE_RECORDS` and membership in the bucket's `splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv`.

**Recommended correction:** Name the provenance of each chip ("upstream test-split chips; `USA_430764` is also one of our 12 test chips"), or replace `USA_430764` with another test-split chip outside the sample; optionally fetch the two other chips' `LabelHand` masks and print their IoU.

**Acceptance check:** Section 8 prose names the provenance of each segmented chip, and no chip described as new is among the 44 sample records.

**Spec:** INF2 (spirit), GDL8.

### FL-m3 — Minor: BYOD split ignores the spatial boundary the notebook tells learners to keep

**Location:** `split_dataset` (`samples.py`), Section 4 BYOD branch; Interpretation ("Split by scene or event, not by chip").

**Observed issue:** The BYOD split is a seeded shuffle by chip with no grouping key; `pairs.csv` has no group or split column, so a learner cannot follow the notebook's own advice inside the notebook. The docstring says "group them yourself", which the notebook has no way to accept.

**Consequence:** BYOD held-out numbers on chips cut from one scene or event will look better than they are, the failure the interpretation warns about. SPL5 is a `MUST` for spatial data.

**Evidence:** Source inspection (`split_dataset`, `load_byod_dataset` read only `id,image,label`).

**Recommended correction:** Accept an optional `group` (or `split`) column in `pairs.csv`, split by group when present, and warn when absent.

**Acceptance check:** A BYOD zip with a `group` column never puts one group in two roles; one with a `split` column uses those roles.

**Spec:** SPL5, SPL10 (SPL2 for a `split` column).

### FL-m4 — Minor: interpretation and optional experiments overstate or pre-state results

**Location:** Interpretation (cell 22); Section 6/7 markdown; Prerequisites (cell 1).

**Observed issue:** (a) The interpretation says adaptation "leaves those numbers about where they were", while the record shows recall 0.776 → 0.8002 and precision 0.9251 → 0.9165, a trade the notebook never points out. (b) "try `LEARNING_RATE = 1e-4` to see the frozen model win every epoch" states a result with no record, and on a rerun epoch 0 is no longer the frozen model (FL-M4). (c) The optional experiments give no rerun scope. (d) The data contract reads `{{id, image, label}}` (template brace escape leaked into markdown).

**Consequence:** The learner is told nothing moved while recall rose by 2.4 points, and is promised an experimental outcome that is not established.

**Evidence:** Documented execution evidence (Section 7 output); source inspection; P1 (`brace_typo_present: true`).

**Recommended correction:** Name the recall/precision trade and what it means for flood mapping (fewer missed flooded pixels, slightly more false alarms); phrase the learning-rate experiment as a prediction to test; state the rerun scope for each experiment; fix the brace escape in the template.

**Acceptance check:** The interpretation mentions the precision and recall change; no optional experiment states its outcome as fact; each experiment names the cells to rerun; the rendered markdown shows `{id, image, label}`.

**Spec:** GDL8, GDL10, GDL14.

### FL-m5 — Minor: a segmentation tutorial that never shows an image or a mask

**Location:** Sections 4, 5, 7 and 8.

**Observed issue:** No code cell displays a chip, a label, a predicted mask or an error map. Masks are written to GeoTIFFs and summarised as water fractions.

**Consequence:** The learner cannot see what water looks like in six-band data, where the model misses water (the dominant error on this sample, see FL-M2), or what the ignore class covers, so IoU and recall stay abstract.

**Evidence:** Source inspection; P1 (`image_display_calls: 0`).

**Recommended correction:** Add a small figure: false-colour composite (SWIR 2 / NIR / red), label, frozen and adapted masks for two test chips, and an error overlay distinguishing missed water from false water.

**Acceptance check:** Running the default path displays at least one composite + label + prediction figure inline.

**Spec:** UX3, UX11.

### FL-m6 — Minor: threshold and calibration ownership are not stated, though the docs say they are

**Location:** Section 5 markdown (cell 14); `tutorials/README.md` conformance note "Score semantics (UNC1–UNC4)".

**Observed issue:** The notebook says the scores are not calibrated probabilities and prints `decision_rule: argmax … (no threshold)`, but never says that a water threshold and its cost trade-off belong to the deployment, or who owns calibration. `tutorials/README.md` claims "the notebook says the threshold and its cost trade-off are the deployment's to set".

**Consequence:** A learner reusing the masks operationally is not told that argmax is a default, not a tuned operating point, and with recall below precision on this sample, a lower threshold is the obvious lever. The conformance note overstates the notebook.

**Evidence:** Source inspection; P1 (`threshold` absent from all markdown).

**Recommended correction:** Add one sentence to Section 5: the decision rule is argmax; deployments choose a threshold on their own validation data and own calibration.

**Acceptance check:** Section 5 markdown states the default rule and that threshold choice and calibration are the deployment's; the README note then matches.

**Spec:** UNC3, UNC4.

### FL-m7 — Minor: the recorded source revision names code without the scaling fix

**Location:** `metadata.dimer.generated_from.revision`, `NOTEBOOK_SOURCE['repository_revision']` (cell 3), the opening cell ("at revision `a961b976fab5`"), Section 2 heading, and the exported `…_result.json`; `_head_revision` in `tools/build_notebook.py`.

**Observed issue:** The notebook records revision `a961b97` as the source of its carried modules. `a961b97:src/prithvi_flood_segmentation_pipeline/pipeline.py` does not contain `REFLECTANCE_MAX = 2.0`; the carried cell does (it equals `b6240ae` and `HEAD`). The generator records `HEAD` at generation time, which was the parent of the fix commit.

**Consequence:** Every exported `result.json` attributes its numbers to a revision whose code produces the defective metrics. Anyone reproducing from the recorded revision gets 0.5857, not 0.7301.

**Evidence:** Direct inspection: `git show a961b97:…/pipeline.py | grep -c 'REFLECTANCE_MAX = 2.0'` → 0; same at `b6240ae` → 1; notebook → 1. Parity (`build --check`, `test_notebook_parity.py`) passes because it compares content, not the label.

**Recommended correction:** Regenerate from a committed tree (or record the commit that contains the carried modules, e.g. the last commit touching `src/`), and have the parity test assert that `generated_from.revision`'s modules hash to the recorded `module_sha256`.

**Acceptance check:** `git show <recorded revision>:<each carried module>` equals the carried cell for every module; the parity test enforces it.

**Spec:** ST5 (standalone provenance).

### Suggestions

- **FL-S1 — Regenerate against NOTEBOOK_SPEC 2.2.** The notebook, `metadata.dimer` and the docs declare 2.0. Acceptance: metadata and opening cell declare 2.2 and the validator checks 2.2.
- **FL-S2 — Record a Colab run.** Only Kaggle T4 runs exist for this notebook although Colab is the stated runtime. Acceptance: a Colab row for the regenerated blob in `docs/release-verification.md`.
- **FL-S3 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`.** Cell 3 reads it silently. Acceptance: a sentence in Section 1 names it and says it only skips the install. (EXE5)
- **FL-S4 — Show per-chip water IoU for the 12 test chips.** The prose rightly says pooled metrics let large chips dominate; Section 5 prints water fractions for only 6 chips. Acceptance: Sections 5/7 print per-chip IoU for frozen and adapted, with its range. (EVAL6)
- **FL-S5 — Back the GPU-memory figure.** Print `torch.cuda.max_memory_allocated()` after adaptation so "about 2.5 GB" is evidenced in each run. Acceptance: the value appears in the adapt output and the result JSON.

Checked and not raised: the exported `…_sample_pairs.csv` names the upstream objects (`<chip>_S2Hand.tif` / `_LabelHand.tif`), not the written sample pair, but zipping it with those 13-band objects loads cleanly (P7: 12 records, split 7/2/3) because `read_chip` reduces 13-band L1C files to the six bands. Section 6's "46 tensors", "15.1 M parameters — 4.7 %", Section 8's "about 60 MB" and Section 6's validation-loss story (0.1306 → 0.1178 at epoch 2, then 0.1273 / 0.1243) match the record.

## 4. Readiness

**Needs revision.** No Blocker. Five Majors are open (FL-M1 one-pass `Run all`; FL-M2 stale expected results; FL-M3 BYOD crash; FL-M4 frozen baseline on rerun; FL-M5 guided layer), and the open `MUST`s RUN1, RUN10, ENV6, REL2/REL11, DAT12, DAT14, DAT19, SPL5, UNC4 and REL12 fail release under the spec regardless of severity. The repository's `Release-grade` status rests on a two-pass run and should return to `Candidate`. Remaining gates after fixes: a one-pass hosted run of the regenerated blob (Colab preferred, FL-S2), and a BYOD run that reaches export and reload (REL12).

## 5. Verified vs inferred

- **Verified by direct execution (CPU, stub model, synthetic or cached data):** generator parity, validator PASS, 73 offline tests; the BYOD `KeyError: 'source_id'` in the notebook's own Section 5 cell; the "frozen" mislabel on rerun and the unexported last block after a `TRAINABLE` switch; BYOD minimum of 7 and the unactionable errors; the refusal probes failing on count for BYOD; the example-chip identity (digest and upstream split lists); the exported pairs CSV loading as a BYOD zip.
- **Verified from documented execution evidence (Kaggle T4, this blob):** pass-1 restart error; default-path metrics (0.7301 → 0.7458, precision > recall, 10 test regions), history and reload parity; the pre-fix run as the source of the 0.59 prose.
- **Inferred, not executed:** the size of the BYOD frozen-baseline error on real weights; the reload-parity failure after a `TRAINABLE` switch on real weights; the optional experiments; the cancel-upload `StopIteration`; GPU memory.
- **Most likely to be wrong:** FL-M2's severity. Every misstatement is confirmed against the record, but the notebook does call these "build record" numbers, and a reader could rate stale expected values Minor. Kept Major because the stated error direction is reversed (over- vs under-prediction), the "Look for" check would fail for every learner, and the stale figure is the one the repository has already corrected elsewhere as wrong.
