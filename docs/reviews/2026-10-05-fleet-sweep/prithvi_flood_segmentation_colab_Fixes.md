# Fleet-sweep fixes: `prithvi_flood_segmentation_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `8e49acc`. All changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`); the notebook is regenerated. Status and release labels are unchanged. The capstone notebook
(`DIMER_Philippines_Flood_Mapping_Capstone.ipynb`) is untouched: it carries `pipeline.py` verbatim, so the carried modules are
left unchanged and the SWP-F fix lives in the tutorial cells. `validate_release_assets.py` (workshop-notebooks check included) and
`split_capstone_carrier.py --check` pass.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules. Generator → `build_notebook.py/2.2`; the template opts in. One kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (131 packages; the same pins and lock as prithvi-eo-feature-extraction; `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. Section 8 imports `importlib.metadata` itself. | Section 1, Section 8; generator, template, validator, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 1 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4–7), What to notice + Check your reasoning after Sections 4–8 quoting the recorded Kaggle T4 run of 2026-09-25 (frozen water IoU 0.7301 → adapted 0.7458, baseline accuracy 0.8049, validation loss 0.1306 → 0.1178, validation IoU 0.8614 → 0.8748), Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. Sections 5 and 7 and the closing quoted the pre-scaling-fix build record ("about 0.59", "recall above precision", "near 0.6"), which contradicted the recorded run and the new checkpoints; they now quote the recorded run. The literal `{{id, image, label}}` in the Prerequisites now renders as `{id, image, label}`. | opening, Sections 4–8 markdown, closing, Prerequisites | `test_swp_g_*` (3 tests) |
| SWP-A (quality asserts) | Fixed (no direction was asserted) | The sweep flagged Section 7's `assert kept-epoch val_loss <= frozen val_loss`. It is a procedure invariant (epoch 0 is a selection candidate); it and the validation re-scoring check are now explicit `RuntimeError` contract checks, and a recorded verdict (`improved` / `no change` / `worse`, with the delta) is added to the report and `result.json`. | Section 7, Section 8 | `test_swp_a_*` (5 tests) |
| SWP-F (frozen re-run) | Fixed | Confirmed: `adapt()` trains the neck/decoder/head in place from whatever weights the model holds, so a Section 6 re-run (the closing's optional experiments) continued training while epoch 0 was labelled "frozen model", and a Section 5 re-run scored the adapted model as frozen. Section 5 now keeps the pinned-base value of every tensor any adaptation mode may change, once, before any training; `restore_pinned_base(pipe)` puts it back and detaches the adapter. Section 5 calls it before the frozen evaluation and Section 6 before `adapt` (printing `started_from`). | Sections 5–6 | `test_swp_f_pinned_base_is_kept_once_and_restored`, `test_swp_f_sections_5_and_6_restore_before_evaluating_or_training` |
| SWP-B (BYOD upload only) | Fixed | Confirmed: BYOD used only `files.upload()`. Added `BYOD_PATH` (zip or folder; Kaggle/Jupyter); guarded upload fallback (off Colab, cancelled, multi-file, non-.zip each name the file or rule). | Section 4 | `test_swp_b_*` (3 tests) |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes several minutes; reused afterwards). Linux x86_64 only.
- Sections 5 and 6 restore the pinned base first; a re-run of Section 5 after Section 6 puts the base back (re-run 6–8 next). Section 6 prints `started_from`.
- New `BYOD_PATH` field; Section 7 prints a `verdict`; `result.json` carries `verdict`.
- Guided-layer cells; infrastructure collapsed; Section 5/7 and closing numbers now match the recorded run.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable). The Section 1 cell runs for real against a stand-in environment; the BYOD block, the Section 5 snapshot/restore block and the Section 7 cell run with stand-ins (NumPy tensors). Plumbing evidence, not model evidence.
- `build_notebook.py --check` up to date; `validate_release_assets.py` PASS (includes the capstone carrier checks); `split_capstone_carrier.py --check` OK; `ruff check src tests tools` clean.
- `pytest` with CI's dependencies only (pytest, ruff, numpy, tifffile, rasterio; torch absent): 64 passed, 4 skipped before → 82 passed, 4 skipped after.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9, quality asserts 0.

## Remaining gates

- A hosted **Run all in one pass** in a fresh Colab T4 runtime (no restart expected), then a re-run of the Section 8 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with `BYOD_PATH`).
- A full Notebook Review Framework v1 review has not been done.
