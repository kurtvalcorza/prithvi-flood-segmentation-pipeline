# Release status

Current status: **Release-grade** — the `E2E` tutorial notebook `tutorials/prithvi_flood_segmentation_colab.ipynb` (blob `a61580e4`, committed at `b6240ae`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-25 (10/10 ok (1 restart after install cell), 392.9 s, 106 files, 2664 MB staged), recorded in `docs/release-verification.md`. At that revision the pipeline package with its adaptation contract, the offline unit suite, the model-backed smoke, `MODEL_CARD.md` (MODEL_CARD_SPEC 1.1), the static validator (`tools/validate_release_assets.py`), the generator parity checks and the CI workflow exist and are green. Any later change to the carried modules or to the notebook yields a new blob and returns the registry to **Candidate** until a clean run of that blob is recorded.

## Philippines flood-mapping capstone

`tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb` (`TASK-INFERENCE`, `WORKSHOP`, Notebook Spec 2.2) is a **Candidate**, tracked separately; the status above does not qualify it. The data feasibility gate and the CPU pre-flights are recorded in `docs/release-verification.md`. A fresh Colab T4 `Run all` of the committed blob, the dataset licensing review and the task-training overlap audit are still required.
