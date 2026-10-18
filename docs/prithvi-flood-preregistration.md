# Pre-registration: Prithvi flood tutorial — validation grid, fresh test and reading rule

**Registered:** 2026-10-19, before any code change, any grid run or any scoring of the fresh test.
**Base revision:** `292a70fe403241272557a070c8dd5bb769813b69` (notebook blob `e04ef87b`).
**Scope:** `tutorials/prithvi_flood_segmentation_colab.ipynb`, generated from `tools/notebook_template.py`.
**Owner rulings:** R70-D1, R70-D2, R70-D3 and R70-DET1 (option A), Kurt Valcorza, row-70 tutorial review, 2026-10-19.

This file is the contract that the notebook's fresh-test scoring checks against. Nothing below may change in
response to a fresh-test number. A later change to anything listed here is an amendment: it is dated and committed
alone, and it starts a new pre-registered evaluation.

## 1. Data

| Role | Chips | Source | Pinned by |
|---|---|---|---|
| train | 24 | official Sen1Floods11 `flood_train_data.csv` | `SAMPLE_RECORDS` (bytes + SHA-256 of S2Hand and LabelHand) |
| validation | 8 | official `flood_valid_data.csv` | `SAMPLE_RECORDS` |
| **exposed development test** | 12 | official `flood_test_data.csv` (the notebook's former "test") | `SAMPLE_RECORDS` |
| **fresh test** | 23 | official `flood_test_data.csv`, never scored by this notebook | `FRESH_TEST_RECORDS` (appendix) |

**The 12 former test chips are the "exposed development test".** They were scored in every earlier run, and the
2026-09-19 build sweep read their IoU while the learning rate was chosen (`MODEL_CARD.md`, build record). They are
reported for continuity only and never decide anything.

**Fresh-test filter (pre-registered, labels only).** The pool is built in four steps:
1. Take the 90 official test chips and drop the 12 already in the sample, which leaves 78.
2. Keep each chip whose hand label is at least 60 % valid (not −1) and at least 3 % water. This is the filter the
   44-chip sample used, and it leaves 25 chips.
3. Drop `India_900498` and `Spain_7370579`. Both are upstream example chips that the notebook predicts and displays
   in its example section. That leaves **23 chips**.
4. Use nothing else. No model output or score played any part in choosing the chips, and `Bolivia` (the official
   held-out event) is **not** used.

**What the fresh test can and cannot show:**
- The 23 chips come from the same 8 flood events as the training data: India, Mekong, Pakistan, Paraguay, Somalia,
  Spain, Sri-Lanka and USA.
- 22 of the 25 filtered chips lie within one chip width of a training or validation chip (20 of the 23 kept). None
  overlaps one by more than 1 % of its area.
- The upstream checkpoint was fine-tuned on the 252-chip official training split. The 89-chip validation split drove
  its early stopping. No test chip was trained on; this is inferred from the 630 steps = 42 × ⌊252/16⌋.
- The fresh test therefore measures **within-event generalisation to unseen chips, not generalisation to new
  events**. An out-of-event check (for example Bolivia) is not part of this evaluation.

**Fresh-pool digest:** `b7706fad689c52b3b848e2bed19da276d59d00cc187f693a0ce249f229ef4815`. It is computed by
`fresh_pool_digest()` as the SHA-256 of the compact JSON of the sorted
`[chip, image_bytes, image_sha256, label_bytes, label_sha256]` rows in the appendix.
- The fresh objects total 53,017,076 bytes.
- They are fetched from `https://storage.googleapis.com/sen1floods11/v1.1/`.
- Any object with a size or digest mismatch is refused.

## 2. Fixed settings (every one enters `preregistered`)

| Setting | Value |
|---|---|
| data source | the pinned Sen1Floods11 sample (no BYOD) |
| train / validation / exposed dev / fresh | 24 / 8 / 12 / 23 chips, digests as pinned |
| epochs | 4 |
| batch size | 2 |
| seed | 0 (torch, numpy, random; seeded flips; no DataLoader workers) |
| optimiser | AdamW, weight decay 0, gradient-norm clip 1.0 |
| precision | float16 autocast + GradScaler on CUDA |
| BatchNorm | running statistics frozen (eval mode) |
| epoch selection | lowest validation loss, strict `<`, frozen model = epoch 0 |
| grid | C0, C1, C2 below, all run, selection margin 0.005 |
| device | CUDA, Tesla T4 |
| determinism | the flags and replacements in §4 |
| bootstrap | paired over chips, n_boot = 10 000, seed = 0 |

Any other value of any of these settings makes a run exploratory. That covers form edits, BYOD, the optional
activity, a smaller batch for OOM, CPU or another GPU, and `DIMER_MASK_FRESH_TEST=1`.

An exploratory run is compared on validation only; the exposed development test is shown. It never scores the
fresh test.

## 3. Validation grid and selection (R70-D2)

| Config | Learning rate | Trainable scope |
|---|---|---|
| **C0 (default)** | 1e-5 | `decoder` (neck, decoder, head; 15.1 M parameters) |
| C1 | 3e-5 | `decoder` |
| C2 | 1e-5 | `decoder+last_block` (also encoder block 23; 27.7 M parameters) |

**Selection rule:**
- Each config is trained from the pinned base under §2. Its score is the kept epoch's validation loss.
- A challenger replaces C0 only if its loss is lower than C0's by **at least 0.005**.
- If several challengers clear the margin, the lowest loss wins. On an exact tie, C1 is preferred to C2.
- Otherwise C0 stays.
- Inside each run, the frozen model (epoch 0) is a candidate: "keep frozen unless beaten".
- The 8 validation chips also steered the upstream early stopping, and the report says so beside the result.

**When the grid runs:**
- It is **on** when a GPU is present: about 3 × 20 s of training on a T4.
- It is **off** on CPU. A CPU run trains the recorded selection and is exploratory. Until a hosted record exists, the
  recorded selection is C0.

**Grid hash.** Section 6 prints a SHA-256 over the grid rows: config, per-epoch validation losses to 6 decimals,
kept epoch and selection.
- Two masked hosted T4 runs must print the same grid hash before the run of record.
- If they differ, STOP: there is no run of record.

## 4. Determinism (R70-DET1, option A)

These flags are set before training and recorded in the report:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` with no `warn_only`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set when the carried pipeline module is imported, before CUDA initialises
- seeded `torch`, `numpy` and `random`

torch 2.14.0 documents three ops on the training path as having no deterministic CUDA kernel, so they throw under
these flags. They are replaced in carried code (`pipeline.py`) by patching the rebuilt TerraTorch modules when the
model is built. Site-packages are not edited.

| Op | Where | Replacement |
|---|---|---|
| `F.cross_entropy` / NLLLoss (CUDA) | `adapt` training and validation loss | `log_softmax` × one-hot, masked mean over the labelled pixels (same `ignore_index=-1`, no class weights) |
| bilinear `F.interpolate` backward (CUDA) | UperNet FPN top-down, FPN fuse and PPM upsample; the `PixelWiseModel` output rescale; the `_logits` resize | separable matmul with precomputed bilinear interpolation matrices (same `align_corners`) |
| `AdaptiveAvgPool2d` backward (CUDA) | UperNet PPM (pool scales 1, 2, 3, 6) | separable matmul with the adaptive-pool averaging matrices (same bins) |

**Checks:**
- On CPU, each replacement's forward outputs and gradients must match the original op within 1e-5 (float32) on
  random inputs at the real shapes.
- The patched model must build and load the converted checkpoint with `strict=True`.
- The loss must equal `F.cross_entropy` within 1e-6.

Scaled-dot-product attention (block 23 under C2) uses its deterministic backward under these flags (torch 2.14.0
`attention_backward.cu`). No other op on the path is listed. If one is found, this registration stops and is amended.

## 5. Reading rule (R70-D3)

- **Primary endpoint:** pooled water IoU on the fresh test, selected config minus the frozen model. It is computed
  from unrounded per-chip confusion counts, pooled over the 23 chips.
- **Interval:** a paired bootstrap 95 % percentile interval that resamples chips with replacement, using the same
  draw for both systems, with n_boot = 10 000 and `numpy.random.default_rng(0)`.
- **Verdict:**
  - "improved" if the interval excludes 0 (lower bound > 0) and the change is ≥ 0.02;
  - "worse" if the interval excludes 0 (upper bound < 0) and the change is ≤ −0.02;
  - otherwise "no detectable change".
- **Power:** low.
  - 23 chips from 8 events, mostly adjacent to training chips, cannot detect a change below about 0.02.
  - Within-event chips are correlated, so the chip bootstrap understates the event-level uncertainty.
  - "No detectable change" is the expected verdict for a small fine-tuning effect.
  - This is a descriptive tutorial result, not a skill claim.
- **Secondary endpoints**, never decisive: the frozen model against the no-water baseline, F1, precision, recall,
  and the exposed development test.

## 6. Score-once (SCORE_ONCE_RULE 1–8)

The fresh test is scored only in the fresh-test section, and only when all of these hold:
- `preregistered` is true over every §2 setting;
- the grid is complete and matches §3;
- the model state equals the selected system (tensor fingerprint);
- the fresh-pool digest matches;
- `DIMER_MASK_FRESH_TEST` is not `1`.

A look counts only after scoring completes and the report is written. The report and the ledger are written
atomically (temporary file, then `os.replace`). These are not looks: a failed fetch or decode, a failed check, or an
interrupted write.

`test_look` labels:
- The first scoring in a runtime is labelled "pre-registered evaluation (this runtime)".
- A repeat in the same runtime writes `_reproduction_<n>` and is labelled "reproduction".
- Once `RECORDED_EVALUATION` is set (after the first hosted T4 run), every run is a "reproduction".
- Exploratory runs write `_exploratory_<n>` with `test_look` "not scored (exploratory)" and the list of settings that
  differ.
- An unreadable ledger never crashes a run and never promotes a run to the evaluation; the run is labelled
  "reproduction".

Fixer CPU checks run with the fresh test masked. The first hosted T4 run under these settings is the evaluation of
record (`docs/release-verification.md`). Later identical hosted runs are reproductions.

## Appendix: fresh-test pins (23 chips)

| # | chip | S2Hand bytes | S2Hand SHA-256 | LabelHand bytes | LabelHand SHA-256 |
|---|---|---|---|---|---|
| 1 | India_1018327 | 2334123 | `2d162175fec6842370d1bb224414f132dbc5463d0a0ef64baae8692942b5d585` | 11303 | `98290650b74df82e3acc22aaef90bcb4272821b2afb53f05af683f94b5fe1b56` |
| 2 | India_399883 | 2129825 | `3af2d0b68c1615fbf3ccf93a1ceeaee50145810854c9bab14d8f9ed77c584079` | 9902 | `2024277074e45016fa82d8300353d8ff1476dfe68eb72d6623b838b1d4dfe973` |
| 3 | India_570384 | 2485522 | `615bb89a7871fdb5bcb577e4cb3ad5a3b7ce581779d1d54a470fd36594fa57d3` | 9209 | `15345107b9bef08c30951b22fe5240758bfbff1873daf55b6ff7c4ff9c83e96b` |
| 4 | India_591317 | 2166164 | `399e75dcc6b3bc268f427cb3aa70c47d828eb09eccb08f9ae8866b75c6cb3937` | 8323 | `6f5f4c7a34bae3b62348af93b642b2e6e420c9a2d77a857145b46fda8348873a` |
| 5 | India_828067 | 2650684 | `77cb8feb1c094931f7bc23ca97d6fa12e2dd6cfe8c7d2e16c772350ee03af7b6` | 10782 | `5f32ba153c9fa448b9ebdd8368bfff54b74c724a22ff644aaac1c55ace122237` |
| 6 | India_952728 | 2612711 | `55dca19c550154c560dcee7df23f7866eff90b96856e7c5f931f0d704c4a9b47` | 11069 | `45edbf83b77d384bff6a15dd23d8a4e3ff5ddc3df28efe0704be843ca7b0583b` |
| 7 | Mekong_1443339 | 2349634 | `afbc01a747e5c6d7b6b51c733c92c187323f25d7a24f8f7411836a326b09201b` | 13991 | `e306812aaf1587c46f9b2a0c63e4af4fe46ac9fd46a9e6286dd41d70f824454e` |
| 8 | Pakistan_694942 | 2309701 | `e6706858531cf83702787164c6b58125e55ce658cd28155c99fba14c7fccbb40` | 4661 | `095b3e78ea2bed54ed45f453344afd9067fec33fc96b9e3ce8f9d144a4301323` |
| 9 | Paraguay_1029191 | 2189228 | `fc6a1bab4c554bb3370396e5412ff446302941dad1226ef6465fb3a1def39892` | 6080 | `4186a68dae7bee570685f0c4879441ae5a9af6977e1097f215a685a098dd3cf5` |
| 10 | Paraguay_232281 | 2009196 | `b03027bb22e7940e541ec74b5a0f50a6a2c4d9a5d28cbe5610e3a017acab9fdd` | 6469 | `ce0d15ffd97158c4340641268b31f7e73687c81ec3e2184c0a7d6881fd7e2f06` |
| 11 | Paraguay_511199 | 2144324 | `47dec4832e5ec24df44d6286f5251b7bbc00907ed2eb8340d2649f57a27045da` | 6187 | `2ed431be505531a2147a4237d9d927ad89690b1a59e2b2d20d0b47ae70f2b0a8` |
| 12 | Paraguay_651904 | 2090599 | `180ab057b125584b7e83581e79be763b9e8b3368909ab388587b5f4119aa0fa9` | 3472 | `44d1514eeef21b3a0c04aa0fa7c1c49ef780f4c4835c734bfa2db172b455d289` |
| 13 | Paraguay_790830 | 2213654 | `8bba3ef74cefe2f53d05043a5fd8dac50c1f94642b78ea5e58039706021d6189` | 3950 | `256861e8f5e55203471d81b6dcdfe70c2ec9a915dc52eb3805fdd540f950a435` |
| 14 | Somalia_322855 | 2319134 | `908f9600f5d9537542656315f3dc1f79f5f6c92ca9fb72d53b2722f1d81e42af` | 11099 | `c38f887de34306b7a40e8a0edbbc756f5dad289ee52c41f559cae9b29dcf4e43` |
| 15 | Somalia_685158 | 2672633 | `20d281d5cc8cafb963a89bdddbceb711e35b6b7784e64fcae38923db4d53283b` | 14360 | `9a7bed202008b4ac9525d07c98c3d3882a9f1f121125df5b6c150731c9d6ff3a` |
| 16 | Spain_5650136 | 2206996 | `7d56099682c125ad85a5bcaac6623c0768ec3bb567dc8982ef3e8a4b36fc8a24` | 9758 | `c0a53cf82bd45a0fb9b7f1c2e9408e9dd817c06a633045fce88b017b7aa990b5` |
| 17 | Spain_7387658 | 2192267 | `80b0abea99212e477d4925d04df7226b98d151218598631282dd6fb6d6da3d8d` | 19124 | `d7f6c03457e5191733d46f9d52f8342af1d73cfbeafeeba100794443b01f79fd` |
| 18 | Sri-Lanka_534068 | 1760466 | `5aae329bce9537116dbff0de3d1fac40039a49f4b6c45447fe7a5eb62eff098c` | 1588 | `e77e5e64025852b82ab5f8624ed126e866a6fff9c661635313aab9276522ad4b` |
| 19 | Sri-Lanka_649970 | 2344414 | `ea8a1788c248cfcd4217189cb67eefc30d0b3bc8235cd79b067b41fb70f743f6` | 8972 | `d9ad0a6879b7a507e4030e7228ca021ee96666c15eb72e6b7700b7243ac5bc78` |
| 20 | Sri-Lanka_713926 | 2361023 | `40239d792db24b7c3027370699782b189fde8b10cd696422b17c68f596d7fcf6` | 4008 | `b07e813683173d668958cb456048d9904ac9f7ed9e5c8d4791d93cdec04412cc` |
| 21 | USA_595451 | 3030491 | `0bef53ef00039674cde3ddc984d27c822a619c81467546be726c90126693b9be` | 6150 | `5bd0328de97ba3435661ef47a313d96541234fb21240bba5e9f8c65f19b5317d` |
| 22 | USA_67102 | 2163989 | `d2f7395fcd935345d9435866ccb1c556c520c53ecf7acebd2325fac04cd0e98d` | 2413 | `48c52abfe9c81476960a6a22924e5dbb42fe9016674ff13f3fa9e68693bcfe50` |
| 23 | USA_905409 | 2093474 | `e44a749f3d8cbbaebdb56c1284354040d7b965899dcdbb40454f5c4ba46c60eb` | 3954 | `dfb9d85d04740b0f4555043171b581fef080ea989120c6d43694def8df416de1` |
