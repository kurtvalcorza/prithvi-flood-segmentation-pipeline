# Weight provenance, the pickle audit, the conversion and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`. Its checkpoint is a pickle, which this pipeline audits and converts but never serves.

## Prithvi-EO-2.0-300M-TL Sen1Floods11 weights

- Upstream: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`
- Immutable revision: `91ce9d38086a80b078a192b374df758b8855b732` (2026-08-25, "Migrate off deprecated decoder_scale_modules (terratorch PR #1095)"); the checkpoint was first published in commit `60bda88c` (2024-12-05, "Add model weights"); the file bytes are identical at both revisions.
- Source format: `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` — torch zip archive (`Prithvi-EO-V2-300M-TL-Sen1Floods11/data.pkl`, 387 entries) holding a PyTorch Lightning 2.4.0 checkpoint: `state_dict` (381 tensors, float32 and int64, under the `model.` prefix), `epoch` 41, `global_step` 630, plain-dict `hyper_parameters` (the `terratorch.tasks.SemanticSegmentationTask` arguments) and `loops`.
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter).
- Local layout: `weights/prithvi-eo-2.0-300m-tl-sen1floods11/` holds the 7 manifest entries (the checkpoint, upstream `README.md`, `config.json`, `config.yaml`, three example chips; 1,283,524,950 bytes total) with byte size and SHA-256 for each, plus the converted file described below. `verify_snapshot()` in `src/prithvi_flood_segmentation_pipeline/pipeline.py` checks the manifest entries, asserts the checkpoint's digest against the package constant, and checks the converted file against its pinned digest when present.
- Cross-check: the manifest's checkpoint digest `76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270` equals the `oid sha256` of the Hub LFS pointer at the pinned revision.

## What the pickle would execute, and how it is audited

Under the fleet asset specification (§11) a pickle is executable serialization. `audit_pickle()` disassembles the file with `pickletools.genops` — every `.pkl` inside the torch zip archive — collects every `GLOBAL` / `STACK_GLOBAL` it would import, and refuses anything outside the allow-list, executing nothing:

| File | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|
| `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` | `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2` | exactly those four | `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10` |

The audit reports 0 violations and its digest is pinned in `PICKLE_AUDIT_SHA256`; `convert_model()` refuses a file whose audit digest differs. Tests craft a torch archive carrying `os.system` and a plain pickle of a `complex` number, and assert that the audit refuses each before anything is constructed.

An allow-list bounds what the unpickler can name; the loader below bounds what it can construct. The digest pins tie the audited bytes to the loaded bytes, and the unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs size check → SHA-256 check against the package constant → static audit and audit-digest check, and only then:

- `torch.load(map_location="cpu", weights_only=True)` — torch's restricted unpickler, which constructs tensors and containers and nothing else — must return a dict with a `state_dict` of tensors whose every key starts with `model.`;
- the prefix is stripped and the 381 tensors are loaded with `strict=True` into `EncoderDecoderFactory().build_model(task="segmentation", backbone="prithvi_eo_v2_300_tl", backbone_pretrained=False, backbone_bands=[BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2], decoder="UperNetDecoder", decoder_channels=256, num_classes=2, rescale=True, head_dropout=0.1, necks=[SelectIndices(5, 11, 17, 23), ReshapeTokensToImage, LearnedInterpolateToPyramidal])` — the pinned `config.yaml`'s model arguments; the model's own state dict is saved as safetensors.

Serving file (both identities recorded, `derived_from_sha256` = the source digest above):

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `prithvi-eo-2.0-300m-tl-sen1floods11.safetensors` | 1,276,749,320 | 381 (319,177,489 elements; 318,968,580 parameters) | `65e4377f96c651dff586bf6ef08c9b8d6c6b59c4dc05e5dfbfe320b264ac47fd` | no (regenerated) |

The conversion is deterministic: the digest was reproduced on the build run, the smoke runs and the executed tutorial notebook, which converts the file it downloads. `verify_converted()` checks size and digest; `from_pretrained()` loads the safetensors with `strict=True` and asserts the parameter count.

Layout note: the checkpoint's own `hyper_parameters` describe the neck as the deprecated `decoder_scale_modules: true` with two necks, while the pinned `config.yaml` (the revision's purpose) expresses the same module as the `LearnedInterpolateToPyramidal` neck; both produce the 11 `neck.*` tensors and the strict load matched every key with 0 missing, 0 unexpected and 0 shape mismatches.

## Fidelity

No upstream regression fixture is published for this checkpoint. The evidence is the strict key-and-shape match against the architecture rebuilt from the pinned configuration under `terratorch 1.2.13`, and sane predictions on the upstream example chips: 0.470 water on `India_900498` (its public Sen1Floods11 label holds 47.3 % water), 0.273 on `Spain_7370579`, 0.056 on `USA_430764`; and on the 8 pinned validation chips a water IoU of 0.861 (the frozen model; `MODEL_CARD.md`, *Runtime*). Behaviour under the TerraTorch version that produced the checkpoint (0.99.8) was not measured.

## Runtime facts

- The model is float32 as shipped; `predict` runs under `torch.inference_mode()` with float16 autocast on CUDA and moves results to the CPU; `adapt` trains with gradients only on the selected tensors, with BatchNorm running statistics frozen.
- Inputs are standardised with the six-band means and standard deviations of TerraTorch's Sen1Floods11 datamodule after scaling reflectance × 10 000 to reflectance (`CONSTANT_SCALE = 1e-4`) and replacing no-data (0 or −9999) by 0 — the same preprocessing the fine-tune used; the `TL` temporal and location embeddings receive no coordinates, as in the fine-tune (`use_metadata: false`).
- `terratorch` pulls `torchgeo`, `lightning`, `timm`, `segmentation-models-pytorch`, `albumentations`, `rasterio`, `geopandas`, `lightly`, `h5py`, `tensorboard` and others; the pipeline uses `terratorch.models.EncoderDecoderFactory` only, and reads chips with `tifffile` (no rasterio, no georeferencing).

## The tutorial data: pinned Sen1Floods11 objects

`samples.py` fetches 44 hand-labelled chips from the public store `https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/` — each a 13-band int16 `S2Hand` GeoTIFF (Sentinel-2 L1C reflectance × 10 000, 512 × 512) and its single-band int16 `LabelHand` mask (0 / 1 / −1) — 88 objects, 103,757,095 bytes, each pinned by byte size and SHA-256 in `SAMPLE_RECORDS`. The chips were chosen on 2026-09-19 with a fixed seed from the official `flood_train_data`, `flood_valid_data` and `flood_test_data` splits (24 / 8 / 12), round-robin over the flood-event regions, from the chips whose label is at least 60 % valid and 3 % water; the roles follow the official splits. `fetch_object` refuses a mismatch before decoding and caches the raw bytes under `weights/sen1floods11/` (git-ignored). Sen1Floods11 (Bonafilia et al., 2020) is released by Cloud to Street under CC BY 4.0.

## Files deliberately not staged

The upstream repository at the pinned revision also carries `inference.py` (the authors' rasterio/terratorch inference script) and `requirements.txt`; neither is listed in the manifest or executed. The pretrained (non-fine-tuned) `Prithvi-EO-2.0-300M-TL` backbone is not fetched: `backbone_pretrained=False` and the fine-tuned checkpoint carries the encoder.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the licence and notices. DIMER may host the converted safetensors in its model store under those terms; it is derived from, and recorded beside, the unmodified upstream checkpoint.
- Upload set: `prithvi-eo-2.0-300m-tl-sen1floods11.safetensors`. **The `.pt` file must not be uploaded** — a profile that carries it would reintroduce the executable-serialization boundary this conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path; the model class comes from `terratorch==1.2.13` on PyPI, the served state dict is safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified file without the manifest or the checkpoint.
- Serving shape: a chip needs the 1.28 GB weights and one 512 × 512 six-band array; on an RTX 5070 Ti laptop GPU 12 chips take 1.3 s in float16 autocast at 2.1 GB; a CPU takes tens of seconds per chip. An adapted profile needs the weights plus a 60 MB adapter.
- One review item is open: whether the one-time restricted unpickle (in the build and, for the tutorial, in the runtime) meets the DIMER deserialization-trust bar or whether DIMER hosts only maintainer-converted files. The served artifact is the same file either way.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
