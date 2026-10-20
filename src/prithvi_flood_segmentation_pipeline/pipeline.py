"""Prithvi-EO-2.0-300M-TL Sen1Floods11 (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`) DIMER
pipeline: verified snapshot, one-time conversion of the pickled Lightning checkpoint into safetensors, flood-extent
segmentation of six-band Sentinel-2 chips, held-out evaluation against a no-water baseline, and bounded fine-tuning
of the decoder to a user's labelled chips with a portable adapter.

Prithvi-EO-2.0 (Szwarcman et al., 2024) is a ViT-L masked-autoencoder foundation model for Harmonized Landsat
Sentinel-2 imagery; the `TL` variant adds temporal and location embeddings. The checkpoint packaged here is the
upstream authors' fine-tune for flood mapping: the 300 M-parameter encoder, a `LearnedInterpolateToPyramidal` neck
over four encoder depths, a UPerNet decoder and a two-class head, fine-tuned with TerraTorch on the 252-chip official
training split of the hand-labelled Sen1Floods11 chips (512 × 512, Sentinel-2 L1C top-of-atmosphere reflectance;
six bands: blue, green, red, narrow NIR, SWIR 1, SWIR 2), with the 89-chip validation split used for early stopping.

Deterministic training (R70-DET1): torch 2.14.0 has no deterministic CUDA kernel for the backward of bilinear
`F.interpolate` and `AdaptiveAvgPool2d`, nor for NLLLoss. `build_model` therefore patches the rebuilt TerraTorch
UperNet decoder, its pooling pyramid and the `PixelWiseModel` output rescale (instance-level; site-packages are not
edited) to use separable matmuls with precomputed interpolation / averaging matrices, and `adapt` uses
`deterministic_cross_entropy`. `set_determinism` sets the cuDNN flags and `torch.use_deterministic_algorithms(True)`
without `warn_only`; `CUBLAS_WORKSPACE_CONFIG` is set when this module is imported, before CUDA initialises.

The upstream asset is a PyTorch Lightning checkpoint — a torch zip archive whose pickle references only
`collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and two storage classes (verified statically by
`audit_pickle`). Under the fleet asset specification (§11) that is executable serialization, so this package
converts it once — `torch.load(weights_only=True)`, the `state_dict` entry, the `model.` prefix stripped — into
safetensors with a pinned digest, and serves only the converted file. The architecture is rebuilt from the
`terratorch` package on PyPI with `backbone_pretrained=False` and loaded strictly; nothing is fetched from the Hub
at load time except the manifest-listed files.

Everything model-related is imported lazily so that snapshot verification, the pickle audit and input validation
run (and can refuse) before `torch` or `terratorch` are imported (fleet RTM-001). `numpy` and `tifffile` are used
for chips and are imported freely.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickletools
import random
import tempfile
import time
import types
import warnings
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# cuBLAS reads this when its handle is created; set before any CUDA work so deterministic GEMMs are available.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
MODEL_REVISION = "91ce9d38086a80b078a192b374df758b8855b732"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "prithvi-eo-2.0-300m-tl-sen1floods11"
ARTIFACT_FORMAT = "org.valcorza.prithvi-flood-segmentation.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Immutable upstream source asset (a Lightning checkpoint, i.e. a pickle; see docs/WEIGHTS.md).
SOURCE_CKPT_NAME = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
SOURCE_CKPT_BYTES = 1_276_843_350
SOURCE_CKPT_SHA256 = "76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270"
# Code-free serving file produced deterministically by `convert_model` (asset spec §11.2).
CONVERTED_WEIGHTS_NAME = "prithvi-eo-2.0-300m-tl-sen1floods11.safetensors"
CONVERTED_SHA256 = "65e4377f96c651dff586bf6ef08c9b8d6c6b59c4dc05e5dfbfe320b264ac47fd"
CONVERTED_BYTES = 1_276_749_320
# Static-audit digest of the source pickle (sorted global names), see `audit_pickle`.
PICKLE_AUDIT_SHA256 = "5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10"
CKPT_ALLOWED_GLOBALS = frozenset(
    {"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage", "torch.LongStorage"}
)
STATE_DICT_PREFIX = "model."

# Architecture (config.yaml of the pinned snapshot) and data-contract facts.
BACKBONE = "prithvi_eo_v2_300_tl"
DECODER = "UperNetDecoder"
DECODER_ARGS: dict[str, Any] = {"decoder_channels": 256}
NECKS: tuple[dict[str, Any], ...] = (
    {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
    {"name": "ReshapeTokensToImage"},
    {"name": "LearnedInterpolateToPyramidal"},
)
HEAD_ARGS: dict[str, Any] = {"head_dropout": 0.1}
PARAMETER_COUNT = 318_968_580  # nn.Parameters; the state dict also carries 208,909 buffer elements
STATE_NUMEL = 319_177_489
STATE_TENSORS = 381
ENCODER_TENSORS = 296
NUM_CLASSES = 2
CLASS_NAMES: tuple[str, ...] = ("no water", "water")
IGNORE_INDEX = -1
BANDS: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
# Sentinel-2 L1C 13-band order -> the six Prithvi bands (B2, B3, B4, B8A, B11, B12), as upstream's inference script.
S2_L1C_BAND_INDICES: tuple[int, ...] = (1, 2, 3, 8, 11, 12)
# TerraTorch's Sen1Floods11 datamodule statistics for the six bands (reflectance scale, after CONSTANT_SCALE).
MEANS: tuple[float, ...] = (0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015)
STDS: tuple[float, ...] = (0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938)
CONSTANT_SCALE = 1e-4  # S2Hand chips are int16 reflectance × 10 000
# The plausible reflectance ceiling. A chip whose maximum exceeds it is read as reflectance × 10 000 and scaled by
# CONSTANT_SCALE; every checked chip is then refused unless it lies within [-0.5, REFLECTANCE_MAX]. Scaling and
# refusal share one threshold, so a checked record never triggers scaling again: re-checking it (as predict,
# evaluate and adapt do) is a no-op. A lower trigger (the former 1.0) rescaled bright checked chips a second time.
REFLECTANCE_MAX = 2.0
NO_DATA_VALUES: tuple[float, ...] = (0.0, -9999.0)  # replaced by 0 before normalisation, as upstream
IMAGE_SIZE = 512
MIN_RECORDS = 4
MAX_RECORDS = 2_000
ADAPTATION_MODES = ("decoder", "decoder+last_block")  # the only scopes an adapter may declare
LAST_BLOCK_PREFIX = "encoder.blocks.23."
TRAINABLE_PREFIXES: dict[str, tuple[str, ...]] = {
    "decoder": ("neck.", "decoder.", "head."),
    "decoder+last_block": ("neck.", "decoder.", "head.", LAST_BLOCK_PREFIX),
}


# --------------------------------------------------------------------------------------------------
# manifest, staging, static pickle audit and conversion
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    if SOURCE_CKPT_NAME not in listed:
        raise ValueError(f"manifest does not list {SOURCE_CKPT_NAME}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == SOURCE_CKPT_NAME and (size, digest) != (SOURCE_CKPT_BYTES, SOURCE_CKPT_SHA256):
            raise ValueError(f"{entry['path']}: manifest digest disagrees with the package constant")
    return manifest


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check the converted serving file (safetensors) against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    file_path = root / CONVERTED_WEIGHTS_NAME
    if not file_path.is_file():
        raise FileNotFoundError(f"converted file missing: {file_path}")
    size = file_path.stat().st_size
    if size != CONVERTED_BYTES:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: size {size} != pinned {CONVERTED_BYTES}")
    digest = _sha256_file(file_path)
    if digest != CONVERTED_SHA256:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: sha256 {digest} != pinned {CONVERTED_SHA256}")
    return {"files": [{"path": CONVERTED_WEIGHTS_NAME, "bytes": size, "sha256": digest}]}


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed Hub file) and, when the
    converted serving file is present, that against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    converted = (root / CONVERTED_WEIGHTS_NAME).is_file()
    if converted:
        verify_converted(root)
    return {**manifest, "converted": converted}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest and git-ignores the
    1.28 GB checkpoint and the safetensors it converts to)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_globals(data: bytes) -> dict[str, int]:
    """Every global a pickle stream would import, collected with `pickletools.genops` (no execution)."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
    return found


def audit_pickle(path: str | Path, *, allowed: frozenset[str] = CKPT_ALLOWED_GLOBALS) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, or inside a torch zip archive) would import and refuse any
    outside `allowed`. Executes nothing. Returns the sorted globals and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    nested = 0
    if data[:4] == b"PK\x03\x04":
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                nested += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "torch_archive": data[:4] == b"PK\x03\x04",
        "pickles": nested if nested else 1,
        "globals": sorted(found),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}")
    return summary


def _check_pinned_source(root: Path) -> dict[str, Any]:
    source = root / SOURCE_CKPT_NAME
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != SOURCE_CKPT_BYTES:
        raise ValueError(f"{SOURCE_CKPT_NAME}: size {size} != pinned {SOURCE_CKPT_BYTES}")
    digest = _sha256_file(source)
    if digest != SOURCE_CKPT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: sha256 {digest} != pinned {SOURCE_CKPT_SHA256}")
    audit = audit_pickle(source)
    if audit["audit_sha256"] != PICKLE_AUDIT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: pickle audit digest {audit['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256}")
    return {"path": SOURCE_CKPT_NAME, "bytes": size, "sha256": digest, "audit": audit}


def build_model() -> Any:
    """Instantiate the fine-tuned architecture from the installed `terratorch` package (no pretrained download),
    patched by `make_deterministic` (no state-dict change)."""
    from terratorch.models import EncoderDecoderFactory

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = EncoderDecoderFactory().build_model(
            task="segmentation",
            backbone=BACKBONE,
            backbone_pretrained=False,
            backbone_bands=list(BANDS),
            decoder=DECODER,
            num_classes=NUM_CLASSES,
            rescale=True,
            necks=[dict(n) for n in NECKS],
            **DECODER_ARGS,
            **HEAD_ARGS,
        )
    make_deterministic(model)  # R70-DET1: deterministic CUDA kernels for every op on the training path
    return model


def convert_model(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the pinned Lightning checkpoint into safetensors, deterministically, after size, digest and
    static-audit checks: torch's weights-only unpickler, the `state_dict` entry with the `model.` prefix
    stripped, a strict load into the rebuilt architecture, and the model's own state dict saved."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    source = _check_pinned_source(root)
    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    payload = torch.load(root / SOURCE_CKPT_NAME, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{SOURCE_CKPT_NAME} did not unpickle to a Lightning checkpoint with a state_dict")
    state = payload["state_dict"]
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{SOURCE_CKPT_NAME}: state_dict is not a dict of tensors")
    stripped = {}
    for key, value in state.items():
        if not key.startswith(STATE_DICT_PREFIX):
            raise ValueError(f"{SOURCE_CKPT_NAME}: unexpected state-dict key {key!r} outside {STATE_DICT_PREFIX!r}")
        stripped[key[len(STATE_DICT_PREFIX) :]] = value
    model = build_model()
    model.load_state_dict(stripped, strict=True)
    canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
    n_params = sum(v.numel() for v in canonical.values())
    if len(canonical) != STATE_TENSORS or n_params != STATE_NUMEL:
        raise ValueError(
            f"converted state dict has {len(canonical)} tensors / {n_params} elements; expected {STATE_TENSORS} / {STATE_NUMEL}"
        )
    save_file(canonical, str(root / CONVERTED_WEIGHTS_NAME), metadata={"format": "pt"})
    report = verify_converted(root)
    return {
        "source": {k: v for k, v in source.items() if k != "audit"},
        "audit": source["audit"],
        "checkpoint": {
            "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
            "lightning_version": payload.get("pytorch-lightning_version"),
        },
        "converted": report["files"],
        "seconds": round(time.perf_counter() - started, 2),
    }


# --------------------------------------------------------------------------------------------------
# deterministic replacements for ops without a deterministic CUDA kernel (R70-DET1, torch 2.14.0)
# --------------------------------------------------------------------------------------------------

DETERMINISM_FLAGS: dict[str, Any] = {
    "cudnn.deterministic": True,
    "cudnn.benchmark": False,
    "use_deterministic_algorithms": True,
    "warn_only": False,
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
DETERMINISTIC_REPLACEMENTS: dict[str, str] = {
    "cross_entropy/NLLLoss": "deterministic_cross_entropy: log_softmax x one-hot, masked mean (ignore_index -1)",
    "interpolate(bilinear)": "deterministic_resize: separable matmul with bilinear interpolation matrices",
    "AdaptiveAvgPool2d": "DeterministicAdaptiveAvgPool2d: separable matmul with adaptive-pool averaging matrices",
}


def _bilinear_matrix(n_in: int, n_out: int, align_corners: bool) -> Any:
    """(n_out, n_in) float64 weights reproducing 1-D linear interpolation as torch computes it for a given output
    size (`area_pixel_compute_source_index`, non-cubic)."""
    import numpy as np

    matrix = np.zeros((n_out, n_in), dtype=np.float64)
    for dst in range(n_out):
        if align_corners:
            src = dst * (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
        else:
            src = max((dst + 0.5) * (n_in / n_out) - 0.5, 0.0)
        i0 = min(int(math.floor(src)), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        lam = src - i0
        matrix[dst, i0] += 1.0 - lam
        matrix[dst, i1] += lam
    return matrix


def _adaptive_pool_matrix(n_in: int, n_out: int) -> Any:
    """(n_out, n_in) float64 averaging weights of torch's adaptive average pooling bins
    [floor(i * n_in / n_out), ceil((i + 1) * n_in / n_out))."""
    import numpy as np

    matrix = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        start = (i * n_in) // n_out
        end = -((-(i + 1) * n_in) // n_out)
        matrix[i, start:end] = 1.0 / (end - start)
    return matrix


def _separable(x: Any, rows: Any, cols: Any) -> Any:
    """rows @ x @ cols.T over the last two dims of a (B, C, H, W) tensor, in x's dtype and device."""
    import torch

    a = torch.as_tensor(rows, dtype=x.dtype, device=x.device)
    b = torch.as_tensor(cols, dtype=x.dtype, device=x.device)
    return torch.matmul(torch.matmul(a, x), b.transpose(0, 1))


def deterministic_resize(x: Any, size: Sequence[int], align_corners: bool | None = False) -> Any:
    """Bilinear resize of (B, C, H, W) to `size`, equal to `F.interpolate(mode="bilinear")` up to rounding, built
    from matmuls (deterministic forward and backward on CUDA)."""
    h, w = int(size[0]), int(size[1])
    if tuple(x.shape[-2:]) == (h, w):
        return x
    corners = bool(align_corners)
    return _separable(x, _bilinear_matrix(x.shape[-2], h, corners), _bilinear_matrix(x.shape[-1], w, corners))


def deterministic_cross_entropy(logits: Any, target: Any, ignore_index: int = IGNORE_INDEX) -> Any:
    """Mean cross-entropy over the pixels whose target is not `ignore_index`, equal to
    `F.cross_entropy(logits, target, ignore_index=ignore_index)` (no class weights), from elementwise ops and sums
    only, so its CUDA forward and backward are deterministic. `logits` (B, K, H, W), `target` (B, H, W) int."""
    import torch

    keep = target != ignore_index
    safe = torch.where(keep, target, torch.zeros_like(target))
    one_hot = torch.nn.functional.one_hot(safe.long(), logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    nll = -(torch.log_softmax(logits, dim=1) * one_hot).sum(dim=1)
    weight = keep.to(nll.dtype)
    return (nll * weight).sum() / weight.sum()


def _deterministic_pool_class() -> Any:
    import torch

    class DeterministicAdaptiveAvgPool2d(torch.nn.Module):
        """Parameter-free drop-in for `nn.AdaptiveAvgPool2d(output_size)` as a separable matmul."""

        def __init__(self, output_size: Any) -> None:
            super().__init__()
            size = output_size if isinstance(output_size, (tuple, list)) else (output_size, output_size)
            self.output_size = (int(size[0]), int(size[1]))

        def forward(self, x: Any) -> Any:
            return _separable(
                x,
                _adaptive_pool_matrix(x.shape[-2], self.output_size[0]),
                _adaptive_pool_matrix(x.shape[-1], self.output_size[1]),
            )

    return DeterministicAdaptiveAvgPool2d


def _ppm_forward(self: Any, x: Any) -> list[Any]:
    """terratorch 1.2.13 `PPM.forward` with `deterministic_resize` for the bilinear upsampling."""
    return [deterministic_resize(ppm(x), x.shape[2:], align_corners=self.align_corners) for ppm in self]


def _upernet_forward(self: Any, inputs: Any) -> Any:
    """terratorch 1.2.13 `UperNetDecoder.forward` with `deterministic_resize` for both bilinear resizes."""
    import torch

    laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
    laterals.append(self.psp_forward(inputs))
    levels = len(laterals)
    for i in range(levels - 1, 0, -1):
        laterals[i - 1] = laterals[i - 1] + deterministic_resize(
            laterals[i], laterals[i - 1].shape[2:], align_corners=self.align_corners
        )
    fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(levels - 1)]
    fpn_outs.append(laterals[-1])
    for i in range(levels - 1, 0, -1):
        fpn_outs[i] = deterministic_resize(fpn_outs[i], fpn_outs[0].shape[2:], align_corners=self.align_corners)
    return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))


def _pixelwise_forward(self: Any, x: Any, **kwargs: Any) -> Any:
    """terratorch 1.2.13 `PixelWiseModel.forward` (no auxiliary heads) with `deterministic_resize` for the output
    rescale (`mode="bilinear"`, align_corners False)."""
    import sys

    upstream = sys.modules[type(self).__module__]
    image_size_out = self.image_size_out or kwargs.get("image_size") or upstream.get_image_size(x)
    if self.patch_size and self.padding is not None:
        x = upstream.pad_images(x, self.patch_size, self.padding)
    input_size = upstream.get_image_size(x)
    features = self.encoder(x, **kwargs)
    features = self.neck(features, image_size=input_size)
    mask = self.head(self.decoder([f.clone() for f in features]))
    if self.rescale and mask.shape[-2:] != input_size:
        mask = deterministic_resize(mask, input_size, align_corners=False)
    mask = self._check_for_single_channel_and_squeeze(mask)
    mask = upstream.center_crop(mask, image_size_out)
    return upstream.ModelOutput(output=mask, auxiliary_heads={})


def make_deterministic(model: Any) -> list[str]:
    """Patch a rebuilt TerraTorch flood model in place (instance-level; no parameter or buffer is added or renamed,
    so the state dict and strict loading are unchanged): every `AdaptiveAvgPool2d` becomes
    `DeterministicAdaptiveAvgPool2d`, and the UperNet decoder, its PPM and the PixelWiseModel output rescale use
    `deterministic_resize`. Returns the patched sites."""
    import torch

    pool_cls = _deterministic_pool_class()
    patched: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.AdaptiveAvgPool2d):
                setattr(module, child_name, pool_cls(child.output_size))
                patched.append(f"{name}.{child_name}: AdaptiveAvgPool2d -> DeterministicAdaptiveAvgPool2d")
        kind = type(module).__name__
        if kind == "PPM":
            module.forward = types.MethodType(_ppm_forward, module)
            patched.append(f"{name}: PPM.forward bilinear -> deterministic_resize")
        elif kind == "UperNetDecoder":
            module.forward = types.MethodType(_upernet_forward, module)
            patched.append(f"{name}: UperNetDecoder.forward bilinear -> deterministic_resize")
    if type(model).__name__ == "PixelWiseModel":
        if getattr(model, "aux_heads", None):
            raise ValueError("make_deterministic: auxiliary heads are not supported")
        model.forward = types.MethodType(_pixelwise_forward, model)
        patched.append("PixelWiseModel.forward output rescale -> deterministic_resize")
    model.deterministic_patches = list(patched)
    return patched


def set_determinism(seed: int = 0) -> dict[str, Any]:
    """Seed torch / numpy / random and switch on deterministic algorithms without `warn_only` (pre-registered).
    Returns the flags as they are actually set, for the report."""
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DETERMINISM_FLAGS["CUBLAS_WORKSPACE_CONFIG"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return determinism_state()


def determinism_state() -> dict[str, Any]:
    """The determinism flags as they are set right now (read, not set): recorded and re-checked before scoring."""
    import torch

    return {
        "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def state_fingerprint(model: Any, names: Sequence[str]) -> str:
    """SHA-256 over the named tensors (sorted, float32 bytes on CPU): the identity of an adapted state."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(names):
        tensor = state[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write to a temporary file in the same directory, then `os.replace` it onto `path`, so the final path never
    holds a half-written file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


# --------------------------------------------------------------------------------------------------
# chips, labels and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": (
        "{id, image, label?}: image = (6, 512, 512) float32 reflectance chip (or a GeoTIFF path); "
        "label = (512, 512) int mask with 0/1/-1 (or a GeoTIFF path), optional"
    ),
    "bands": list(BANDS),
    "image_size": IMAGE_SIZE,
    "value_units": (
        "Sentinel-2 L1C top-of-atmosphere reflectance × 10 000 (int16, the Sen1Floods11 S2Hand encoding) or "
        "the same reflectance in [0, 1]; products of processing baseline 04.00+ (from 2022-01-25) carry a +1000 DN "
        "offset (RADIO_ADD_OFFSET -1000) that must be removed first; 10 m grid with the 20 m bands resampled; "
        "values above 2 (REFLECTANCE_MAX) are read as reflectance × 10 000 and scaled by 1e-4; "
        "re-checking a checked record never rescales it"
    ),
    "no_data": list(NO_DATA_VALUES),
    "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
    "ignore_index": IGNORE_INDEX,
    "records": [MIN_RECORDS, MAX_RECORDS],
    "validation": (
        "record shape, band count, chip size, finiteness, value range and label values only. Nothing checks that "
        "the bands are the six Prithvi bands in the right order, that the product is L1C top-of-atmosphere "
        "reflectance (an atmospherically corrected L2A or an HLS product is a different input distribution), that "
        "the post-2022 +1000 offset was removed (it adds about 0.1 to every band and still passes the range check), "
        "the pixel size or CRS, or that the label was drawn for this chip -- any six-band 512 × 512 array is "
        "segmented without complaint"
    ),
}


def _read_tiff(path: Path) -> Any:
    """Read a GeoTIFF's pixel array with tifffile as (bands, H, W) or (H, W); no georeferencing is used."""
    import numpy as np
    import tifffile

    with tifffile.TiffFile(path) as tf:
        array = tf.asarray()
        planar = tf.pages[0].planarconfig
    if array.ndim == 3 and planar is not None and int(planar) == 1 and array.shape[-1] <= 16:
        array = np.moveaxis(array, -1, 0)  # pixel-interleaved -> band-sequential
    return np.asarray(array)


def read_chip(path: str | Path, *, band_indices: Sequence[int] | None = None) -> Any:
    """Load a chip from a GeoTIFF as float32 (6, H, W); `band_indices` selects the six Prithvi bands from a
    wider stack (e.g. S2_L1C_BAND_INDICES for a 13-band Sentinel-2 L1C file)."""
    import numpy as np

    array = _read_tiff(Path(path))
    if array.ndim != 3:
        raise ValueError(f"{Path(path).name}: expected a multi-band raster, got shape {array.shape}")
    if band_indices is not None:
        array = array[list(band_indices)]
    elif array.shape[0] != len(BANDS) and array.shape[0] == 13:
        array = array[list(S2_L1C_BAND_INDICES)]
    return np.ascontiguousarray(array.astype(np.float32))


def read_mask(path: str | Path) -> Any:
    """Load a label raster from a GeoTIFF as int64 (H, W)."""
    import numpy as np

    array = _read_tiff(Path(path))
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{Path(path).name}: a label raster must have one band, got shape {array.shape}")
        array = array[0]
    return np.ascontiguousarray(array.astype(np.int64))


def _check_record(record: Any, index: int) -> dict[str, Any]:
    import numpy as np

    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image[/label]")
    for key in ("id", "image"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image = record["id"], record["image"]
    if not isinstance(rid, str) or not rid or len(rid) > 128:
        raise ValueError(f"{label_name}: id must be a non-empty string of at most 128 characters")
    if isinstance(image, str | Path):
        if not Path(image).is_file():
            raise ValueError(f"{label_name}: image file not found: {image}")
        image = read_chip(image)
    try:
        array = np.asarray(image, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_name}: image must be a numeric array") from exc
    if array.shape != (len(BANDS), IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"{label_name}: image must have shape {(len(BANDS), IMAGE_SIZE, IMAGE_SIZE)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label_name}: image contains non-finite values")
    for value in NO_DATA_VALUES:
        array = np.where(array == value, 0.0, array)
    if float(array.max()) > REFLECTANCE_MAX:  # only reflectance × 10 000 exceeds the ceiling; see REFLECTANCE_MAX
        array = array * CONSTANT_SCALE
    if float(array.min()) < -0.5 or float(array.max()) > REFLECTANCE_MAX:
        span = (float(array.min()), float(array.max()))
        raise ValueError(f"{label_name}: reflectance outside the plausible range after scaling: {span}")
    item: dict[str, Any] = {"id": rid, "image": np.ascontiguousarray(array.astype(np.float32))}
    label = record.get("label")
    if label is not None:
        if isinstance(label, str | Path):
            if not Path(label).is_file():
                raise ValueError(f"{label_name}: label file not found: {label}")
            label = read_mask(label)
        try:
            mask = np.asarray(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label_name}: label must be an integer array") from exc
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{label_name}: label must have shape {(IMAGE_SIZE, IMAGE_SIZE)}, got {mask.shape}")
        if not np.issubdtype(mask.dtype, np.integer) and not np.all(mask == np.round(mask)):
            raise ValueError(f"{label_name}: label values must be integers")
        allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
        found = set(np.unique(mask).astype(int).tolist())
        if not found <= allowed:
            raise ValueError(f"{label_name}: label values {sorted(found - allowed)} outside {sorted(allowed)}")
        item["label"] = np.ascontiguousarray(mask.astype(np.int64))
    for key in ("split", "region", "source", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def check_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record and return its normalised copy (float32 reflectance, no-data replaced, int64 label)."""
    return _check_record(record, 0)


def chip_digest(record: Mapping[str, Any]) -> str:
    checked = _check_record(record, 0)
    digest = hashlib.sha256(checked["image"].tobytes())
    if "label" in checked:
        digest.update(checked["label"].tobytes())
    return digest.hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], chip_digest(r)] for r in records]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Structural validation of a chip dataset; raises ValueError before any model import."""
    import numpy as np

    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        if require_labels and "label" not in item:
            raise ValueError(f"records[{index}] has no label; every record of a labelled dataset needs one")
        if "label" in item:
            for c in range(NUM_CLASSES):
                class_pixels[c] += int((item["label"] == c).sum())
            ignored += int((item["label"] == IGNORE_INDEX).sum())
        checked.append(item)
    labelled = sum("label" in r for r in checked)
    if require_labels and labelled and class_pixels[1] == 0:
        raise ValueError(f"no pixel of class 1 ({CLASS_NAMES[1]}) in the dataset; nothing to learn or evaluate")
    total = int(class_pixels.sum())
    return {
        "records": checked,
        "n_records": len(checked),
        "n_labelled": labelled,
        "image_size": IMAGE_SIZE,
        "bands": list(BANDS),
        "class_pixel_fraction": {
            CLASS_NAMES[c]: round(float(class_pixels[c]) / total, 4) if total else None for c in range(NUM_CLASSES)
        },
        "ignored_pixels": ignored,
        "reflectance_range": [
            round(float(min(r["image"].min() for r in checked)), 4),
            round(float(max(r["image"].max() for r in checked)), 4),
        ],
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record; returns its id, shape, reflectance range and label class fractions."""
    item = _check_record(record, 0)
    report = {
        "id": item["id"],
        "shape": tuple(item["image"].shape),
        "reflectance_range": [round(float(item["image"].min()), 4), round(float(item["image"].max()), 4)],
        "has_label": "label" in item,
    }
    if "label" in item:
        label = item["label"]
        valid = int((label != IGNORE_INDEX).sum())
        report["label_fraction"] = {
            CLASS_NAMES[c]: round(float((label == c).sum()) / max(valid, 1), 4) for c in range(NUM_CLASSES)
        }
        report["ignored_pixels"] = int((label == IGNORE_INDEX).sum())
    return report


# --------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------


def _normalise(images: Any) -> Any:
    """(B, 6, H, W) reflectance -> standardised with the upstream datamodule statistics."""
    import numpy as np

    mean = np.asarray(MEANS, dtype=np.float32)[None, :, None, None]
    std = np.asarray(STDS, dtype=np.float32)[None, :, None, None]
    return (images - mean) / std


@dataclass
class PrithviFloodPipeline:
    """Flood-extent segmentation and bounded decoder fine-tuning on top of the verified Prithvi flood model."""

    model: Any
    device: str
    weights_dir: Path
    source: str
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
        report: Callable[[dict[str, Any]], None] | None = None,
    ) -> PrithviFloodPipeline:
        """Verify, convert if needed, rebuild from the installed package and strictly load. With
        `require_source=False` the checkpoint may be absent (the DIMER-hosted case) as long as the converted file
        verifies. `report` receives the audit and conversion records when a conversion happens."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if require_source:
            stage_missing_files(root, allow_download=allow_download)
            snapshot = verify_snapshot(root)
            if not snapshot["converted"]:
                conversion = convert_model(root)
                if report is not None:
                    report({"conversion": conversion})
                snapshot = verify_snapshot(root)
            elif report is not None:
                report({"conversion": "converted file already present and digest-verified"})
            source = "converted from the manifest-verified source checkpoint"
        else:
            verify_converted(root)
            source = "converted file, pinned digest (source checkpoint not required)"
        import torch
        from safetensors.torch import load_file

        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if chosen.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        model = build_model()
        state = load_file(str(root / CONVERTED_WEIGHTS_NAME))
        model.load_state_dict(state, strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != PARAMETER_COUNT:
            raise ValueError(f"rebuilt model has {n_params} parameters, expected {PARAMETER_COUNT}")
        model.to(torch.device(chosen)).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model=model, device=chosen, weights_dir=root, source=source)

    # ---- forward ---------------------------------------------------------------------------------------

    def _logits(self, images: Any, *, grad: bool = False) -> Any:
        """(B, 6, H, W) float32 reflectance -> (B, 2, H, W) float32 logits at input resolution."""
        import numpy as np
        import torch

        batch = torch.from_numpy(_normalise(np.asarray(images, dtype=np.float32))).to(self.device)
        use_amp = self.device.startswith("cuda")
        context = torch.enable_grad() if grad else torch.inference_mode()
        with context, torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            out = self.model(batch)
        logits = out.output if hasattr(out, "output") else out
        if tuple(logits.shape[-2:]) != tuple(batch.shape[-2:]):
            logits = deterministic_resize(logits.float(), batch.shape[-2:], align_corners=False)
        return logits.float()

    # ---- inference -------------------------------------------------------------------------------------

    def predict(self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 4) -> dict[str, Any]:
        """Segment chips: per record the argmax mask (H, W) uint8, the softmax scores (2, H, W) float32 and the
        fraction of pixels in each class. Softmax scores are the model's own outputs, not calibrated probabilities."""
        import numpy as np
        import torch

        checked = validate_dataset(records, min_records=1, require_labels=False)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        started = time.perf_counter()
        predictions = []
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            logits = self._logits(np.stack([r["image"] for r in batch]))
            scores = torch.softmax(logits, dim=1).cpu().numpy()
            masks = scores.argmax(axis=1).astype(np.uint8)
            for record, score, mask in zip(batch, scores, masks, strict=True):
                predictions.append(
                    {
                        "id": record["id"],
                        "mask": mask,
                        "scores": score.astype(np.float32),
                        "class_fraction": {CLASS_NAMES[c]: round(float((mask == c).mean()), 4) for c in range(NUM_CLASSES)},
                    }
                )
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "classes": list(CLASS_NAMES),
            "decision_rule": "argmax over the two class scores (no threshold)",
            "predictions": predictions,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 4) -> dict[str, Any]:
        """Pixel-level metrics on labelled chips (ignore index excluded): per-class IoU, mean IoU, accuracy and the
        class-1 F1, precision and recall, with the all-class-0 baseline scored on the same pixels."""
        from .metrics import majority_baseline, segmentation_metrics

        checked = validate_dataset(records, min_records=1)["records"]
        started = time.perf_counter()
        result = self.predict(checked, batch_size=batch_size)
        masks = [p["mask"] for p in result["predictions"]]
        labels = [r["label"] for r in checked]
        metrics = segmentation_metrics(masks, labels)
        return {
            "n_records": len(checked),
            "metric": "pixel IoU / F1 over the labelled pixels of the held-out chips (ignore index excluded)",
            "model": metrics,
            "baseline_no_water": majority_baseline(labels),
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 3),
        }

    # ---- adaptation ------------------------------------------------------------------------------------

    def _trainable(self, mode: str) -> list[str]:
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"trainable must be one of {ADAPTATION_MODES}")
        prefixes = TRAINABLE_PREFIXES[mode]
        return sorted(name for name, _param in self.model.named_parameters() if name.startswith(prefixes))

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 4,
        lr: float = 1e-5,
        batch_size: int = 2,
        trainable: str = "decoder",
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the neck, decoder and head (`trainable="decoder"`; `"decoder+last_block"` also
        unfreezes the last encoder block) on labelled chips: cross-entropy over the labelled pixels (ignore index
        excluded), AdamW at a fixed learning rate, seeded horizontal/vertical flips, float16 autocast with loss
        scaling on CUDA. Epoch 0 records the frozen model; the epoch with the lowest validation loss is kept."""
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        names = self._trainable(trainable)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        import numpy as np
        import torch

        torch.manual_seed(seed)
        random.seed(seed)
        started = time.perf_counter()
        model = self.model
        name_set = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in name_set)
        params = [p for n, p in model.named_parameters() if n in name_set]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        use_amp = self.device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(seed)

        def val_loss() -> float | None:
            if val_checked is None:
                return None
            model.eval()
            losses = []
            for start in range(0, len(val_checked), batch_size):
                batch = val_checked[start : start + batch_size]
                logits = self._logits(np.stack([r["image"] for r in batch]))
                target = torch.from_numpy(np.stack([r["label"] for r in batch])).to(self.device)
                losses.append(float(deterministic_cross_entropy(logits, target, ignore_index=IGNORE_INDEX)))
            return sum(losses) / len(losses)

        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        try:
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val_loss": val_loss(), "note": "frozen model"}
            if val_checked is not None:
                entry["val"] = self.evaluate(val_checked, batch_size=batch_size)["model"]
            history.append(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            best_epoch = 0
            if progress:
                progress(entry)
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                for module in model.modules():  # BatchNorm statistics stay frozen: tiny batches would corrupt them
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                order = rng.permutation(len(train_checked)).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    images = np.stack([r["image"] for r in batch])
                    labels = np.stack([r["label"] for r in batch])
                    if rng.random() < 0.5:
                        images, labels = images[..., ::-1], labels[..., ::-1]
                    if rng.random() < 0.5:
                        images, labels = images[..., ::-1, :], labels[..., ::-1, :]
                    logits = self._logits(np.ascontiguousarray(images), grad=True)
                    target = torch.from_numpy(np.ascontiguousarray(labels)).to(self.device)
                    loss = deterministic_cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
                    optimiser.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_loss()}
                if val_checked is not None:
                    entry["val"] = self.evaluate(val_checked, batch_size=batch_size)["model"]
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the model as it
            # was before adapt() (trained tensors restored), frozen, with no adapter attached.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable": trainable,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "augmentation": "seeded horizontal/vertical flips",
            "batchnorm": "running statistics frozen (eval mode) during adaptation",
            "precision": "float16 autocast + GradScaler" if use_amp else "float32",
            "n_train_records": len(train_checked),
            "n_steps": n_steps,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts -------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors as safetensors with a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        final = Path(output_dir)
        final.parent.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))  # renamed onto `final` when complete
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "converted_sha256": CONVERTED_SHA256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if final.exists():
            import shutil

            shutil.rmtree(final)
        os.replace(out, final)
        return final

    @staticmethod
    def check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> tuple[Path, str]:
        """Static checks on an adapter manifest, before any model or weights work: format and version, the pinned
        base and converted digest, exactly one weights entry named `adapter.safetensors` inside the artifact
        directory, and an adaptation mode that is one of the declared scopes. Returns the weights path and mode."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("converted_sha256") != CONVERTED_SHA256:
            raise ValueError("artifact records a different converted-base digest")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        mode = adapter.get("trainable") if isinstance(adapter, Mapping) else None
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"artifact adapter.trainable must be one of {ADAPTATION_MODES}")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path, mode

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, scope and digest, then overwrite exactly the tensors the scope allows."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path, mode = self.check_artifact_manifest(root, manifest)
        expected = self._trainable(mode)
        if sorted(manifest["tensors"]) != expected:
            raise ValueError(
                f"artifact tensor list does not match the {len(expected)} tensors that trainable={mode!r} may change"
            )
        entry = manifest["files"][0]
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the validated manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, model has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
    ) -> PrithviFloodPipeline:
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.check_artifact_manifest(root, manifest)
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download, require_source=require_source
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
