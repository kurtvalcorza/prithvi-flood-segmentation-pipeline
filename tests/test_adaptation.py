"""Adaptation, evaluation and artifact tests on a stub model (torch required, no weights, no terratorch): the
scope of the trainable tensors, epoch selection, the transactional guarantee and the artifact round trip."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import synthetic_records  # noqa: E402
from prithvi_flood_segmentation_pipeline import ADAPTATION_MODES, PrithviFloodPipeline  # noqa: E402
from prithvi_flood_segmentation_pipeline import pipeline as pl  # noqa: E402


class _StubModel(torch.nn.Module):
    """Parameter names follow the terratorch layout; the logits depend on decoder/head tensors so training moves them."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(24)])
        for block in self.encoder.blocks:
            block.w = torch.nn.Parameter(torch.ones(1))
        self.neck = torch.nn.Module()
        self.neck.scale = torch.nn.Parameter(torch.ones(1))
        self.decoder = torch.nn.Module()
        self.decoder.weight = torch.nn.Parameter(torch.zeros(6))  # frozen stub predicts class 0 everywhere
        self.head = torch.nn.Module()
        self.head.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x):
        feature = (x * self.decoder.weight[None, :, None, None]).sum(dim=1) * self.neck.scale * self.encoder.blocks[23].w
        logits = torch.stack([-feature + self.head.bias[0], feature + self.head.bias[1]], dim=1)
        return SimpleNamespace(output=logits)


def _pipeline() -> PrithviFloodPipeline:
    return PrithviFloodPipeline(model=_StubModel(), device="cpu", weights_dir=Path("unused"), source="stub")


def test_trainable_scopes():
    pipe = _pipeline()
    decoder = pipe._trainable("decoder")
    assert decoder == ["decoder.weight", "head.bias", "neck.scale"]
    with_block = pipe._trainable("decoder+last_block")
    assert with_block == ["decoder.weight", "encoder.blocks.23.w", "head.bias", "neck.scale"]
    assert ADAPTATION_MODES == ("decoder", "decoder+last_block")
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe._trainable("everything")


def test_predict_and_evaluate_shapes():
    pipe = _pipeline()
    records = synthetic_records(3)
    result = pipe.predict(records, batch_size=2)
    assert len(result["predictions"]) == 3 and result["predictions"][0]["mask"].shape == (512, 512)
    assert result["predictions"][0]["scores"].shape == (2, 512, 512) and result["decision_rule"].startswith("argmax")
    assert abs(sum(result["predictions"][0]["class_fraction"].values()) - 1.0) < 1e-3
    report = pipe.evaluate(records)
    assert set(report["model"]) >= {"iou", "mean_iou", "f1", "accuracy"} and report["baseline_no_water"]["recall"] == 0.0
    with pytest.raises(ValueError, match="batch_size"):
        pipe.predict(records, batch_size=0)


def test_adapt_trains_only_the_scope_and_keeps_the_best_epoch():
    pipe = _pipeline()
    train, val = synthetic_records(6), synthetic_records(2, seed=50)
    encoder_before = pipe.model.encoder.blocks[0].w.clone()
    before = pipe.evaluate(val)["model"]["f1"]
    seen = []
    result = pipe.adapt(train, val, epochs=3, lr=1e-2, batch_size=2, seed=0, progress=seen.append)
    assert [e["epoch"] for e in seen] == [0, 1, 2, 3] and seen[0]["note"] == "frozen model" and seen[0]["val"]["f1"] == before
    assert result["best_epoch"] == min(range(4), key=lambda i: result["history"][i]["val_loss"])
    assert result["n_trainable"] == 9 and result["n_steps"] == 9 and result["precision"] == "float32"
    assert torch.equal(pipe.model.encoder.blocks[0].w, encoder_before)
    assert pipe.adapter is not None and all(not p.requires_grad for p in pipe.model.parameters())
    assert pipe.evaluate(val)["model"] == result["history"][result["best_epoch"]]["val"]
    assert result["history"][-1]["train_loss"] < result["history"][1]["train_loss"] or result["best_epoch"] >= 1


def test_adapt_is_transactional_when_the_progress_callback_raises():
    pipe = _pipeline()
    initial = {k: v.clone() for k, v in pipe.model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipe.adapt(synthetic_records(4), None, epochs=2, lr=1e-2, progress=boom)
    assert pipe.adapter is None
    assert all(torch.equal(initial[k], v) for k, v in pipe.model.state_dict().items())
    assert all(not p.requires_grad for p in pipe.model.parameters())


def test_adapt_refusals():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(synthetic_records(4), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(synthetic_records(4), lr=1.0)
    with pytest.raises(ValueError, match="4..2000"):
        pipe.adapt(synthetic_records(3))
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe.adapt(synthetic_records(4), trainable="all")
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact("unused")


def test_artifact_round_trip_and_refusals(tmp_path):
    pipe = _pipeline()
    records = synthetic_records(4)
    pipe.adapt(records, epochs=1, lr=1e-2, trainable="decoder+last_block")
    adapted = pipe.evaluate(records)["model"]
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"tutorial": "test"})
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["tensors"] == ["decoder.weight", "encoder.blocks.23.w", "head.bias", "neck.scale"]
    assert manifest["adapter"]["trainable"] == "decoder+last_block" and manifest["metadata"] == {"tutorial": "test"}
    fresh = _pipeline()
    assert fresh.evaluate(records)["model"] != adapted
    fresh.load_artifact(out)
    assert fresh.evaluate(records)["model"] == adapted and fresh.adapter["best_epoch"] == 1
    # a scope narrower than the tensor list is refused before deserialising
    narrowed = dict(manifest, adapter={**manifest["adapter"], "trainable": "decoder"})
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(narrowed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _pipeline().load_artifact(out)
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (out / pl.ARTIFACT_WEIGHTS_NAME).write_bytes((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest or size"):
        _pipeline().load_artifact(out)
    assert np.isfinite(adapted["f1"])
