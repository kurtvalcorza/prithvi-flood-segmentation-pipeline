"""Offline tests for the pinned Sen1Floods11 sample, role assignment, BYOD loaders and sample export."""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest

from conftest import synthetic_chip, synthetic_records
from prithvi_flood_segmentation_pipeline import (
    CORPUS_BASE_URL,
    IMAGE_SIZE,
    SAMPLE_RECORDS,
    check_split_disjoint,
    dataset_manifest,
    fetch_corpus,
    fetch_object,
    fetch_sample_dataset,
    load_byod_dataset,
    read_corpus,
    split_dataset,
    write_dataset_csv,
    write_sample_pair,
)
from prithvi_flood_segmentation_pipeline import samples as sm


def test_pinned_records_are_consistent(forbid_model_imports):
    assert len(SAMPLE_RECORDS) == 44
    roles = {}
    for name, role, image_bytes, image_sha, label_bytes, label_sha in SAMPLE_RECORDS:
        roles[role] = roles.get(role, 0) + 1
        assert role in sm.ROLES and "_" in name and image_bytes > label_bytes > 0
        assert len(image_sha) == 64 and len(label_sha) == 64
    assert roles == {"train": 24, "validation": 8, "test": 12}
    assert len({r[0] for r in SAMPLE_RECORDS}) == 44
    assert sum(r[2] + r[4] for r in SAMPLE_RECORDS) == sm.CORPUS_BYTES
    regions = {r[0].split("_")[0] for r in SAMPLE_RECORDS}
    assert len(regions) >= 8
    assert sm.object_path("India_1", "image") == "data/flood_events/HandLabeled/S2Hand/India_1_S2Hand.tif"
    assert sm.object_path("India_1", "label").endswith("LabelHand/India_1_LabelHand.tif")


def _tiff_bytes(array, **kwargs) -> bytes:
    import tifffile

    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, photometric="minisblack", **kwargs)
    return buffer.getvalue()


def _fake_corpus(monkeypatch, n_per_role=(4, 2, 2)):
    """Replace the record table with synthetic 13-band chips whose bytes the fake bucket serves."""
    records, served = [], {}
    index = 0
    for role, count in zip(sm.ROLES, n_per_role, strict=True):
        for _ in range(count):
            image, label = synthetic_chip(seed=index)
            stack = np.zeros((13, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int16)
            for out_index, band_index in enumerate(sm.S2_L1C_BAND_INDICES):
                stack[band_index] = (image[out_index] * 10_000).astype(np.int16)
            image_bytes = _tiff_bytes(stack, planarconfig="separate")
            label_bytes = _tiff_bytes(label.astype(np.int16))
            name = f"Region{index % 3}_{index}"
            image_sha, label_sha = hashlib.sha256(image_bytes).hexdigest(), hashlib.sha256(label_bytes).hexdigest()
            records.append((name, role, len(image_bytes), image_sha, len(label_bytes), label_sha))
            served[CORPUS_BASE_URL + sm.object_path(name, "image")] = image_bytes
            served[CORPUS_BASE_URL + sm.object_path(name, "label")] = label_bytes
            index += 1
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(records))
    return served


def test_fetch_object_verifies_and_caches(tmp_path, forbid_model_imports):
    payload = b"tif-bytes"
    calls = []

    def fetcher(url):
        calls.append(url)
        return payload

    expected = (len(payload), hashlib.sha256(payload).hexdigest())
    assert fetch_object("data/x/a.tif", expected=expected, cache_dir=tmp_path, fetcher=fetcher) == payload
    assert fetch_object("data/x/a.tif", expected=expected, cache_dir=tmp_path, fetcher=fetcher) == payload
    assert calls == [CORPUS_BASE_URL + "data/x/a.tif"]
    with pytest.raises(ValueError, match="pinned"):
        fetch_object("data/x/b.tif", expected=expected, cache_dir=tmp_path, fetcher=lambda url: b"tampered")


def test_sample_dataset_roles_and_manifest(tmp_path, monkeypatch, forbid_model_imports):
    served = _fake_corpus(monkeypatch)
    splits = fetch_sample_dataset(cache_dir=tmp_path / "cache", fetcher=lambda url: served[url])
    assert {k: len(v) for k, v in splits.items()} == {"train": 4, "validation": 2, "test": 2}
    assert splits["test"][0]["id"] == "test-000" and splits["test"][0]["image"].shape == (6, IMAGE_SIZE, IMAGE_SIZE)
    assert 0.0 <= float(splits["test"][0]["image"].max()) <= 1.0  # scaled from int16 reflectance
    assert set(np.unique(splits["test"][0]["label"])) <= {-1, 0, 1}
    manifest = dataset_manifest(splits)
    assert manifest["disjoint"] == {"train": 4, "validation": 2, "test": 2} and len(manifest["digest"]) == 64
    assert manifest["splits"]["train"]["regions"] == ["Region0", "Region1", "Region2"]
    files = fetch_corpus(cache_dir=tmp_path / "cache", fetcher=lambda url: (_ for _ in ()).throw(AssertionError(url)))
    assert len(files) == 8  # served from the cache, nothing fetched
    assert read_corpus(files)["train"][0]["source"].startswith(CORPUS_BASE_URL)


def test_check_split_disjoint_detects_leakage(forbid_model_imports):
    records = synthetic_records(4)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:2], "test": [records[0]]})


def test_split_dataset(forbid_model_imports):
    records = synthetic_records(12)
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 12 and len(splits["train"]) >= 4 and splits["test"]
    check_split_disjoint(splits)
    assert sum(len(v) for v in split_dataset(records + [{**records[0], "id": "dup"}]).values()) == 12
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, test_fraction=0.9)


def test_byod_directory_and_zip_loaders(tmp_path, forbid_model_imports):
    records = synthetic_records(5)
    folder = tmp_path / "byod"
    folder.mkdir()
    for record in records:
        write_sample_pair(record, folder / f"{record['id']}_S2Hand.tif", folder / f"{record['id']}_LabelHand.tif")
    write_dataset_csv(records, folder / "pairs.csv")
    loaded = load_byod_dataset(folder)
    assert [r["id"] for r in loaded] == [r["id"] for r in records]
    assert np.allclose(loaded[0]["image"], records[0]["image"], atol=1e-6)
    assert np.array_equal(loaded[0]["label"], records[0]["label"])
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"nested/{path.name}")
    assert len(load_byod_dataset(archive)) == 5
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "no table")
    with pytest.raises(ValueError, match="pairs.csv"):
        load_byod_dataset(bad)
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "missing.tar")
    (folder / "pairs.csv").write_text("id,image\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(folder)
