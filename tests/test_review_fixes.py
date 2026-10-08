"""Regression tests for the 2026-10-02 Notebook Review Framework v1 review of prithvi_flood_segmentation_colab.ipynb
(FL-M2, FL-M3, FL-M4, FL-M5 activity, FL-m1..FL-m7, FL-S1; FL-M1 is covered by tests/test_sweep_fixes.py and
tests/test_worker_colab_stubs.py).

The carried modules stay byte-identical to the capstone's carried copies, so every fix lives in tutorial cells; the
tests execute those cells. Most need only CI's dependencies (NumPy, tifffile). The end-to-end test at the bottom runs the
notebook's own Section 4-8 cells on the repository stub model (torch, safetensors and matplotlib required, skipped
otherwise): stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import csv
import io
import json
import random
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from conftest import synthetic_chip, synthetic_records
from prithvi_flood_segmentation_pipeline import SAMPLE_RECORDS, check_split_disjoint, split_dataset, validate_dataset
from prithvi_flood_segmentation_pipeline.pipeline import MIN_RECORDS
from prithvi_flood_segmentation_pipeline.samples import ROLES

with contextlib.suppress(ImportError):  # import torch before any NumPy linear algebra (Windows DLL order, FIX_PACKET row 6)
    import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_flood_segmentation_colab.ipynb"
STEM = "prithvi_flood_segmentation"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _code(notebook: dict) -> list[str]:
    return [_source(c) for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [s for s in _code(notebook) if marker in s]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _helpers(notebook: dict) -> dict:
    """The Section 4 BYOD helpers, executed with the carried package's functions."""
    source = _cell(notebook, "if USE_BYOD:")
    block = source[source.index("def byod_minimum_records(") : source.index("\nif USE_BYOD:")]
    namespace = {
        "csv": csv, "io": io, "random": random, "zipfile": zipfile, "Path": Path, "tifffile": tifffile,
        "MIN_RECORDS": MIN_RECORDS, "ROLES": ROLES, "split_dataset": split_dataset, "validate_dataset": validate_dataset,
        "check_split_disjoint": check_split_disjoint, "printed": [],
    }
    namespace["print"] = namespace["printed"].append
    exec(compile(block, "section4-helpers", "exec"), namespace)
    return namespace


def _write_chips(root: Path, n: int, *, extra: dict[str, list[str]] | None = None) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    lines = [",".join(["id", "image", "label", *extra])]
    for k in range(n):
        image, label = synthetic_chip(seed=40 + k)
        tifffile.imwrite(root / f"c{k}.tif", image, photometric="minisblack", planarconfig="separate")
        tifffile.imwrite(root / f"c{k}.mask.tif", label.astype(np.int16), photometric="minisblack")
        lines.append(",".join([f"chip{k}", f"c{k}.tif", f"c{k}.mask.tif", *(values[k] for values in extra.values())]))
    (root / "pairs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [f"chip{k}" for k in range(n)]


def _zip(folder: Path, archive: Path) -> Path:
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, path.name)
    return archive


def _records_and_rows(folder: Path) -> tuple[list[dict], list[dict]]:
    from prithvi_flood_segmentation_pipeline import load_byod_dataset

    rows = list(csv.DictReader(io.StringIO((folder / "pairs.csv").read_text(encoding="utf-8"))))
    return load_byod_dataset(folder), rows


# --- FL-m1: the true minimum is derived, stated and enforced; failures name the row, the file and the fix ----------


def test_fl_m1_minimum_is_seven_and_is_what_the_notebook_states(notebook, tmp_path):
    ns = _helpers(notebook)
    assert ns["byod_minimum_records"]() == ns["BYOD_MINIMUM"] == 7
    assert [len(split_dataset(synthetic_records(7), seed=0)[k]) for k in ("train", "validation", "test")] == [4, 1, 2]
    with pytest.raises(ValueError, match="at least 4 are required"):
        split_dataset(synthetic_records(6), seed=0)
    _write_chips(tmp_path / "six", 6)
    with pytest.raises(ValueError, match="pairs.csv lists 6 chips; bring at least 7 labelled chips"):
        ns["read_byod_table"](_zip(tmp_path / "six", tmp_path / "six.zip"))
    _write_chips(tmp_path / "seven", 7)
    assert len(ns["read_byod_table"](_zip(tmp_path / "seven", tmp_path / "seven.zip"))) == 7
    md = _markdown(notebook)
    assert "at least **seven** labelled chips" in md and "at least four chips" not in md
    assert "'byod_minimum_chips': BYOD_MINIMUM" in _cell(notebook, "if USE_BYOD:")


def test_fl_m1_missing_member_unreadable_tiff_and_missing_table_name_the_row_and_the_fix(notebook, tmp_path):
    read = _helpers(notebook)["read_byod_table"]
    folder = tmp_path / "byod"
    _write_chips(folder, 8)
    (folder / "c1.tif").unlink()
    with pytest.raises(ValueError, match=r"row 'chip1': image file 'c1\.tif' is listed but not in byod; add the file or correct the row"):
        read(folder)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pairs.csv", "id,image,label\nchip0,c0.tif,c0.mask.tif\n")
        zf.writestr("c0.tif", b"not a tiff")
        zf.writestr("c0.mask.tif", b"not a tiff either")
    with pytest.raises(ValueError, match=r"row 'chip0': image file 'c0\.tif' is not a readable GeoTIFF; save it as a six-band 512 x 512 GeoTIFF"):
        read(archive)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="empty has no pairs.csv"):
        read(tmp_path / "empty")


def test_fl_m1_refusal_probes_fail_on_their_own_rule_with_a_two_chip_test_split(notebook):
    source = _cell(notebook, "probe_fill = ")
    block = source[source.index("probe_fill = ") :]
    records = synthetic_records(8)
    namespace = {
        "np": np, "MIN_RECORDS": MIN_RECORDS, "validate_dataset": validate_dataset,
        "test_records": records[:2], "train_records": records[2:7], "val_records": records[7:],
    }
    out: list[dict] = []
    namespace["print"] = out.append
    exec(compile(block, "section4-probes", "exec"), namespace)
    rejected = {row["probe"]: row.get("rejected", "") for row in out}
    assert set(rejected) == {"five-band chip", "unknown label class", "reflectance out of range"}
    assert all(message and "records;" not in message for message in rejected.values()), rejected
    assert "(6, 512, 512), got (5, 512, 512)" in rejected["five-band chip"] and "label values [7]" in rejected["unknown label class"]
    assert "reflectance outside the plausible range" in rejected["reflectance out of range"]


# --- FL-m3: a group column keeps each scene or event in one role; a split column sets the roles -----------------------


def test_fl_m3_group_column_keeps_each_group_in_one_role(notebook, tmp_path):
    ns = _helpers(notebook)
    groups = [f"event{k // 2}" for k in range(12)]
    _write_chips(tmp_path / "g", 12, extra={"group": groups})
    records, rows = _records_and_rows(tmp_path / "g")
    splits, rule = ns["split_byod"](records, rows, seed=0)
    assert rule == "by group (6 groups)"
    role_of: dict[str, set[str]] = {}
    for role, part in splits.items():
        for record in part:
            role_of.setdefault(record["region"], set()).add(role)
    assert all(len(r) == 1 for r in role_of.values()) and len(role_of) == 6, role_of
    assert len(splits["train"]) >= MIN_RECORDS and splits["validation"] and splits["test"]
    rows_missing = [dict(r) for r in rows]
    rows_missing[0]["group"] = ""
    with pytest.raises(ValueError, match="give every chip a group or remove the column"):
        ns["split_byod"](records, rows_missing, seed=0)
    two = [{**r, "group": "a" if i % 2 else "b"} for i, r in enumerate(rows)]
    with pytest.raises(ValueError, match="2 groups; a group split needs at least 3"):
        ns["split_byod"](records, two, seed=0)


def test_fl_m3_split_column_sets_the_roles_and_no_column_warns(notebook, tmp_path):
    ns = _helpers(notebook)
    roles = ["train"] * 4 + ["validation", "test"]
    _write_chips(tmp_path / "s", 6, extra={"split": roles})
    records, rows = _records_and_rows(tmp_path / "s")
    assert len(ns["read_byod_table"](tmp_path / "s")) == 6  # a split column may go below the per-chip minimum
    splits, rule = ns["split_byod"](records, rows, seed=0)
    assert rule == "by the split column"
    assert {role: [r["id"] for r in part] for role, part in splits.items()} == {"train": ["chip0", "chip1", "chip2", "chip3"], "validation": ["chip4"], "test": ["chip5"]}
    bad = [{**r, "split": "holdout" if r["id"] == "chip5" else r["split"]} for r in rows]
    with pytest.raises(ValueError, match="split 'holdout' must be one of train, validation, test"):
        ns["split_byod"](records, bad, seed=0)
    _write_chips(tmp_path / "plain", 8)
    records, rows = _records_and_rows(tmp_path / "plain")
    ns["printed"].clear()
    splits, rule = ns["split_byod"](records, rows, seed=0)
    assert rule.startswith("per chip") and "no group or split column" in ns["printed"][0]["warning"]


# --- FL-M2: the learner-facing expected results are the current record's, not the pre-fix run's ----------------------


def test_fl_m2_no_pre_fix_expected_results_remain(notebook):
    md = _markdown(notebook)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in (md, readme):
        for stale in ("about 0.59", "0.59 to 0.60", "IoU near 0.6 ", "recall above precision", "recall above its precision", "seven regions", "over-predicts water"):
            assert stale not in text, stale
    assert "ten regions" in md and "precision 0.9251 is above recall 0.776" in md
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    for value in ("0.7301", "0.7458", "0.8049", "0.1306", "0.1178", "0.8614", "0.8748"):  # quoted answers = the record row
        assert value in md and value in record, value
    regions = {name.split("_")[0] for name, role, *_ in SAMPLE_RECORDS if role == "test"}
    assert len(regions) == 10


# --- FL-M3: no tutorial cell indexes a sample-only key -----------------------------------------------------------------


def test_fl_m3_no_cell_indexes_a_sample_only_key(notebook):
    assert [s for s in _code(notebook) if "record['source_id']" in s or 'record["source_id"]' in s] == []
    assert "'chip': record.get('source_id', record['id'])" in _cell(notebook, "frozen_test = pipe.evaluate(test_records)")


def test_fl_m3_section_4_clears_this_notebooks_earlier_exports(notebook, tmp_path, monkeypatch):
    import os
    import shutil

    source = _cell(notebook, "if USE_BYOD:")
    block = source[source.index("os.makedirs('outputs'") : source.index("def byod_minimum_records(")]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / f"{STEM}_adapter").mkdir(parents=True)
    (tmp_path / "outputs" / f"{STEM}_adapter" / "adapter.safetensors").write_bytes(b"x")
    (tmp_path / "outputs" / f"{STEM}_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / "someone_else.txt").write_text("keep", encoding="utf-8")
    exec(compile(block, "section4-clear", "exec"), {"os": os, "Path": Path, "shutil": shutil})
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == ["someone_else.txt"]


# --- FL-m2, FL-m4, FL-m5, FL-m6, FL-M5 activity, FL-S1: text and structure ---------------------------------------------


def test_fl_m2_section_8_names_the_provenance_of_each_example_chip(notebook):
    roles = {name: role for name, role, *_ in SAMPLE_RECORDS}
    assert roles.get("USA_430764") == "test" and "India_900498" not in roles and "Spain_7370579" not in roles
    md = _markdown(notebook)
    assert "`USA_430764` is also one of this notebook's 12 test chips" in md and "upstream hand-labelled **test** split" in md
    assert "which carry no labels here" not in md
    code = _cell(notebook, "new_predictions = pipe.predict(new_records)")
    assert "'also_a_sample_chip': sample_role.get(record['id'].removesuffix('_S2Hand'))" in code


def test_fl_m4_interpretation_and_activity(notebook):
    md = _markdown(notebook)
    closing = md[md.index("## Interpretation and limits") :]
    assert "recall rose from 0.776 to 0.8002" in closing and "precision fell from 0.9251 to 0.9165" in closing
    assert "leaves those numbers about" not in closing and "to see the frozen model win every epoch" not in md
    assert "## Optional activity: Predict → Change → Run → Observe → Explain" in md and "**Scope of a re-run:**" in md
    assert "**Run** Sections 6, 7 and 8 in that order" in md
    assert "{id, image, label}" in md and "{{id" not in md


def test_fl_m5_section_7_draws_the_mask_figure(notebook):
    code = _cell(notebook, "delta_water_iou = ")
    assert "import matplotlib.pyplot as plt" in code and "plt.subplots(" in code and "plt.show()" in code
    assert "frozen_predictions['predictions']" in code


def test_fl_m6_threshold_and_calibration_ownership_is_stated(notebook):
    md = _markdown(notebook)
    section5 = md[md.index("## 5. The frozen model") : md.index("## 6. Bounded fine-tuning")]
    assert "no threshold is tuned here" in section5 and "calibrating the scores belong to whoever deploys the model" in section5
    assert "the notebook says the threshold and its cost trade-off are the deployment's" in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")


def test_fl_s1_declares_notebook_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert "DIMER Notebook Specification 2.2 — **standalone** (§4)" in _markdown(notebook)


# --- FL-m7: the recorded source revision holds exactly the carried modules -------------------------------------------


def test_fl_m7_recorded_revision_contains_the_carried_modules(notebook):
    generated = notebook["metadata"]["dimer"]["generated_from"]
    revision = generated["revision"]
    try:
        present = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{revision}^{{commit}}"], capture_output=True).returncode == 0
    except OSError:
        present = False
    if not present:
        pytest.skip(f"revision {revision[:12]} is not in this clone (shallow checkout)")
    for module in generated["modules"]:
        recorded = subprocess.run(["git", "-C", str(ROOT), "show", f"{revision}:{module}"], capture_output=True, check=True).stdout
        assert recorded.replace(b"\r\n", b"\n") == (ROOT / module).read_bytes().replace(b"\r\n", b"\n"), module


# --- End to end on the stub model: default run, then the documented BYOD re-run from Section 4 (FL-M3, FL-M4) --------


def _stub_model_class(torch):
    class StubModel(torch.nn.Module):
        """tests/test_adaptation.py's stub: terratorch parameter names; logits depend on the decoder/head tensors."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(24)])
            for block in self.encoder.blocks:
                block.w = torch.nn.Parameter(torch.ones(1))
            self.neck = torch.nn.Module()
            self.neck.scale = torch.nn.Parameter(torch.ones(1))
            self.decoder = torch.nn.Module()
            self.decoder.weight = torch.nn.Parameter(torch.zeros(6))
            self.head = torch.nn.Module()
            self.head.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x):
            feature = (x * self.decoder.weight[None, :, None, None]).sum(dim=1) * self.neck.scale * self.encoder.blocks[23].w
            return types.SimpleNamespace(output=torch.stack([-feature + self.head.bias[0], feature + self.head.bias[1]], dim=1))

    return StubModel


def test_fl_m3_m4_byod_rerun_from_section_4_after_a_default_run(notebook, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    namespace: dict = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell.get("metadata", {}).get("dimer", {}).get("embedded_module"):
            exec(compile(_source(cell), "carried-module", "exec"), namespace)
    pfp = namespace["PrithviFloodPipeline"]
    stub_model = _stub_model_class(torch)
    weights = tmp_path / "weights"
    (weights / "examples").mkdir(parents=True)
    for k, name in enumerate(("India_900498", "USA_430764")):
        image, _ = synthetic_chip(seed=80 + k)
        wide = np.zeros((13, 512, 512), dtype=np.float32)
        wide[list(namespace["S2_L1C_BAND_INDICES"])] = image * 10000.0
        tifffile.imwrite(weights / "examples" / f"{name}_S2Hand.tif", wide, photometric="minisblack", planarconfig="separate")

    def stub_pipe():
        return pfp(model=stub_model(), device="cpu", weights_dir=weights, source="stub")

    monkeypatch.setattr(pfp, "from_pretrained", classmethod(lambda cls, **kw: stub_pipe()))
    sample: dict[str, list] = {"train": [], "validation": [], "test": []}
    k = 0
    for role, n in (("train", 4), ("validation", 2), ("test", 2)):
        for j in range(n):
            image, label = synthetic_chip(seed=100 + k)
            name = ("USA_430764", "Ghana_1")[j] if role == "test" else f"Region{k}_{k}"
            sample[role].append(namespace["check_record"]({"id": f"{role}-{j:03d}", "source_id": name, "region": name.split("_")[0], "split": role, "image": image, "label": label, "source": "synthetic"}))
            k += 1
    for fake in ("timm", "lightning"):
        monkeypatch.setitem(sys.modules, fake, types.SimpleNamespace(__version__="stand-in"))
    pipe = stub_pipe()
    namespace.update(
        pipe=pipe, WEIGHTS_DIR=weights, torch=torch, timm=sys.modules["timm"], lightning=sys.modules["lightning"],
        fetch_sample_dataset=lambda **kw: {r: list(v) for r, v in sample.items()},
        verify_converted=lambda *a, **k: {"files": [{"path": "stand-in"}]}, MANIFEST={"files": []},
        NOTEBOOK_SOURCE={"repository_revision": "stand-in"},
    )
    markers = ("if USE_BYOD:", "frozen_test = pipe.evaluate(test_records)", "adapt_result = pipe.adapt(", "delta_water_iou = ", "new_predictions = pipe.predict(new_records)")
    cells = [_cell(notebook, m).replace("importlib.metadata.version('terratorch')", "'stand-in'") for m in markers]
    cells[2] = cells[2].replace("EPOCHS = 4  #", "EPOCHS = 2  #").replace("LEARNING_RATE = 1e-5  #", "LEARNING_RATE = 1e-2  #")
    monkeypatch.chdir(tmp_path)
    out: list[str] = []
    namespace["print"] = lambda *a, **k: out.append(" ".join(map(str, a)))

    def run(sources):
        for index, source in enumerate(sources):
            exec(compile(source, f"section-{index + 4}", "exec"), namespace)

    run(cells)  # the default path, adapting the stub away from its base
    assert any("'split_rule': 'the official Sen1Floods11 splits'" in line for line in out)
    assert any("'chip': 'USA_430764'" in line for line in out)  # Section 5 per-chip line on the sample
    assert any("'chip': 'USA_430764_S2Hand'" in line and "'also_a_sample_chip': 'test'" in line for line in out)
    assert any("'chip': 'India_900498_S2Hand'" in line and "'also_a_sample_chip': None" in line for line in out)
    assert float(pipe.model.decoder.weight.abs().sum()) > 0  # the default run moved the stub off its base
    sample_result = json.loads((tmp_path / "outputs" / f"{STEM}_result.json").read_text(encoding="utf-8"))
    assert sample_result["data_source"] == namespace["SAMPLE_LABEL_SOURCE"]

    byod_dir = tmp_path / "my_chips"
    _write_chips(byod_dir, 8)
    archive = _zip(byod_dir, tmp_path / "byod8.zip")
    section4 = cells[0].replace("USE_BYOD = False  #", "USE_BYOD = True  #").replace("BYOD_PATH = ''  #", f"BYOD_PATH = {str(archive)!r}  #")
    out.clear()
    run([section4, *cells[1:]])  # FL-M3: the documented re-run from Section 4 completes (no KeyError 'source_id')

    fresh = stub_pipe()
    test_records, val_records, train_records = namespace["test_records"], namespace["val_records"], namespace["train_records"]
    assert namespace["frozen_test"]["model"] == fresh.evaluate(test_records)["model"]  # FL-M4: frozen = the packaged base
    assert namespace["frozen_val"]["model"] == fresh.evaluate(val_records)["model"]
    fresh_epoch0 = fresh.adapt(train_records, val_records, epochs=1, lr=1e-2, batch_size=2, trainable="decoder")["history"][0]
    assert namespace["adapt_result"]["history"][0]["val_loss"] == fresh_epoch0["val_loss"]
    assert any("'restored_pinned_base': 3" in line for line in out)  # Section 5 put the adapted tensors back first
    assert any("no group or split column" in line for line in out)
    outputs = tmp_path / "outputs"
    for name in (f"{STEM}_result.json", f"{STEM}_evaluation_report.json"):
        assert json.loads((outputs / name).read_text(encoding="utf-8"))["data_source"] == "BYOD (byod8.zip)"
    assert "BYOD (byod8.zip)" in (outputs / f"{STEM}_adapter" / "manifest.json").read_text(encoding="utf-8")
    assert any("'reload_parity'" in line for line in out)

    # FL-M4: decoder+last_block, then decoder: the second run starts from the pinned base, so the last block is the base's
    run([cells[2].replace("TRAINABLE = 'decoder'  #", "TRAINABLE = 'decoder+last_block'  #")])
    assert float(pipe.model.encoder.blocks[23].w) != 1.0
    run(cells[2:])  # Sections 6-8 with the default 'decoder': the notebook's own reload-parity assert must hold
    assert float(pipe.model.encoder.blocks[23].w) == 1.0
    assert not [n for n in pipe.adapter["trainable_names"] if n.startswith("encoder.")]
