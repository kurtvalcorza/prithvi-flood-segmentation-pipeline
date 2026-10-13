"""Philippines flood-mapping capstone (WORKSHOP, NOTEBOOK_SPEC 2.2): carried-source integrity, bootstrap fixes,
negative controls and CPU checks of the carried stage runner (capstone specification §12).

The runner checks use small synthetic arrays; they test the scoring and bookkeeping contract, not model skill.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_release_assets.py"
CAPSTONE = "DIMER_Philippines_Flood_Mapping_Capstone.ipynb"


def _load_validator(root: Path):
    spec = importlib.util.spec_from_file_location("validate_release_assets_capstone", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


def _notebook(root: Path = ROOT) -> dict:
    return json.loads((root / "tutorials" / CAPSTONE).read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _carried_cell(notebook: dict) -> dict:
    return next(c for c in notebook["cells"] if "CARRIED_FILES = " in _source(c))


def _carried(notebook: dict) -> tuple[dict, dict]:
    body = ast.parse(_source(_carried_cell(notebook))).body
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in body if isinstance(n, ast.Assign)}
    return literals["CARRIED_FILES"], literals["CARRIED_HASHES"]


def _install_cell(notebook: dict) -> str:
    return next(_source(c) for c in notebook["cells"] if "UV_URL = " in _source(c))


def _edit(root: Path, mutate) -> None:
    path = root / "tutorials" / CAPSTONE
    notebook = json.loads(path.read_text(encoding="utf-8"))
    mutate(notebook)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("docs", "tutorials", "src"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = "weights/prithvi-eo-2.0-300m-tl-sen1floods11/dimer-base-manifest.json"
    (tmp_path / manifest).parent.mkdir(parents=True)
    shutil.copy2(ROOT / manifest, tmp_path / manifest)
    shutil.copy2(ROOT / "LICENSE", tmp_path / "LICENSE")
    return tmp_path


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    """Import the carried capstone.py exactly as the notebook writes it."""
    pytest.importorskip("rasterio")
    files, _ = _carried(_notebook())
    folder = tmp_path_factory.mktemp("capstone")
    path = folder / "capstone.py"
    path.write_text(files["capstone.py"], encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("carried_capstone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["carried_capstone"] = module
    spec.loader.exec_module(module)
    return module


# --- notebook integrity -------------------------------------------------------------------------------------------


def test_committed_capstone_passes() -> None:
    _load_validator(ROOT).validate_workshop_notebooks()


def test_carried_files_verify_and_pin_two_scenes() -> None:
    files, hashes = _carried(_notebook())
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()} == hashes
    data = json.loads(files["data_manifest.json"])
    assert [(r["id"], r["role"]) for r in data["records"]] == [
        ("EMSR312_08CANDON_DEL_MONIT01_v1", "development"),
        ("EMSR312_07VIGAN_DEL_MONIT01_v1", "heldout"),
    ]
    assert data["processing_level"] == "L1C" and data["scale"] == 0.0001
    assert data["licence"] == "CC-BY-NC-4.0"


def test_control_tampered_carried_file_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        cell = _carried_cell(notebook)
        text = _source(cell)
        body = ast.parse(text).body
        files = ast.literal_eval(body[0].value)
        files["capstone.py"] = files["capstone.py"].replace("SIZE = 512", "SIZE = 256", 1)
        cell["source"] = text.replace(ast.get_source_segment(text, body[0].value), repr(files), 1)

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match="CARRIED_HASHES digest"):
        module.validate_workshop_notebooks()


def test_control_package_drift_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    metrics = tree / "src" / "prithvi_flood_segmentation_pipeline" / "metrics.py"
    metrics.write_text(metrics.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(module.ValidationError, match="differs from src/prithvi_flood_segmentation_pipeline/metrics.py"):
        module.validate_workshop_notebooks()


def test_control_persisted_output_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _edit(tree, lambda notebook: _carried_cell(notebook).__setitem__("execution_count", 3))
    with pytest.raises(module.ValidationError, match="persists outputs"):
        module.validate_workshop_notebooks()


@pytest.mark.parametrize("key", ["notebook_profile", "notebook_mode"])
def test_control_metadata_key_is_required(tree: Path, key: str) -> None:
    module = _load_validator(tree)
    _edit(tree, lambda notebook: notebook["metadata"]["dimer"].pop(key))
    with pytest.raises(module.ValidationError, match=key):
        module.validate_workshop_notebooks()


def test_control_uncited_reference_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        for cell in notebook["cells"]:
            body, sep, refs = _source(cell).partition("## References")
            cell["source"] = body.replace("[14]", "(the spatial-validation literature)") + sep + refs

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match=r"reference \[14\] is never cited"):
        module.validate_workshop_notebooks()


def test_control_citation_without_reference_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        cell = next(c for c in notebook["cells"] if "## 4. Run frozen Prithvi" in _source(c))
        cell["source"] = _source(cell) + "\n\nSegmentation transformers are widely used [99].\n"

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match=r"citation \[99\] has no reference entry"):
        module.validate_workshop_notebooks()


def test_control_author_year_citation_is_rejected_in_the_ieee_notebook(tree: Path) -> None:
    # Notebook Spec 2.5 DOC14: one reference style throughout.
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        cell = next(c for c in notebook["cells"] if "## 4. Run frozen Prithvi" in _source(c))
        cell["source"] = _source(cell) + "\n\nSegmentation transformers are widely used (Nobody et al., 2024).\n"

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match="mixes author-year citations"):
        module.validate_workshop_notebooks()


def test_infrastructure_cells_are_labelled_and_collapsed() -> None:
    notebook = _notebook()
    infrastructure = [c for c in notebook["cells"] if _source(c).startswith("# @title Infrastructure:")]
    assert len(infrastructure) == 3
    assert all(c["metadata"].get("cellView") == "form" for c in infrastructure)
    learner = [c for c in notebook["cells"] if c["cell_type"] == "code" and c not in infrastructure]
    assert all(_source(c).startswith("run_stage(") for c in learner)


# --- bootstrap fixes found in review --------------------------------------------------------------------------------


def test_isolated_environment_forces_a_file_backend_for_matplotlib() -> None:
    # Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline, which the isolated environment cannot import.
    install = _install_cell(_notebook())
    assert install.index("ENV['MPLBACKEND'] = 'Agg'") < install.index("subprocess.run([str(UV), 'venv'")


def test_sdist_only_pin_is_built_from_the_lock_before_the_wheel_only_install() -> None:
    # antlr4-python3-runtime 4.9.3 exists on PyPI only as a source archive; `--only-binary :all:` alone cannot
    # install the lock. The notebook builds it first with the lock's own setuptools, then installs wheels only.
    notebook = _notebook()
    files, _ = _carried(notebook)
    install = _install_cell(notebook)
    namespace: dict = {}
    start, end = install.index("def lock_entry"), install.index("PIP = ")
    exec(install[start:end], namespace)  # the notebook's own helper, run on the carried lock
    namespace["LOCK"] = files["requirements.txt"]
    antlr = namespace["lock_entry"]("antlr4-python3-runtime")
    setuptools = namespace["lock_entry"]("setuptools")
    assert antlr.startswith("antlr4-python3-runtime==4.9.3 \\\n") and antlr.count("--hash=sha256:") == 1
    assert setuptools.startswith("setuptools==") and setuptools.count("--hash=sha256:") >= 1
    wheels_only = "'--only-binary', ':all:', '-r', str(ROOT / 'requirements.txt')"
    assert install.index("'--no-build-isolation'") < install.index(wheels_only)


# --- carried runner: scoring and bookkeeping contract -----------------------------------------------------------------


def test_label_mapping_keeps_only_clear_labelled_pixels(runner) -> None:
    cloud = np.array([[1, 1, 2, 0]])
    water = np.array([[1, 2, 2, 1]])
    mapped = runner.map_labels(np.stack([cloud, water]))
    assert mapped.tolist() == [[0, 1, 255, 255]]
    with pytest.raises(ValueError, match="encoded 0,1,2"):
        runner.map_labels(np.stack([cloud, water + 1]))


def test_reflectance_is_converted_once_and_bounded(runner) -> None:
    scaled = runner.reflectance(np.array([[10000.0, 5000.0]]))
    assert scaled.tolist() == [[1.0, 0.5]]
    assert runner.reflectance(scaled, units="reflectance").tolist() == [[1.0, 0.5]]
    with pytest.raises(ValueError, match=r"\[0,2\]"):
        runner.reflectance(scaled * 10000, units="reflectance")
    with pytest.raises(ValueError, match="Unknown reflectance units"):
        runner.reflectance(scaled, units="percent")


def test_mndwi_excludes_near_zero_denominators(runner) -> None:
    image = np.zeros((6, 1, 3), dtype=np.float32)
    image[1] = [[0.3, 0.0, 0.1]]  # green
    image[4] = [[0.1, 0.0, 0.3]]  # SWIR1
    index, valid = runner.mndwi(image)
    assert valid.tolist() == [[True, False, True]]
    assert index[0, 0] == pytest.approx(0.5) and index[0, 2] == pytest.approx(-0.5) and index[0, 1] == -9999


def test_threshold_selection_breaks_ties_towards_zero_then_smaller(runner) -> None:
    rows = [{"threshold": t, "water_iou": iou} for t, iou in ((-0.1, 0.9), (0.1, 0.9), (0.2, 0.9), (0.0, 0.5))]
    assert runner.select_threshold(rows) == -0.1
    rows.append({"threshold": 0.05, "water_iou": 0.9})
    assert runner.select_threshold(rows) == 0.05
    with pytest.raises(ValueError, match="No defined development IoU"):
        runner.select_threshold([{"threshold": 0.0, "water_iou": None}])


def test_no_water_metrics_report_undefined_precision_not_a_score(runner) -> None:
    target = np.array([[1, 1, 0, 255]], dtype=np.uint8)
    values = runner.metrics(runner.confusion(target, np.zeros_like(target)))
    assert values["precision"] is None and "precision" in values["undefined_reasons"]
    assert values["recall"] == 0 and values["f1"] == 0 and values["water_iou"] == 0
    assert values["valid_pixels"] == 3 and values["accuracy"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="Invalid prediction"):
        runner.confusion(target, np.array([[2, 0, 0, 0]], dtype=np.uint8))


def test_windows_tile_the_grid_once_and_padding_is_cropped(runner) -> None:
    seen = np.zeros((700, 1100), dtype=int)
    for window in runner.windows(700, 1100):
        seen[window.row_off : window.row_off + window.height, window.col_off : window.col_off + window.width] += 1
    assert (seen == 1).all()
    chip = np.arange(6 * 3 * 2, dtype=np.float32).reshape(6, 3, 2)
    padded = runner.pad_chip(chip)
    assert padded.shape == (6, 512, 512) and np.array_equal(padded[:, :3, :2], chip)
    with pytest.raises(ValueError, match="six-band"):
        runner.pad_chip(chip[:5])


def test_manifest_rejects_one_optical_scene_in_two_roles(runner) -> None:
    files, _ = _carried(_notebook())
    manifest = json.loads(files["data_manifest.json"])
    runner.validate_manifest(manifest)
    duplicate = json.loads(files["data_manifest.json"])
    duplicate["records"][1]["assets"]["image"]["sha256"] = duplicate["records"][0]["assets"]["image"]["sha256"]
    with pytest.raises(ValueError, match="Duplicate optical scene"):
        runner.validate_manifest(duplicate)


def test_reload_parity_fails_on_changed_decisions(runner) -> None:
    probe = np.array([[0.2, 0.7]], dtype=np.float32)
    assert runner.parity(probe, probe.copy())["identical_binary_predictions"]
    with pytest.raises(ValueError, match="parity failed"):
        runner.parity(probe, np.array([[0.2, 0.4]], dtype=np.float32))


def test_csv_and_geotiff_round_trips(runner, tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    runner.write_csv(tmp_path / "t.csv", [{"a": 1, "b": None, "c": {"k": [1, 2]}}])
    profile = dict(driver="GTiff", height=4, width=5, count=1, dtype="uint8", crs="EPSG:32651",
                   transform=from_origin(300000, 2000000, 10, 10), nodata=255)
    data = np.array([[0, 1, 255, 1, 0]] * 4, dtype=np.uint8)
    with rasterio.open(tmp_path / "a.tif", "w", **profile) as dst:
        dst.write(data, 1)
    with rasterio.open(tmp_path / "a.tif") as src, rasterio.open(tmp_path / "a.tif") as other:
        assert runner.same_grid(src, other) and np.array_equal(src.read(1), data) and src.nodata == 255


def test_changed_inputs_invalidate_downstream_receipts(runner, tmp_path: Path) -> None:
    for name in ("data_manifest.json", "requirements.txt", "source.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    product = tmp_path / "outputs" / "x.json"
    product.parent.mkdir()
    product.write_text("1", encoding="utf-8")
    stamp = runner.begin_stage(tmp_path, "prepare")
    runner.finish_stage(tmp_path, "prepare", stamp, [product], 0.1)
    runner.verify_receipt(tmp_path, "prepare")
    (tmp_path / "data_manifest.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Stale prepare receipt"):
        runner.verify_receipt(tmp_path, "prepare")


# --- 2026-09-28 review fixes (F01-F05) ------------------------------------------------------------------------------


def _markdown(notebook: dict) -> list[str]:
    return [_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown"]


def _helpers(monkeypatch, root: Path) -> tuple[dict, list]:
    """The notebook's own display helpers, executed with a recording stand-in for IPython.display."""
    import types

    shown: list = []
    display = types.ModuleType("IPython.display")
    display.display = shown.append
    display.Markdown = lambda text: ("markdown", text)
    display.Image = lambda filename: ("image", Path(filename).name)
    display.FileLink = lambda path: ("link", path)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    install = _install_cell(_notebook())
    namespace: dict = {"ROOT": root, "json": json}
    exec(install[install.index("import csv") :], namespace)
    return namespace, shown


def test_threshold_answer_keys_do_not_promise_monotonic_precision() -> None:
    # F01: raising a strict threshold removes predicted water, so area and recall cannot rise, but TP/(TP+FP) can move
    # either way. F05: the interior optimum and the held-out wording must not overclaim.
    markdown = "\n".join(_markdown(_notebook()))
    for stale in ("precision rises", "the range was wide enough", "Vigan has not been touched"):
        assert stale not in markdown
    assert markdown.count("Precision may rise, fall or stay unchanged") == 2
    assert "cannot increase predicted water area or recall" in markdown
    assert "In the recorded Candon run, precision rose with the threshold; that is an observation" in markdown
    assert "Vigan performance has not been used to select any threshold" in markdown
    assert "**Declared inspection:**" in markdown and "the optimum among the tested candidates is not at a boundary" in markdown


@pytest.mark.parametrize(
    ("labels", "precision_direction"),
    [((1, 1, 0), "falls"), ((0, 1, 1), "rises")],
)
def test_raising_the_threshold_never_raises_recall_but_precision_can_move_either_way(runner, labels, precision_direction):
    # The review's P08 fixture: evaluated scores 0.4 / 0.6 / 0.8 plus one excluded pixel, scored as the activity does.
    target = np.array([[*labels, 255]], dtype=np.uint8)
    probability = np.array([[0.4, 0.6, 0.8, -9999]], dtype=np.float32)
    rows = [runner.metrics(runner.confusion(target, probability > t)) for t in (0.3, 0.5, 0.7)]
    predicted = [r["tp"] + r["fp"] for r in rows]
    recall = [r["recall"] for r in rows]
    precision = [r["precision"] for r in rows]
    assert predicted == sorted(predicted, reverse=True) and recall == sorted(recall, reverse=True)
    assert all(r["valid_pixels"] == 3 for r in rows)
    if precision_direction == "falls":
        assert precision == pytest.approx([2 / 3, 0.5, 0.0])
    else:
        assert precision == pytest.approx([2 / 3, 1.0, 1.0])


def _write_raster(path: Path, array: np.ndarray, dtype: str, nodata=None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    array = array if array.ndim == 3 else array[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2], count=array.shape[0],
                       dtype=dtype, crs="EPSG:32651", transform=from_origin(300000, 2000000, 10, 10), nodata=nodata) as dst:
        dst.write(array.astype(dtype))


def test_activity_reports_falling_precision_and_leaves_canonical_products_unchanged(runner, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    record = {"id": "SYN_DEV", "role": "development", "assets": {}}
    manifest = {"records": [record]}
    _write_raster(runner.scene_output(tmp_path, record, "reference"), np.array([[1, 1, 0, 255]]), "uint8", 255)
    _write_raster(runner.scene_output(tmp_path, record, "probability"), np.array([[0.4, 0.6, 0.8, -9999]]), "float32", -9999)
    runner.write_json(tmp_path / "outputs" / "selection.json", {"selected_threshold": 0.15, "canonical_prithvi_threshold": 0.5})
    (tmp_path / "outputs" / "figures").mkdir()
    watched = [runner.scene_output(tmp_path, record, k) for k in ("reference", "probability")]
    watched.append(tmp_path / "outputs" / "selection.json")
    before = {p: runner.sha256(p) for p in watched}
    runner.activity(tmp_path, manifest)
    import csv

    with (tmp_path / "outputs" / "activity.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(r["precision"]) for r in rows] == pytest.approx([2 / 3, 0.5, 0.0])
    assert [float(r["recall"]) for r in rows] == pytest.approx([1.0, 0.5, 0.0])
    assert {p: runner.sha256(p) for p in watched} == before
    assert runner.read_json(tmp_path / "outputs" / "selection.json")["canonical_prithvi_threshold"] == 0.5


def test_diagnostic_figures_are_shown_in_semantic_order_not_filename_order(monkeypatch, tmp_path: Path) -> None:
    # F03: sorted filenames would put fixed_location and the error windows before the whole-area score maps.
    namespace, shown = _helpers(monkeypatch, tmp_path)
    figures = tmp_path / "outputs" / "figures"
    figures.mkdir(parents=True)
    for name in ("fixed_location", "highest_false_negative", "highest_reference_disagreement", "probability_disagreement"):
        (figures / f"development_{name}.png").write_bytes(b"png")
    namespace["show_diagnostics"]("development")
    images = [item[1] for item in shown if item[0] == "image"]
    assert images == [
        "development_probability_disagreement.png",
        "development_fixed_location.png",
        "development_highest_reference_disagreement.png",
        "development_highest_false_negative.png",
    ]
    captions = [item[1] for item in shown if item[0] == "markdown"]
    assert captions[0].startswith("**Whole area:") and all("Error window" in c for c in captions[1:])
    (figures / "development_fixed_location.png").unlink()
    with pytest.raises(RuntimeError, match="fixed_location"):
        namespace["show_diagnostics"]("development")


def test_verification_display_expands_reload_details(monkeypatch, tmp_path: Path) -> None:
    # F04: the nested reload record (score difference, decision parity) is shown, not only scalar checks.
    namespace, shown = _helpers(monkeypatch, tmp_path)
    reload = {"probability_allclose": True, "max_absolute_error": 3.2e-7, "identical_binary_predictions": True}
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "verification.json").write_text(json.dumps({"csv_round_trips": "passed", "reload": reload}))
    namespace["show_record"]("verification.json")
    table = shown[0][1]
    assert "| reload: max_absolute_error | 3.2e-07 |" in table
    assert "| reload: identical_binary_predictions | True |" in table and "| csv_round_trips | passed |" in table


def test_candidate_inundation_is_surfaced_with_its_area_and_limits(monkeypatch, tmp_path: Path) -> None:
    namespace, shown = _helpers(monkeypatch, tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for scene, role, area in (("EMSR312_07VIGAN_X", "heldout", 14.7), ("EMSR312_08CANDON_X", "development", 1.69)):
        (outputs / f"{scene}_candidate_inundation.json").write_text(json.dumps({
            "scene": scene, "role": role, "meaning": "m", "candidate_inundation_km2": area, "confirmed_flood": False,
            "limitations": "not pre-event evidence"}))
    namespace["show_candidates"]()
    table = shown[0][1]
    assert table.index("08CANDON") < table.index("07VIGAN")
    assert "| 08CANDON | development | 1.6900 | no |" in table and "| 07VIGAN | heldout | 14.7000 | no |" in table
    assert [item[1] for item in shown[1:]] == ["development_candidate_inundation.png", "heldout_candidate_inundation.png"]


def test_candidate_inundation_classes_sidecar_and_figure(runner, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    record = {"id": "SYN_DEV", "role": "development", "assets": {"permanent": {"path": "SYN/permanent.tif"}}}
    _write_raster(tmp_path / "cache" / "SYN" / "permanent.tif", np.array([[0, 1, 2, 3, 1]]), "uint8")
    _write_raster(runner.scene_output(tmp_path, record, "prithvi"), np.array([[1, 1, 1, 1, 255]]), "uint8", 255)
    products = runner.candidate_inundation(tmp_path, {"permanent_water": {"verified": True}}, record)
    import rasterio

    with rasterio.open(products[0]) as source:
        assert source.read(1).tolist() == [[255, 1, 1, 0, 255]]
    sidecar = runner.read_json(products[1])
    assert sidecar["confirmed_flood"] is False and (sidecar["scene"], sidecar["role"]) == ("SYN_DEV", "development")
    assert sidecar["candidate_inundation_km2"] == pytest.approx(2 * 100 / 1e6)
    assert products[2].name == "development_candidate_inundation.png" and products[2].stat().st_size > 0


def test_error_windows_show_the_optical_image_of_the_same_window(runner, tmp_path: Path) -> None:
    # F02: each error window now reads the source image for true- and false-colour crops and a locator.
    pytest.importorskip("matplotlib")
    record = {"id": "SYN_DEV", "role": "development", "assets": {"image": {"path": "SYN/image.tif"}}}
    reference = np.array([[0, 1, 1, 0, 255]] * 4)
    _write_raster(runner.scene_output(tmp_path, record, "reference"), reference, "uint8", 255)
    _write_raster(runner.scene_output(tmp_path, record, "prithvi"), np.array([[1, 1, 0, 0, 255]] * 4), "uint8", 255)
    _write_raster(runner.scene_output(tmp_path, record, "mndwi"), np.array([[-0.2, 0.4, 0.4, -0.3, -9999]] * 4), "float32", -9999)
    runner.write_json(tmp_path / "outputs" / "selection.json", {"selected_threshold": 0.15})
    (tmp_path / "outputs" / "figures").mkdir()  # created by the prepare stage in the notebook
    rows = [{"scene": "SYN_DEV", "method": "prithvi", "row": 0, "col": 0, "valid_pixels": 16, "fp": 4, "fn": 4}]
    with pytest.raises(Exception, match="image.tif"):
        runner.diagnostic_figures(tmp_path, record, rows)
    _write_raster(tmp_path / "cache" / "SYN" / "image.tif", np.full((15, 4, 5), 1200), "uint16")
    paths = runner.diagnostic_figures(tmp_path, record, rows)
    assert [p.name for p in paths] == [
        "development_fixed_location.png",
        "development_highest_reference_disagreement.png",
        "development_highest_false_positive.png",
        "development_highest_false_negative.png",
    ]
    assert runner.error_classes(reference, np.array([[1, 1, 0, 0, 0]] * 4))[0].tolist() == [2, 1, 3, 0, 4]


def _load_splitter():
    spec = importlib.util.spec_from_file_location("split_capstone_carrier", ROOT / "tools" / "split_capstone_carrier.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_capstone_cell_has_a_line_over_2000_characters() -> None:
    # Colab became unresponsive on a notebook whose carrier was one very long line; the carrier is written in pieces.
    splitter = _load_splitter()
    assert splitter.MAX_LINE == 2000
    assert splitter.long_lines(_notebook()) == []
    for index, cell in enumerate(_notebook()["cells"]):
        assert max(len(line) for line in _source(cell).split("\n")) <= 2000, f"cell {index}"


def test_carrier_splitter_round_trips_exactly() -> None:
    splitter = _load_splitter()
    files = {
        "empty.txt": "",
        "long.txt": "x" * 2500 + "\n",
        "multi.py": "a = 1\r\nb = 'two'\n\n\"\"\"q\"\"\"\tend",
        "é.md": "café\n",
    }
    literal = splitter.carried_literal(files)
    assert ast.literal_eval(literal) == files
    assert max(len(line) for line in literal.split("\n")) <= splitter.CARRIER_PIECE + 20
    source = f"# head\nCARRIED_FILES = {files!r}\nCARRIED_HASHES = {{}}\nprint('ok')\n"
    split = splitter.split_carrier(source)
    tree = ast.parse(split).body
    assert ast.literal_eval(tree[0].value) == files
    assert split.startswith("# head\nCARRIED_FILES = {\n") and split.endswith("}\nCARRIED_HASHES = {}\nprint('ok')\n")
    assert splitter.split_carrier(split) == split


# --- 2026-10-12 review fixes (FLD-M1, FLD-M2, FLD-m1, FLD-m2, FLD-m4) ------------------------------------------------


def test_comparison_scope_states_unequal_tuning_and_competing_explanations() -> None:
    # Wording/disclosure test: it checks the scope text is present, not that any causal explanation is correct.
    markdown = "\n".join(_markdown(_notebook()))
    assert "In the primary comparison MNDWI and Prithvi are **not equally tuned**" in markdown
    assert "Prithvi's primary row runs at a fixed default of 0.5 that is not selected on Candon" in markdown
    assert "The declared secondary row gives Prithvi the same kind of Candon tuning" in markdown
    assert "The secondary row tests the first one" in markdown
    assert "as a declared secondary row, Prithvi at its Candon-selected threshold of [t] achieved" in markdown
    assert "cannot isolate either approach's intrinsic transfer ability or establish broader transfer performance" in markdown
    for stale in ("sensitivity to geography", "examine geography", "does not show which approach transfers better"):
        assert stale not in markdown
    conclusion = _carried(_notebook())[0]["capstone.py"]
    assert "as a declared secondary row, Prithvi at its Candon-selected threshold [t] achieved [metrics]" in conclusion


def test_secondary_product_is_named_for_what_it_computes() -> None:
    # FLD-M2 = B (wording/disclosure test): class 0 treated as not water, product renamed; no "known class" wording.
    markdown = "\n".join(_markdown(_notebook()))
    assert "**Predicted water outside JRC 2018 permanent water:**" in markdown
    assert "Class 0 is therefore treated as not water, and the area is computed on every clear valid pixel" in markdown
    assert "Prithvi predicts *x* km² of water outside JRC 2018 permanent water on the clear pixels of this image" in markdown
    for stale in ("known class", "Candidate inundation", "candidate-inundation map", "meaning is unverified"):
        assert stale not in markdown


def test_jrc_class0_is_treated_as_not_water(runner, monkeypatch, tmp_path: Path) -> None:
    # FLD-M2 = B: only JRC class 3 is removed; class 0 counts as not water; only invalid predictions are excluded.
    monkeypatch.setattr(runner, "candidate_figure", lambda root, record, *args: tmp_path / "figure.png")
    record = {"id": "SYN_DEV", "role": "development", "assets": {"permanent": {"path": "SYN/permanent.tif"}}}
    _write_raster(tmp_path / "cache" / "SYN" / "permanent.tif", np.array([[0, 0, 1, 2, 3, 0]]), "uint8")
    _write_raster(runner.scene_output(tmp_path, record, "prithvi"), np.array([[1, 0, 1, 1, 1, 255]]), "uint8", 255)
    products = runner.candidate_inundation(tmp_path, {"permanent_water": {"verified": True}}, record)
    import rasterio

    with rasterio.open(products[0]) as source:
        assert source.read(1).tolist() == [[1, 0, 1, 1, 0, 255]]
    sidecar = runner.read_json(products[1])
    assert sidecar["candidate_inundation_km2"] == pytest.approx(3 * 100 / 1e6)
    assert sidecar["predicted_water_outside_jrc_permanent_km2"] == sidecar["candidate_inundation_km2"]
    assert (sidecar["clear_valid_pixels"], sidecar["jrc_class0_pixels"]) == (5, 2)
    assert sidecar["jrc_class0_fraction"] == pytest.approx(0.4)
    assert sidecar["product"] == "Predicted water outside JRC 2018 permanent water"
    assert "excluded_jrc_class0_fraction" not in sidecar and "treated as not water" in sidecar["jrc_class0_rule"]


def test_candidate_table_shows_the_class0_share(monkeypatch, tmp_path: Path) -> None:
    namespace, shown = _helpers(monkeypatch, tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "EMSR312_08CANDON_X_candidate_inundation.json").write_text(json.dumps({
        "scene": "EMSR312_08CANDON_X", "role": "development", "meaning": "m", "candidate_inundation_km2": 1.69,
        "confirmed_flood": False, "limitations": "l", "jrc_class0_fraction": 0.9}))
    namespace["show_candidates"]()
    assert "| 08CANDON | development | 1.6900 | no | 90.0000 |" in shown[0][1]
    assert "Predicted water outside JRC 2018 permanent water (km²)" in shown[0][1]
    assert "Clear valid pixels in JRC class 0, treated as never water (%)" in shown[0][1]


def test_run_stage_prints_the_log_tail_only_when_a_stage_fails() -> None:
    # FLD-m1: the live stream already shows each line; the tail is repeated only for diagnostics on failure.
    install = _install_cell(_notebook())
    function = next(n for n in ast.walk(ast.parse(install)) if isinstance(n, ast.FunctionDef) and n.name == "run_stage")
    tail_prints = [n for n in ast.walk(function) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
                   and "read_text" in ast.unparse(n)]
    assert len(tail_prints) == 1
    failure = next(s for s in function.body if isinstance(s, ast.If) and "returncode" in ast.unparse(s.test))
    assert tail_prints[0] in list(ast.walk(failure))


def test_only_the_numpy_shape_deprecation_warning_is_filtered() -> None:
    capstone = _carried(_notebook())[0]["capstone.py"]
    filters = [line for line in capstone.splitlines() if "filterwarnings" in line or "simplefilter" in line]
    assert filters == ["warnings.filterwarnings("]
    assert 'message="Setting the shape on a NumPy array has been deprecated", category=DeprecationWarning' in capstone


def test_label_lineage_kind_notice_and_runtime_are_stated() -> None:
    # FLD-m2 and FLD-m4 = B (wording/structure test): Notebook Spec 2.5 write-up layer and label lineage.
    notebook = _notebook()
    meta = notebook["metadata"]["dimer"]
    assert meta["notebook_kind"] == "CAPSTONE" and meta["notebook_spec"] == "2.5"
    opening = _source(notebook["cells"][0])
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    assert first_code > 0 and "> **⚠️ Notice.**" in opening and "disaster-response" in opening
    assert opening.index("> **⚠️ Notice.**") < opening.index("### Problem statement")
    assert "**Expected runtime:** about 5 minutes" in opening and "Tesla T4 on 2026-10-03" in opening
    markdown = "\n".join(_markdown(notebook))
    assert "curated the maps manually" in markdown and "create_gt.py" in markdown
    assert "**Where the reference labels come from (label lineage).**" in markdown and "RADARSAT-2" in markdown
    for heading in ("### Problem statement", "### Data context", "## What did not work",
                    "**Answer to the problem statement.**"):
        assert heading in markdown
    assert "DIMER Notebook Specification `2.5`" in opening


# --- 2026-10-12 peer-review follow-up (CX1, CX4-CX8): wording/disclosure tests, not validation of semantics ----------


def test_reference_timing_is_stated_as_a_timestamp_difference_not_a_fact_about_the_mask() -> None:
    notebook = _notebook()
    markdown = "\n".join(_markdown(notebook))
    capstone = _carried(notebook)[0]["capstone.py"]
    for stale in ("16 hours before the image", "made 16 hours", "made about 16 hours", "reference precedes image"):
        assert stale not in markdown and stale not in capstone
    assert markdown.count("temporal support is not fully established") >= 4
    assert "reference temporal support not fully established" in capstone


def test_iou_f1_optimum_and_threshold_change_are_described_correctly() -> None:
    markdown = "\n".join(_markdown(_notebook()))
    assert "error balance" not in markdown
    assert "reach their maximum at the same tested threshold on this fixed mask, because F1 = 2·IoU / (1 + IoU)" in markdown
    assert "The notebook does exactly that as a declared secondary row" in markdown


def test_precision_is_not_presented_as_sufficient_under_imbalance() -> None:
    markdown = "\n".join(_markdown(_notebook()))
    assert "precision-based measures stay informative" not in markdown
    assert "Read precision and recall together, alongside water IoU, class support and coverage." in markdown


def test_figure_captions_name_the_score_margin_and_the_candidate_population() -> None:
    notebook = _notebook()
    capstone = _carried(notebook)[0]["capstone.py"]
    assert "small = ambiguous" not in capstone and "Water probability >" not in capstone
    assert "Distance from decision threshold 0.5" in capstone and 'f"Softmax water score > {threshold}"' in capstone
    figure = capstone[capstone.index("def candidate_figure"):capstone.index("def report")]
    assert "the reference-conditioned scoring population" in figure and "is treated as not water" in figure
    assert "class0_percent" in figure and "Water outside JRC 2018 permanent water" in figure
    assert "score-margin diagnostic, not calibrated uncertainty" in "\n".join(_markdown(notebook))


# --- 2026-10-13 owner decisions: FLD-M1 = C (declared secondary Prithvi row) -------------------------------------


def test_prithvi_secondary_threshold_uses_the_mndwi_rule_on_candon(runner, tmp_path: Path) -> None:
    # Ties pick the threshold closest to zero, then the smaller one: on a 0..1 score that is the smallest tied value.
    record = {"id": "SYN_DEV", "role": "development", "assets": {}}
    _write_raster(runner.scene_output(tmp_path, record, "reference"), np.array([[1, 1, 0, 0, 255]]), "uint8", 255)
    _write_raster(runner.scene_output(tmp_path, record, "probability"),
                  np.array([[0.72, 0.9, 0.31, 0.05, -9999]]), "float32", -9999)
    path = runner.select_prithvi_threshold(tmp_path, record)
    selection = runner.read_json(path)
    assert selection["grid"] == [step / 20 for step in range(1, 20)] == runner.PRITHVI_GRID
    best = max(r["water_iou"] for r in selection["sweep"])
    tied = [r["threshold"] for r in selection["sweep"] if r["water_iou"] == best]
    assert tied == [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] and selection["selected_threshold"] == 0.35
    assert selection["selected_threshold"] == runner.select_threshold(selection["sweep"])
    assert selection["primary_threshold"] == 0.5 and runner.secondary_threshold(tmp_path) == 0.35


def test_secondary_row_is_scored_beside_the_primary_row(runner, tmp_path: Path) -> None:
    record = {"id": "SYN_HELD", "role": "heldout", "assets": {}}
    reference = np.array([[1, 1, 0, 0, 255]] * 4)
    probability = np.array([[0.4, 0.8, 0.45, 0.1, -9999]] * 4)
    _write_raster(runner.scene_output(tmp_path, record, "reference"), reference, "uint8", 255)
    _write_raster(runner.scene_output(tmp_path, record, "probability"), probability, "float32", -9999)
    _write_raster(runner.scene_output(tmp_path, record, "prithvi"), np.where(reference == 255, 255, probability > 0.5),
                  "uint8", 255)
    _write_raster(runner.scene_output(tmp_path, record, "mndwi"), np.array([[0.3, 0.3, -0.2, -0.3, -9999]] * 4),
                  "float32", -9999)
    runner.write_json(tmp_path / "outputs" / "prithvi_selection.json", {"selected_threshold": 0.35})
    rows, _, products = runner.evaluate_exports(tmp_path, record, 0.15)
    by_method = {r["method"]: r for r in rows}
    assert list(by_method) == ["no_water", "mndwi_prediction", "prithvi", "prithvi_candon_selected"]
    assert (by_method["prithvi"]["tp"], by_method["prithvi"]["fn"]) == (4, 4)
    assert (by_method["prithvi_candon_selected"]["tp"], by_method["prithvi_candon_selected"]["fp"]) == (8, 4)
    assert runner.scene_output(tmp_path, record, "prithvi_candon_selected") in products


def test_secondary_row_is_declared_before_any_prithvi_score_and_required_before_vigan() -> None:
    capstone = _carried(_notebook())[0]["capstone.py"]
    baseline = capstone[capstone.index("def baseline("):capstone.index("def load_model(")]
    assert '"prithvi_secondary_declaration"' in baseline and "PRITHVI_GRID" in baseline
    infer = capstone[capstone.index("def infer("):capstone.index("def write_csv(")]
    assert 'if role == "development":' in infer and "select_prithvi_threshold(root, record)" in infer
    assert "must be selected on Candon before Vigan" in infer
    assert "show_record('prithvi_selection.json')" in "\n".join(_source(c) for c in _notebook()["cells"])
