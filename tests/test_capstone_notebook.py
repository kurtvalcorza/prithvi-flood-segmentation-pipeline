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
            cell["source"] = _source(cell).replace("(Ploton et al., 2020)", "(the spatial-validation literature)")

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match=r"reference Ploton \(2020\) is never cited"):
        module.validate_workshop_notebooks()


def test_control_citation_without_reference_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        cell = next(c for c in notebook["cells"] if "## 4. Run frozen Prithvi" in _source(c))
        cell["source"] = _source(cell) + "\n\nSegmentation transformers are widely used (Nobody et al., 2024).\n"

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match=r"citation \(Nobody, 2024\) has no reference entry"):
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
