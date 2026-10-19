"""Row-70 tutorial review (2026-10-19): deterministic op replacements (R70-DET1 option A), the pre-registered grid,
reading rule and fail-closed score-once wiring (R70-M3, SCORE_ONCE_RULE 1-8), the fresh-test pins (R70-D1), the BYOD
contract (R70-M1) and the disclosures (R70-M2, R70-M4, R70-m1..m8)."""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

with contextlib.suppress(ImportError):  # import torch before any NumPy linear algebra (Windows DLL order)
    import torch  # noqa: F401

from prithvi_flood_segmentation_pipeline import SAMPLE_RECORDS
from prithvi_flood_segmentation_pipeline import pipeline as P
from prithvi_flood_segmentation_pipeline import samples as S
from prithvi_flood_segmentation_pipeline import scoring as SC

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_flood_segmentation_colab.ipynb"
STEM = "prithvi_flood_segmentation"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


# --- R70-DET1: each replacement equals the original op (forward and gradient) at the real shapes --------------------

# UperNet on a 512 x 512 chip: neck levels 128/64/32/16; PPM pools 16 -> 1/2/3/6 and upsamples back; head 128 -> 512.
RESIZES = [(16, 32, True), (32, 64, True), (64, 128, True), (16, 128, True), (32, 128, True), (1, 16, True), (2, 16, True),
           (3, 16, True), (6, 16, True), (128, 512, False)]
TOL = 1e-5  # float32, stated in docs/prithvi-flood-preregistration.md §4


@pytest.mark.parametrize(("n_in", "n_out", "corners"), RESIZES)
def test_det_resize_matches_bilinear_interpolate(n_in, n_out, corners):
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(n_in * 1000 + n_out)
    x = torch.randn(2, 8, n_in, n_in, generator=gen, requires_grad=True)
    ref = torch.nn.functional.interpolate(x, size=(n_out, n_out), mode="bilinear", align_corners=corners)
    grad = torch.randn(ref.shape, generator=gen)
    (g_ref,) = torch.autograd.grad(ref, x, grad)
    out = P.deterministic_resize(x, (n_out, n_out), align_corners=corners)
    (g_out,) = torch.autograd.grad(out, x, grad)
    assert torch.allclose(out, ref, atol=TOL) and torch.allclose(g_out, g_ref, atol=TOL * 10)


@pytest.mark.parametrize("scale", [1, 2, 3, 6])
def test_det_adaptive_pool_matches_adaptive_avg_pool2d(scale):
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(scale)
    x = torch.randn(2, 16, 16, 16, generator=gen, requires_grad=True)
    ref = torch.nn.AdaptiveAvgPool2d(scale)(x)
    grad = torch.randn(ref.shape, generator=gen)
    (g_ref,) = torch.autograd.grad(ref, x, grad)
    out = P._deterministic_pool_class()(scale)(x)
    (g_out,) = torch.autograd.grad(out, x, grad)
    assert torch.allclose(out, ref, atol=TOL) and torch.allclose(g_out, g_ref, atol=TOL)


def test_det_cross_entropy_equals_f_cross_entropy_with_ignore_index():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 2, 512, 512, generator=gen, requires_grad=True)
    target = torch.randint(-1, 2, (2, 512, 512), generator=gen)
    ref = torch.nn.functional.cross_entropy(logits, target, ignore_index=-1)
    (g_ref,) = torch.autograd.grad(ref, logits)
    out = P.deterministic_cross_entropy(logits, target, ignore_index=-1)
    (g_out,) = torch.autograd.grad(out, logits)
    assert abs(float(out) - float(ref)) < 1e-6 and torch.allclose(g_out, g_ref, atol=1e-9)


def test_det_upernet_patch_matches_upstream_forward_and_gradients():
    torch = pytest.importorskip("torch")
    upernet = pytest.importorskip("terratorch.models.decoders.upernet_decoder")
    import copy

    torch.manual_seed(0)
    # float64: the patched decoder must equal upstream exactly (up to float64 rounding); float32 parameter gradients
    # summed over 128 x 128 pixels differ only by float32 accumulation order.
    reference = upernet.UperNetDecoder(embed_dim=[64, 64, 128, 128], channels=32).double().eval()
    patched = copy.deepcopy(reference)
    sites = P.make_deterministic(patched)
    assert sum("AdaptiveAvgPool2d" in s for s in sites) == 4 and any("UperNetDecoder" in s for s in sites) and any("PPM" in s for s in sites)
    assert list(reference.state_dict()) == list(patched.state_dict())  # no parameter or buffer added or renamed
    inputs = [torch.randn(1, c, n, n, dtype=torch.float64, requires_grad=True) for c, n in ((64, 128), (64, 64), (128, 32), (128, 16))]
    ref = reference(inputs)
    out = patched(inputs)
    assert torch.allclose(out, ref, atol=1e-10)
    grad = torch.randn(ref.shape, dtype=torch.float64)
    g_ref = torch.autograd.grad(ref, [*inputs, *reference.parameters()], grad)
    g_out = torch.autograd.grad(out, [*inputs, *patched.parameters()], grad)
    assert all(torch.allclose(a, b, atol=1e-9, rtol=1e-9) for a, b in zip(g_ref, g_out, strict=True))


def test_det_patched_model_builds_and_loads_the_checkpoint_strictly():
    pytest.importorskip("terratorch")
    model = P.build_model()
    patches = model.deterministic_patches
    assert sum("AdaptiveAvgPool2d" in s for s in patches) == 4 and "PixelWiseModel.forward output rescale -> deterministic_resize" in patches
    assert sum(p.numel() for p in model.parameters()) == P.PARAMETER_COUNT
    weights = os.environ.get("DIMER_PRITHVI_WEIGHTS")
    if not weights:
        pytest.skip("set DIMER_PRITHVI_WEIGHTS to the converted-weights folder to run the strict load")
    from safetensors.torch import load_file

    P.verify_converted(weights)
    model.load_state_dict(load_file(str(Path(weights) / P.CONVERTED_WEIGHTS_NAME)), strict=True)


def test_det_set_determinism_sets_every_flag_without_warn_only():
    torch = pytest.importorskip("torch")
    before = torch.are_deterministic_algorithms_enabled()
    try:
        state = P.set_determinism(0)
        assert state == P.DETERMINISM_FLAGS
    finally:
        torch.use_deterministic_algorithms(before)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"  # set on import, before any CUDA work


def test_det_adapt_uses_the_deterministic_loss_only():
    source = (ROOT / "src" / "prithvi_flood_segmentation_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    adapt = source[source.index("    def adapt(") : source.index("    def save_artifact(")]
    assert "cross_entropy(" in adapt and "functional.cross_entropy" not in adapt and "interpolate(" not in source.split("def _logits", 1)[1].split("def predict", 1)[0]


# --- R70-D2: the grid, its selection rule and hash ------------------------------------------------------------------


def _rows(c0, c1, c2):
    return [{"name": n, "kept_val_loss": v} for n, v in (("C0", c0), ("C1", c1), ("C2", c2))]


@pytest.mark.parametrize(
    ("losses", "selected"),
    [((0.118, 0.115, 0.116), "C0"), ((0.118, 0.113, 0.120), "C1"), ((0.118, 0.120, 0.110), "C2"), ((0.118, 0.110, 0.110), "C1"), ((0.118, 0.105, 0.110), "C1")],
)
def test_grid_selection_keeps_c0_unless_beaten_by_the_margin(losses, selected):
    assert SC.select_config(_rows(*losses))["selected"] == selected


def test_grid_partial_grid_is_refused_and_hash_is_stable():
    with pytest.raises(ValueError, match="incomplete"):
        SC.select_config(_rows(0.1, 0.1, 0.1)[:2])
    rows = [{"name": "C0", "val_loss": [0.1306, 0.1178], "best_epoch": 1, "kept_val_loss": 0.1178}]
    assert SC.grid_hash(rows, "C0") == SC.grid_hash(json.loads(json.dumps(rows)), "C0")
    assert SC.grid_hash(rows, "C0") != SC.grid_hash([{**rows[0], "kept_val_loss": 0.117801}], "C0")
    assert [c["name"] for c in SC.GRID] == ["C0", "C1", "C2"] and SC.SELECTION_MARGIN == 0.005 and SC.RECORDED_SELECTION == "C0"


# --- R70-D3: reading rule -------------------------------------------------------------------------------------------


def test_reading_rule_needs_an_interval_excluding_zero_and_the_tolerance():
    assert SC.reading_rule(0.03, [0.01, 0.05]) == "improved"
    assert SC.reading_rule(0.015, [0.005, 0.03]) == "no detectable change"  # excludes 0 but below 0.02
    assert SC.reading_rule(0.05, [-0.01, 0.1]) == "no detectable change"
    assert SC.reading_rule(-0.03, [-0.05, -0.01]) == "worse"


def test_paired_bootstrap_uses_unrounded_counts_and_a_fixed_seed():
    frozen = [[[90, 10], [20, 80]]] * 5 + [[[50, 0], [30, 20]]] * 5
    selected = [[[88, 12], [10, 90]]] * 5 + [[[50, 0], [25, 25]]] * 5
    a = SC.paired_bootstrap(frozen, selected, n_boot=500)
    b = SC.paired_bootstrap(frozen, selected, n_boot=500)
    assert a == b and a["n_boot"] == 500 and a["seed"] == 0
    assert a["delta"] == pytest.approx(SC.pooled_water_iou(selected) - SC.pooled_water_iou(frozen))
    assert a["ci95"][0] <= a["delta"] <= a["ci95"][1]


# --- R70-M3: fail-closed score-once wiring --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env_unmasked(monkeypatch, request):
    """The score-once tests score stand-ins only; score_fresh_test now re-reads DIMER_MASK_FRESH_TEST itself (F1,
    round 2). The one test that can touch the real fresh objects keeps the caller's mask."""
    if "fresh_pin_metadata" not in request.node.name:
        monkeypatch.delenv(S.MASK_ENV, raising=False)
    # These tests exercise the score-once rules as they stand before a record exists (first look = evaluation);
    # the recorded constants and the reproduction label with the record set are in test_r70_record.py.
    monkeypatch.setattr(SC, "RECORDED_EVALUATION", None)


def _settings(**changes):
    return {**SC.PREREGISTERED, "device": "Tesla T4", **changes}


CHECKS = {"grid_complete_and_matching": True, "model_is_selected_system": True, "fresh_pool_digest": True}


def _score(calls):
    def score():
        calls.append(1)
        return {"chips": ["a", "b"], "frozen": [[[9, 1], [2, 8]], [[5, 0], [3, 2]]], "selected": [[[9, 1], [1, 9]], [[5, 0], [2, 3]]]}

    return score


def _run(tmp_path, calls, settings=None, checks=None, score=None):
    return SC.score_fresh_test(settings=settings or _settings(), checks=checks or CHECKS, score=score or _score(calls),
                               score_dir=tmp_path, stem=STEM, exploratory_payload={"validation": 1})


def test_score_once_first_run_is_the_evaluation_and_a_rerun_is_a_reproduction(tmp_path):
    calls: list = []
    first = _run(tmp_path, calls)
    assert first["test_look"] == "pre-registered evaluation (this runtime)" and first["settings_differing"] == []
    second = _run(tmp_path, calls)
    assert second["test_look"] == "reproduction" and second["file"].endswith("_fresh_test_reproduction_1.json")
    ledger = json.loads((tmp_path / f"{STEM}_fresh_test_ledger.json").read_text(encoding="utf-8"))
    assert [e["test_look"] for e in ledger] == ["pre-registered evaluation (this runtime)", "reproduction"] and len(calls) == 2


@pytest.mark.parametrize(
    "change",
    [{"epochs": 8}, {"batch_size": 1}, {"seed": 1}, {"grid": [{"name": "off-grid", "lr": 1e-4, "trainable": "decoder"}]},
     {"data_source": "BYOD (mine.zip)"}, {"device": "cpu"}, {"device": "Tesla V100"}, {"masked": True},
     {"determinism": {**P.DETERMINISM_FLAGS, "warn_only": True}}, {"fresh_pool_digest": "0" * 64}],
)
def test_score_once_any_changed_setting_is_exploratory_and_never_scores(tmp_path, change):
    calls: list = []
    result = _run(tmp_path, calls, settings=_settings(**change))
    assert result["test_look"] == "not scored (exploratory)" and calls == [] and result["settings_differing"]
    assert Path(result["file"]).name == f"{STEM}_fresh_test_exploratory_1.json"
    assert not (tmp_path / f"{STEM}_fresh_test_ledger.json").exists() and not (tmp_path / f"{STEM}_fresh_test_report.json").exists()
    assert _run(tmp_path, calls)["test_look"] == "pre-registered evaluation (this runtime)"  # an exploratory run is not a look


def test_score_once_partial_grid_or_failed_check_is_not_a_look(tmp_path):
    calls: list = []
    with pytest.raises(RuntimeError, match="not a look"):
        _run(tmp_path, calls, checks={**CHECKS, "grid_complete_and_matching": False})
    assert calls == [] and not list(tmp_path.glob("*.json"))
    assert _run(tmp_path, calls)["test_look"] == "pre-registered evaluation (this runtime)"


def test_score_once_failed_fetch_or_decode_is_not_a_look(tmp_path):
    def failing():
        raise ValueError("fetched 0 bytes")

    with pytest.raises(ValueError):
        _run(tmp_path, [], score=failing)
    assert not list(tmp_path.glob("*.json"))
    assert _run(tmp_path, [])["test_look"] == "pre-registered evaluation (this runtime)"


def test_score_once_interrupted_report_write_is_not_a_look(tmp_path, monkeypatch):
    real = os.replace

    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, [])
    monkeypatch.setattr(os, "replace", real)
    assert not list(tmp_path.glob("*.json")) and not list(tmp_path.glob(".*.tmp"))
    assert _run(tmp_path, [])["test_look"] == "pre-registered evaluation (this runtime)"


def test_score_once_interrupt_between_report_and_ledger_fails_closed(tmp_path, monkeypatch):
    real = os.replace
    seen: list = []

    def second_fails(src, dst):
        seen.append(dst)
        if len(seen) == 2:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(os, "replace", second_fails)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, [])
    monkeypatch.setattr(os, "replace", real)
    assert (tmp_path / f"{STEM}_fresh_test_report.json").exists()
    assert _run(tmp_path, [])["test_look"] == "reproduction"  # the report exists, so the evaluation already happened


@pytest.mark.parametrize("content", ["{not json", "", '{"a": 1}'])
def test_score_once_corrupt_ledger_never_crashes_or_promotes(tmp_path, content):
    (tmp_path / f"{STEM}_fresh_test_ledger.json").write_text(content, encoding="utf-8")
    result = _run(tmp_path, [])
    assert result["test_look"] == "reproduction"
    assert (tmp_path / f"{STEM}_fresh_test_ledger_unreadable_1.json").read_text(encoding="utf-8") == content
    assert len(json.loads((tmp_path / f"{STEM}_fresh_test_ledger.json").read_text(encoding="utf-8"))) == 1


def test_score_once_masked_run_is_exploratory(tmp_path):
    calls: list = []
    assert _run(tmp_path, calls, settings=_settings(masked=True))["settings_differing"] == ["masked"] and calls == []


def test_score_once_recorded_evaluation_makes_a_new_runtime_a_reproduction(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "RECORDED_EVALUATION", {"run": "hosted T4", "revision": "x"})
    assert _run(tmp_path, [])["test_look"] == "reproduction"
    assert SC.RECORDED_EVALUATION is not None


def test_score_once_no_permanent_scored_once_wording():
    text = (ROOT / "src" / "prithvi_flood_segmentation_pipeline" / "scoring.py").read_text(encoding="utf-8")
    assert "scored once" not in text.lower().replace("score-once", "") and "SCORED_ONCE" not in text


# --- R70-D1: the fresh-test pins ----------------------------------------------------------------------------------


def test_fresh_pins_match_the_preregistration_and_are_disjoint():
    names = [r[0] for r in S.FRESH_TEST_RECORDS]
    assert len(names) == 23 == len(set(names)) and S.fresh_pool_digest() == S.FRESH_POOL_DIGEST
    assert sum(r[1] + r[3] for r in S.FRESH_TEST_RECORDS) == S.FRESH_TEST_BYTES
    assert not set(names) & {r[0] for r in SAMPLE_RECORDS} and not set(names) & set(S.FRESH_TEST_EXCLUDED)
    assert {n.split("_")[0] for n in names} == set(S.FRESH_TEST_EVENTS) and not any(n.startswith("Bolivia") for n in names)
    prereg = (ROOT / "docs" / "prithvi-flood-preregistration.md").read_text(encoding="utf-8")
    assert S.FRESH_POOL_DIGEST in prereg and all(f"| {n} |" in prereg for n in names)
    assert all(len(r[2]) == 64 and len(r[4]) == 64 for r in S.FRESH_TEST_RECORDS)


def test_fresh_pin_metadata_resolves_and_parses_without_scoring():
    """Fetch (size + SHA-256 verified) and parse the TIFF headers of every pinned fresh chip; nothing is decoded or
    scored. Runs with DIMER_FRESH_CACHE (a folder holding the objects) or DIMER_NETWORK_TESTS=1 (the public bucket)."""
    cache = os.environ.get("DIMER_FRESH_CACHE")
    if not cache and os.environ.get("DIMER_NETWORK_TESTS") != "1":
        pytest.skip("set DIMER_FRESH_CACHE or DIMER_NETWORK_TESTS=1")
    meta = S.fresh_test_metadata(cache_dir=cache)
    assert len(meta) == 23
    assert all(m["image"]["shape"] == [13, 512, 512] and m["label"]["shape"] == [512, 512] for m in meta)


# --- notebook wiring -------------------------------------------------------------------------------------------------


def test_fresh_test_is_fetched_and_scored_in_exactly_one_cell(notebook):
    cells = _code_cells(notebook)
    learner = [c for c in cells if "FRESH_TEST_RECORDS: tuple" not in c and "def fetch_fresh_test(" not in c]
    fresh = [c for c in learner if "fetch_fresh_test(" in c]
    assert len(fresh) == 1 and "score_fresh_test(" in fresh[0] and "checks = {" in fresh[0]
    assert "restore_pinned_base(pipe)" in fresh[0] and "SELECTED_FINGERPRINT" in fresh[0] and "device_name" in fresh[0]
    for cell in learner:
        if cell is not fresh[0]:
            assert "fresh_records" not in cell
    assert "set_determinism(0)" in next(c for c in learner if "grid_on = RUN_GRID" in c)


def test_r70_prose_contract_disclosures(notebook):
    md = _markdown(notebook)
    assert "never used for training or epoch selection" not in md  # R70-M2
    assert "2026-09-19 build sweep" in md and "reflectance-scaling bug fixed on 2026-09-25" in md
    assert "**L1C top-of-atmosphere**" in md and "+1000 DN offset" in md and "HLS" in md  # R70-M1, R70-m2
    assert "an uncorrected product is segmented" not in md
    assert "252-chip official training split" in md and "89-chip validation split used for early stopping" in md  # R70-M4, m1
    assert "trained on the 446" not in md and "exposed development test" in md and "within-event" in md
    assert "Kaggle Tesla T4 run of 25 September 2026" not in md and "2026-10-08" in md  # R70-m4
    assert "## 9. Upstream example chips, artifact export and fresh reload" in md and "New chips" not in md  # R70-m7
    assert "reloads with identical outputs" not in md  # R70-S1
    schema = P.INPUT_SCHEMA["validation"] + P.INPUT_SCHEMA["value_units"]
    assert "L1C top-of-atmosphere" in schema and "atmospherically corrected," not in schema


def test_r70_m6_sample_writers_are_atomic(tmp_path, monkeypatch):
    records = [{"id": "a", "source_id": "X_1", "region": "X", "source": "s"}]

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        S.write_dataset_csv(records, tmp_path / "pairs.csv")
    assert not (tmp_path / "pairs.csv").exists() and not list(tmp_path.glob(".*.tmp"))


def test_r70_model_card_and_release_docs():
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "surface-reflectance" not in card and "L1C top-of-atmosphere" in card
    assert "252-chip" in card and "fresh test" in card
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "evaluation of record" in record.lower() and "prithvi-flood-preregistration.md" in record


def test_r70_m5_no_local_home_paths_in_committed_files():
    import re
    import subprocess

    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    pattern = re.compile(rb"/home/[a-z][a-z0-9_-]*/|Users[\\/]Kurt|KURTVA~[0-9]")
    this = Path(__file__).resolve().relative_to(ROOT).as_posix()  # holds the pattern itself
    hits = [p for p in tracked if p != this and (ROOT / p).is_file() and pattern.search((ROOT / p).read_bytes())]
    assert hits == []


def test_bootstrap_inputs_are_numpy_compatible():
    assert SC.pooled_water_iou(np.zeros((1, 2, 2))) == 0.0
