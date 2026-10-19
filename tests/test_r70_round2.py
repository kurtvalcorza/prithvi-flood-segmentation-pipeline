"""Row-70 round 2 (peer review F1, F3-F7): the fresh-test mask, the look marker, the dataset digest, the use notice
and the float16 closeness of the deterministic replacements.

Every scoring here uses stand-ins; nothing fetches or scores the real fresh test.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from prithvi_flood_segmentation_pipeline import pipeline as P
from prithvi_flood_segmentation_pipeline import samples as S
from prithvi_flood_segmentation_pipeline import scoring as SC

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_flood_segmentation_colab.ipynb"
STEM = "prithvi_flood_segmentation"
CHECKS = {"grid_complete_and_matching": True, "model_is_selected_system": True, "fresh_pool_digest": True}


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def _settings(**changes):
    return {**SC.PREREGISTERED, "device": "Tesla T4", **changes}


def _stub_score(calls):
    def score():
        calls.append(1)
        frozen, selected = [[[9, 1], [2, 8]], [[5, 0], [3, 2]]], [[[9, 1], [1, 9]], [[5, 0], [2, 3]]]
        return {"chips": ["a", "b"], "frozen": frozen, "selected": selected}

    return score


def _run(score_dir, calls, marker=None, settings=None, score=None):
    return SC.score_fresh_test(settings=settings or _settings(), checks=CHECKS, score=score or _stub_score(calls),
                               score_dir=score_dir, stem=STEM, exploratory_payload={"validation": 1}, marker=marker)


def _never_fetch(url):
    raise AssertionError(f"masked, yet a fresh-test object was fetched: {url}")


# --- F1: the mask is re-read from the environment at every layer ----------------------------------------------------


def test_f1_fetch_fresh_test_refuses_when_the_env_mask_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv(S.MASK_ENV, "1")
    with pytest.raises(RuntimeError, match="masked"):
        S.fetch_fresh_test(cache_dir=tmp_path, fetcher=_never_fetch)
    with pytest.raises(RuntimeError, match="masked"):
        S.fresh_test_metadata(cache_dir=tmp_path, fetcher=_never_fetch)
    assert not list(tmp_path.iterdir())


def test_f1_score_fresh_test_masks_on_the_env_even_when_settings_say_unmasked(tmp_path, monkeypatch):
    monkeypatch.setenv(S.MASK_ENV, "1")
    calls: list = []
    result = _run(tmp_path / "score_once", calls, marker=tmp_path / "marker.json", settings=_settings(masked=False))
    assert result["test_look"] == SC.LABEL_EXPLORATORY and result["settings_differing"] == ["masked"] and calls == []
    assert not (tmp_path / "marker.json").exists()
    monkeypatch.delenv(S.MASK_ENV)
    assert _run(tmp_path / "score_once", calls)["test_look"] == SC.LABEL_EVALUATION and calls == [1]  # control


def _section_1(nb: dict) -> str:
    found = [c for c in _code(nb) if "# dimer: kernel cell" in c and "IsolatedRuntime" in c]
    assert len(found) == 1
    return found[0]


@pytest.fixture
def live_worker(notebook, tmp_path, monkeypatch):
    """The real Section 1 cell with a stand-in interpreter, started with NO mask (the risky state)."""
    if sys.platform == "win32":
        pytest.skip("the isolated worker uses pass_fds and a symlinked interpreter (POSIX only)")
    source = _section_1(notebook)
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    monkeypatch.delenv(S.MASK_ENV, raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        yield source, namespace, runtime
    finally:
        runtime.close()


PROBE = "import os; print('mask=' + str(os.environ.get('DIMER_MASK_FRESH_TEST')))\n"


def test_f1_worker_sees_a_mask_set_in_the_kernel_after_it_started(live_worker, capsys, monkeypatch):
    _, _, runtime = live_worker
    runtime.run(PROBE)
    assert "mask=None" in capsys.readouterr().out  # control: the worker starts unmasked
    monkeypatch.setenv(S.MASK_ENV, "1")  # e.g. %env in the kernel after Section 1
    runtime.run(PROBE)
    assert "mask=1" in capsys.readouterr().out


def test_f1_marked_kernel_cell_after_section_1_masks_the_worker(live_worker, capsys):
    _, namespace, runtime = live_worker
    kernel_cell = "# dimer: kernel cell\nimport os\nos.environ['DIMER_MASK_FRESH_TEST'] = '1'\n"
    route = namespace["_route_to_isolated_runtime"]
    assert route([kernel_cell]) == [kernel_cell]  # not routed: it runs in the kernel
    exec(compile(kernel_cell, "<kernel cell>", "exec"), {})  # the fixture's monkeypatch restores os.environ afterwards
    exec(compile("".join(route([PROBE])), "<routed>", "exec"), namespace)
    assert "mask=1" in capsys.readouterr().out


def test_f1_rerun_of_section_1_with_a_live_worker_keeps_the_mask(live_worker, capsys, monkeypatch):
    source, namespace, runtime = live_worker
    monkeypatch.setenv(S.MASK_ENV, "1")
    exec(compile(source, "<section 1 again>", "exec"), namespace)
    assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()  # reused, not restarted
    capsys.readouterr()
    runtime.run(PROBE)
    assert "mask=1" in capsys.readouterr().out


def test_f1_the_kernel_can_switch_the_mask_on_but_never_off(live_worker, capsys, monkeypatch):
    _, _, runtime = live_worker
    runtime.run("import os; os.environ['DIMER_MASK_FRESH_TEST'] = '1'\n")  # a routed cell masks inside the worker
    monkeypatch.delenv(S.MASK_ENV, raising=False)
    runtime.run(PROBE)
    assert "mask=1" in capsys.readouterr().out


def test_f1_worker_source_forwards_the_kernel_mask_with_every_cell(notebook):
    source = _section_1(notebook)
    assert 'self._send.send(("run", source, os.environ.get("DIMER_MASK_FRESH_TEST")))' in source
    assert 'if len(message) > 2 and message[2] == "1":' in source


def test_f1_section_8_prints_the_mask_state_before_any_fetch(notebook):
    cell = next(c for c in _code(notebook) if "fresh_result = score_fresh_test(" in c)
    printed = cell.index("'will_fetch_fresh_test'")
    assert cell.index("'masked': MASKED") < printed < cell.index("fresh_result = score_fresh_test(")
    assert "marker=LOOK_MARKER" in cell
    assert "'not scored (failed before scoring completed; not a look)'" in cell


# --- F4: a look survives a deleted outputs/ and a failed write ------------------------------------------------------


def test_f4_deleting_outputs_does_not_reset_the_look(tmp_path, monkeypatch):
    monkeypatch.delenv(S.MASK_ENV, raising=False)
    outputs, marker, calls = tmp_path / "outputs" / "score_once", tmp_path / "weights" / "fresh" / "look.json", []
    assert _run(outputs, calls, marker=marker)["test_look"] == SC.LABEL_EVALUATION
    shutil.rmtree(tmp_path / "outputs")
    again = _run(outputs, calls, marker=marker)
    assert again["test_look"] == SC.LABEL_REPRODUCTION and "look marker" in again["label_reason"] and len(calls) == 2


@pytest.mark.parametrize("failing", ["ledger", "report"])
def test_f4_a_failed_write_after_scoring_is_still_a_look(tmp_path, monkeypatch, failing):
    monkeypatch.delenv(S.MASK_ENV, raising=False)
    real = SC.atomic_write_json
    outputs, marker, calls = tmp_path / "score_once", tmp_path / "look.json", []

    def disk_full(path, payload):
        if failing in Path(path).name:
            raise OSError(28, "No space left on device")
        return real(path, payload)

    monkeypatch.setattr(SC, "atomic_write_json", disk_full)
    result = _run(outputs, calls, marker=marker)
    assert result["test_look"] == SC.LABEL_EVALUATION and "bootstrap" in result  # returned, not raised
    assert [e.split(":")[0] for e in result["write_errors"]] == [failing] and marker.exists()
    monkeypatch.setattr(SC, "atomic_write_json", real)
    assert _run(outputs, calls, marker=marker)["test_look"] == SC.LABEL_REPRODUCTION


def test_f4_a_failed_fetch_leaves_no_marker_and_is_not_a_look(tmp_path, monkeypatch):
    monkeypatch.delenv(S.MASK_ENV, raising=False)

    def failing():
        raise ValueError("fetched 0 bytes")

    with pytest.raises(ValueError):
        _run(tmp_path / "score_once", [], marker=tmp_path / "look.json", score=failing)
    assert not (tmp_path / "look.json").exists()
    assert _run(tmp_path / "score_once", [], marker=tmp_path / "look.json")["test_look"] == SC.LABEL_EVALUATION


def test_f4_marker_lives_outside_outputs_and_the_prose_says_so(notebook):
    assert SC.LOOK_MARKER.parts[0] == "weights" and "outputs" not in SC.LOOK_MARKER.parts
    md = " ".join(_markdown(notebook).split())
    assert "does **not** reset the look" in md and "weights/sen1floods11_fresh/score_once_look_marker.json" in md


# --- F5: the pinned dataset digest is a pre-registered setting ------------------------------------------------------


def test_f5_same_counts_with_another_dataset_digest_is_exploratory(tmp_path, monkeypatch):
    monkeypatch.delenv(S.MASK_ENV, raising=False)
    assert SC.PREREGISTERED["dataset_digest"] == SC.SAMPLE_DATASET_DIGEST
    assert re.fullmatch(r"[0-9a-f]{64}", SC.SAMPLE_DATASET_DIGEST)
    calls: list = []
    result = _run(tmp_path, calls, settings=_settings(dataset_digest="0" * 64))
    assert result["test_look"] == SC.LABEL_EXPLORATORY and result["settings_differing"] == ["dataset_digest"] and calls == []
    assert SC.preregistration_check(_settings())["preregistered"]


def test_f5_dataset_digest_is_recomputed_from_the_live_records(tmp_path):
    a = [{"id": "train-000", "image": np.full((6, 512, 512), 0.1, "float32"), "label": np.ones((512, 512), "int64")}]
    b = [{**a[0], "label": a[0]["label"].copy()}]
    b[0]["label"][0, 0] = 0  # an in-place label edit, same counts
    assert S.dataset_manifest({"train": a})["digest"] != S.dataset_manifest({"train": b})["digest"]


def test_f5_notebook_compares_the_dataset_digest(notebook):
    cell = next(c for c in _code(notebook) if "run_settings = {" in c)
    live = "dataset_manifest({'train': train_records, 'validation': val_records, 'test': test_records})['digest']"
    assert f"'dataset_digest': {live}" in cell


# --- F3 / F7: prose ---------------------------------------------------------------------------------------------------


def test_f3_cpu_prose_names_the_settings_cpu_differs_in(notebook):
    md = " ".join(_markdown(notebook).split())
    assert "(`grid`, `precision`, `device`; and `masked` when `DIMER_MASK_FRESH_TEST=1` is set)" in md
    assert "(`device`, `grid`)" not in md


def test_f7_use_notice_names_disaster_response_and_no_clinical_clause(notebook):
    title = "".join(notebook["cells"][0]["source"])
    notice = " ".join(line.lstrip("> ") for line in title.splitlines() if line.startswith(">"))
    assert "disaster-response decisions" in notice and "operational flood mapping" in notice
    assert "clinical" not in notice and "diagnostic" not in notice


# --- F6: float16 closeness of the deterministic replacements --------------------------------------------------------

FP16_ATOL = 5e-3  # peer review measured at most 3.9e-3 between the replacement and upstream in float16


@pytest.mark.parametrize(("n_in", "n_out", "corners"), [(64, 128, True), (128, 512, False), (16, 64, True)])
def test_f6_resize_in_float16_matches_upstream_float16(n_in, n_out, corners):
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 8, n_in, n_in, generator=torch.Generator().manual_seed(n_in + n_out))
    ref32 = torch.nn.functional.interpolate(x, size=(n_out, n_out), mode="bilinear", align_corners=corners)
    up16 = torch.nn.functional.interpolate(x.half(), size=(n_out, n_out), mode="bilinear", align_corners=corners).float()
    ours16 = P.deterministic_resize(x.half(), (n_out, n_out), align_corners=corners).float()
    assert (ours16 - up16).abs().max().item() <= FP16_ATOL and (ours16 - ref32).abs().max().item() <= FP16_ATOL


@pytest.mark.parametrize("scale", [1, 2, 3, 6])
def test_f6_adaptive_pool_in_float16_matches_upstream_float16(scale):
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 16, 32, 32, generator=torch.Generator().manual_seed(scale))
    ref32 = torch.nn.AdaptiveAvgPool2d(scale)(x)
    up16 = torch.nn.AdaptiveAvgPool2d(scale)(x.half()).float()
    ours16 = P._deterministic_pool_class()(scale)(x.half()).float()
    assert (ours16 - up16).abs().max().item() <= FP16_ATOL and (ours16 - ref32).abs().max().item() <= FP16_ATOL


def test_embedded_modules_keep_no_relative_import(notebook):
    """The generator strips package-relative imports; a line it cannot strip would fail at import in the notebook."""
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    embedded = [c for c in code if c["metadata"].get("dimer", {}).get("embedded_module")]
    assert embedded
    for cell in embedded:
        source = "".join(cell["source"])
        assert not re.search(r"^\s*from \.", source, re.M), cell["metadata"]["dimer"]["embedded_module"]
