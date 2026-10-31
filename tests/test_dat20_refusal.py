"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both from the generated notebook:
- the router (Section 1 install-cell tail, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as
  one line, while every other exception keeps its traceback and `DIMER_MASK_FRESH_TEST` is still
  forwarded with each cell;
- the Section 4 BYOD cell turns every user-input problem into `BYODRefusal` and removes the earlier
  run's results from the session before it stops.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from prithvi_flood_segmentation_pipeline import check_split_disjoint, load_byod_dataset, split_dataset, validate_dataset
from prithvi_flood_segmentation_pipeline.pipeline import MIN_RECORDS
from prithvi_flood_segmentation_pipeline.samples import ROLES

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_flood_segmentation_colab.ipynb"
BYOD_CELL_ID = "prithvi_flood_segmentation-18"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _router_cell(cells: list[dict]) -> str:
    return next(_source(c) for c in cells if '_WORKER_SOURCE = r"""' in _source(c))


def _router_namespace(cells: list[dict], start: str) -> dict:
    source = _router_cell(cells)
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(cells, messages):
    namespace = _router_namespace(cells, "class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def test_router_relays_a_refusal_as_one_line_without_a_traceback(cells, capsys, monkeypatch):
    """Acceptance: fails on main, whose router has no refusal channel."""
    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    namespace, runtime = _kernel_runtime(cells, [("refusal", "etc: has no pairs.csv")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: etc: has no pairs.csv"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n", "1")]  # R4: the fresh-test mask still travels with each cell


def test_router_keeps_the_traceback_of_a_genuine_error(cells, capsys):
    namespace, runtime = _kernel_runtime(cells, [("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(cells, capsys, monkeypatch):
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    _, runtime = _kernel_runtime(cells, [("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")
    assert runtime._send.sent == [("run", "print('hello')\n", None)]


def test_router_layout_is_r4(cells):
    """The router stays the tail of the install cell, and the refusal branch sits before the error branch."""
    source = _router_cell(cells)
    assert "# dimer: kernel cell" in source and "pip" in source[: source.index('_WORKER_SOURCE = r"""')]
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    assert "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/." in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


# --- router, end to end with a real worker (POSIX: the router uses pass_fds and start_new_session) ----------------


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(cells, capsys):
    namespace = _router_namespace(cells, '_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('etc: has no pairs.csv')\n")
        assert caught.value._render_traceback_() == ["Refused: etc: has no pairs.csv"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


# --- Section 4 BYOD cell ------------------------------------------------------------------------------------------


def _byod_cell(cells: list[dict]) -> dict:
    return next(c for c in cells if "if USE_BYOD:" in _source(c))


def _byod_code(cells: list[dict]) -> str:
    source = _source(_byod_cell(cells))
    return source[source.index("def byod_minimum_records(") : source.index("else:\n    splits = fetch_sample_dataset")]


EARLIER = {"splits": "old", "test_records": "old", "frozen_test": "old", "adapted_test": "old", "fresh_result": "old", "result_payload": "old"}


def _cases(tmp_path: Path) -> dict:
    etc = tmp_path / "etc"
    etc.mkdir()
    (etc / "hostname").write_text("runtime\n", encoding="utf-8")
    bad_zip = tmp_path / "chips.zip"
    bad_zip.write_bytes(b"not a zip")
    no_table = tmp_path / "empty.zip"
    with zipfile.ZipFile(no_table, "w") as archive:
        archive.writestr("a.tif", b"x")
    listed = tmp_path / "listed.zip"
    with zipfile.ZipFile(listed, "w") as archive:
        archive.writestr("pairs.csv", "id,image,label\nc1,c1.tif,c1_label.tif\n")
        archive.writestr("c1.tif", b"not a tiff")
        archive.writestr("c1_label.tif", b"not a tiff")
    return {
        "folder without pairs.csv (the scripted /etc run)": (str(etc), "etc", "has no pairs.csv"),
        "plain file": (str(etc / "hostname"), "hostname", "give a .zip or a folder"),
        "missing path": (str(tmp_path / "nope.zip"), "nope.zip", "does not exist"),
        "corrupt zip": (str(bad_zip), "chips.zip", "File is not a zip file"),
        "zip without pairs.csv": (str(no_table), "empty.zip", "has no pairs.csv"),
        "unreadable chip": (str(listed), "listed.zip", "is not a readable GeoTIFF"),
    }


@pytest.mark.parametrize(
    "case",
    ["folder without pairs.csv (the scripted /etc run)", "plain file", "missing path", "corrupt zip", "zip without pairs.csv", "unreadable chip"],
)
def test_byod_input_problems_are_one_refused_line_and_clear_earlier_results(cells, tmp_path, monkeypatch, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    path, file_name, rule = _cases(tmp_path)[case]
    namespace: dict = {}
    with pytest.raises(ValueError) as caught:
        exec(compile(_byod_code(cells), "<byod>", "exec"), _seed(namespace, path))
    exc = caught.value
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: ") and rule in line and "\n" not in line
    assert "run this cell again" in line and "earlier results of this session were cleared" in line
    assert not set(EARLIER) & set(namespace), "an earlier run's results survived the refusal"
    assert namespace["pipe"] == "kept"


def _seed(namespace: dict, path: str) -> dict:
    namespace.update(
        {
            "USE_BYOD": True, "BYOD_PATH": path, "Path": Path, "csv": __import__("csv"), "io": io, "zipfile": zipfile,
            "random": __import__("random"), "tifffile": __import__("tifffile"), "MIN_RECORDS": MIN_RECORDS, "ROLES": ROLES,
            "load_byod_dataset": load_byod_dataset, "validate_dataset": validate_dataset, "split_dataset": split_dataset,
            "check_split_disjoint": check_split_disjoint, "pipe": "kept", **EARLIER,
        }
    )
    return namespace


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [(None, "(no file)", "received 0"), ({"scenes.tar": b""}, "scenes.tar", "upload one .zip")])
def test_cancelled_or_wrong_upload_is_one_refused_line(cells, tmp_path, monkeypatch, uploaded, file_name, rule):
    monkeypatch.chdir(tmp_path)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace: dict = {}
    with pytest.raises(ValueError) as caught:
        exec(compile(_byod_code(cells), "<byod>", "exec"), _seed(namespace, ""))
    (line,) = caught.value._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: ") and rule in line
    assert not set(EARLIER) & set(namespace)


def test_genuine_internal_error_keeps_its_traceback(cells, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    folder = tmp_path / "chips"
    folder.mkdir()
    (folder / "pairs.csv").write_text("id,image,label\n", encoding="utf-8")
    namespace = _seed({}, str(folder))

    def broken(_path):
        raise RuntimeError("internal bug")

    namespace["load_byod_dataset"] = broken
    code = _byod_code(cells).replace("        byod_rows = read_byod_table(byod_path)\n", "        byod_rows = []\n")
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        exec(compile(code, "<byod>", "exec"), namespace)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert namespace["frozen_test"] == "old"  # nothing is cleared for an internal error


def test_byod_refusal_is_carried_verbatim_and_the_cell_id_is_stable(cells):
    cell = _byod_cell(cells)
    assert cell["id"] == BYOD_CELL_ID
    carried = (
        "class BYODRefusal(ValueError):\n"
        '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
        "    dimer_refusal = True\n\n"
        "    def __init__(self, file_name, reason):\n"
        '        super().__init__(f"{file_name}: {reason}")\n'
        "        self.file_name = file_name\n"
        "        self.reason = reason\n\n"
        "    def _render_traceback_(self):\n"
        '        return [f"Refused: {self}"]\n'
    )
    assert carried in _source(cell)


def test_default_path_does_not_enter_the_refusal_wrap(cells):
    source = _source(_byod_cell(cells))
    wrap = source[source.index("if USE_BYOD:") : source.index("else:\n    splits = fetch_sample_dataset")]
    assert "fetch_sample_dataset" not in wrap and "refuse_byod(" in wrap


@pytest.mark.skipif(not Path("/etc").is_dir() or Path("/etc/pairs.csv").exists(), reason="needs a POSIX /etc without pairs.csv, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_directory(cells, monkeypatch):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc': a folder present in every fresh runtime that has no pairs.csv."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    namespace: dict = {}
    with pytest.raises(ValueError) as caught:
        exec(compile(_byod_code(cells), "<byod>", "exec"), _seed(namespace, "/etc"))
    (line,) = caught.value._render_traceback_()
    assert line.startswith("Refused: etc: has no pairs.csv: add one with the columns id, image, label")
