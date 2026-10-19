"""Pre-registered validation grid, fresh-test reading rule and fail-closed score-once wiring for the tutorial
(docs/prithvi-flood-preregistration.md; SCORE_ONCE_RULE 1-8).

- `GRID` / `select_config`: three configs compared on the kept epoch's validation loss; a challenger replaces C0
  only by at least `SELECTION_MARGIN`. `grid_hash` makes two runs comparable digit for digit.
- `preregistration_check`: `preregistered` is computed over EVERY fixed setting; anything else is exploratory.
- `paired_bootstrap` / `reading_rule`: the one primary endpoint (pooled water IoU, selected minus frozen, on the fresh
  test) with a paired chip bootstrap and the +/-0.02 practical tolerance.
- `score_fresh_test`: scores the fresh test only under the pre-registered settings, labels the look, and writes the
  report and the ledger atomically. A failed fetch, decode, check or write is not a look; an unreadable ledger never
  crashes a run and never promotes a run to the evaluation.

The fresh test is scored only under the pre-registered settings; see docs/release-verification.md for the recorded
evaluation and its reproductions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import DETERMINISM_FLAGS, atomic_write_json
from .samples import FRESH_POOL_DIGEST, MASK_ENV, SAMPLE_LABEL_SOURCE, fresh_test_masked  # noqa: F401 (MASK_ENV re-exported)

GRID: tuple[dict[str, Any], ...] = (
    {"name": "C0", "lr": 1e-5, "trainable": "decoder"},
    {"name": "C1", "lr": 3e-5, "trainable": "decoder"},
    {"name": "C2", "lr": 1e-5, "trainable": "decoder+last_block"},
)
DEFAULT_CONFIG = "C0"
SELECTION_MARGIN = 0.005
# The hosted record's selection; None until the first hosted T4 run exists, so a CPU run trains C0 (exploratory).
RECORDED_SELECTION: str | None = None
# The first hosted pre-registered T4 evaluation (run id, revision); None until it exists. Once set, every scoring of
# the fresh test - including the first one in a new runtime - is a "reproduction".
RECORDED_EVALUATION: dict[str, Any] | None = None
N_BOOT = 10_000
BOOT_SEED = 0
PRACTICAL_TOLERANCE = 0.02
REQUIRED_DEVICE = "Tesla T4"
LABEL_EVALUATION = "pre-registered evaluation (this runtime)"
LABEL_REPRODUCTION = "reproduction"
LABEL_EXPLORATORY = "not scored (exploratory)"
# The pinned sample's dataset_manifest digest over {'train', 'validation', 'test'} (pre-registration §2 "digests as
# pinned": it follows from the SAMPLE_RECORDS pins and the fixed decoding).
SAMPLE_DATASET_DIGEST = "76b95fb0ce2c60b569ee143674672240a62a5e29cc6b2c603b8e9cf1faa27452"
# The notebook's look marker (`marker=LOOK_MARKER`): beside the fresh-test cache, OUTSIDE outputs/, so deleting
# outputs/ never resets a look.
LOOK_MARKER = Path("weights") / "sen1floods11_fresh" / "score_once_look_marker.json"

PREREGISTERED: dict[str, Any] = {
    "data_source": SAMPLE_LABEL_SOURCE,
    "splits": {"train": 24, "validation": 8, "exposed_development_test": 12, "fresh_test": 23},
    # dataset_manifest(train, validation, exposed test)['digest'] of the pinned sample as decoded: the same counts
    # with a swapped, duplicated or edited chip give another digest.
    "dataset_digest": SAMPLE_DATASET_DIGEST,
    "fresh_pool_digest": FRESH_POOL_DIGEST,
    "epochs": 4,
    "batch_size": 2,
    "seed": 0,
    "grid": [dict(c) for c in GRID],
    "selection_margin": SELECTION_MARGIN,
    "precision": "float16 autocast + GradScaler",
    "device": REQUIRED_DEVICE,
    "determinism": dict(DETERMINISM_FLAGS),
    "n_boot": N_BOOT,
    "boot_seed": BOOT_SEED,
    "masked": False,
}


def grid_config(name: str) -> dict[str, Any]:
    for entry in GRID:
        if entry["name"] == name:
            return dict(entry)
    raise ValueError(f"unknown grid config {name!r}; one of {[c['name'] for c in GRID]}")


def config_for(lr: float, trainable: str) -> str | None:
    """The grid config with this learning rate and scope, or None (an off-grid, exploratory choice)."""
    for entry in GRID:
        if entry["trainable"] == trainable and abs(entry["lr"] - lr) <= 1e-12:
            return entry["name"]
    return None


def kept_val_loss(history: Sequence[Mapping[str, Any]], best_epoch: int) -> float:
    return float(history[best_epoch]["val_loss"])


def select_config(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-registered selection: C0 unless a challenger's kept-epoch validation loss is lower by >= SELECTION_MARGIN;
    among challengers that clear it the lowest loss wins, C1 before C2 on an exact tie."""
    by_name = {r["name"]: r for r in rows}
    if [c["name"] for c in GRID] != [r["name"] for r in rows]:
        raise ValueError(f"grid incomplete or out of order: {[r['name'] for r in rows]}")
    base = float(by_name[DEFAULT_CONFIG]["kept_val_loss"])
    # rounded to 9 places so a gain of exactly the margin (e.g. 0.118 -> 0.113) counts, independent of float error
    cleared = [r for r in rows if r["name"] != DEFAULT_CONFIG and round(base - float(r["kept_val_loss"]), 9) >= SELECTION_MARGIN]
    if not cleared:
        return {"selected": DEFAULT_CONFIG, "rule": f"no challenger beat C0 by >= {SELECTION_MARGIN}; C0 kept"}
    order = {c["name"]: i for i, c in enumerate(GRID)}
    best = min(cleared, key=lambda r: (float(r["kept_val_loss"]), order[r["name"]]))
    rule = f"{best['name']} beat C0 by >= {SELECTION_MARGIN} (lowest loss among those that did)"
    return {"selected": best["name"], "rule": rule}


def grid_row(name: str, adapt_result: Mapping[str, Any]) -> dict[str, Any]:
    history = adapt_result["history"]
    return {
        "name": name,
        "lr": adapt_result["lr"],
        "trainable": adapt_result["trainable"],
        "val_loss": [round(float(h["val_loss"]), 6) for h in history],
        "best_epoch": int(adapt_result["best_epoch"]),
        "kept_val_loss": round(kept_val_loss(history, adapt_result["best_epoch"]), 6),
    }


def grid_hash(rows: Sequence[Mapping[str, Any]], selected: str) -> str:
    """SHA-256 over the grid rows (losses to 6 decimals) and the selection: two runs must print the same value."""
    canonical = json.dumps({"rows": [dict(r) for r in rows], "selected": selected}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def preregistration_check(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the run's actual settings with EVERY pre-registered setting. A device counts when its name contains
    the required one (e.g. 'Tesla T4'); every other setting must be equal. Missing settings differ."""
    differing = []
    for key, expected in PREREGISTERED.items():
        actual = settings.get(key, "<missing>")
        same = (isinstance(actual, str) and REQUIRED_DEVICE in actual) if key == "device" else actual == expected
        if not same:
            differing.append(key)
    return {"preregistered": not differing, "settings_differing": differing}


# ---- reading rule ---------------------------------------------------------------------------------------

def pooled_water_iou(confusions: Sequence[Any]) -> float:
    """Water IoU from the summed (unrounded) 2x2 confusion counts (rows truth, columns prediction)."""
    import numpy as np

    total = np.sum(np.asarray(confusions, dtype=np.float64), axis=0)
    tp, fp, fn = total[1, 1], total[0, 1], total[1, 0]
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def paired_bootstrap(
    frozen: Sequence[Any], selected: Sequence[Any], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict[str, Any]:
    """Pooled water IoU change (selected - frozen) with a paired 95 % percentile interval over chips."""
    import numpy as np

    a = np.asarray(frozen, dtype=np.float64)
    b = np.asarray(selected, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or len(a) == 0:
        raise ValueError("frozen and selected need one 2x2 confusion matrix per chip, same chips")
    delta = pooled_water_iou(b) - pooled_water_iou(a)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        draws[i] = pooled_water_iou(b[idx]) - pooled_water_iou(a[idx])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "n_chips": int(len(a)),
        "frozen_water_iou": pooled_water_iou(a),
        "selected_water_iou": pooled_water_iou(b),
        "delta": float(delta),
        "ci95": [float(low), float(high)],
        "n_boot": n_boot,
        "seed": seed,
    }


def reading_rule(delta: float, ci95: Sequence[float]) -> str:
    low, high = float(ci95[0]), float(ci95[1])
    if low > 0 and delta >= PRACTICAL_TOLERANCE:
        return "improved"
    if high < 0 and delta <= -PRACTICAL_TOLERANCE:
        return "worse"
    return "no detectable change"


def chip_confusions(pipe: Any, records: Sequence[Mapping[str, Any]], *, batch_size: int = 4) -> list[Any]:
    """One unrounded 2x2 confusion matrix per chip (ignore index excluded)."""
    from .metrics import confusion_matrix

    predictions = pipe.predict(records, batch_size=batch_size)["predictions"]
    return [confusion_matrix([p["mask"]], [r["label"]]).tolist() for p, r in zip(predictions, records, strict=True)]


# ---- score-once ledger ----------------------------------------------------------------------------------

def _paths(score_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "ledger": score_dir / f"{stem}_fresh_test_ledger.json",
        "report": score_dir / f"{stem}_fresh_test_report.json",
    }


def read_ledger(score_dir: str | Path, stem: str) -> tuple[list[dict[str, Any]] | None, str]:
    """(entries, state): state 'absent', 'ok' or 'unreadable' (entries None). Never raises."""
    path = _paths(Path(score_dir), stem)["ledger"]
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], "absent"
    except OSError:
        return None, "unreadable"
    try:
        entries = json.loads(raw)
    except ValueError:
        return None, "unreadable"
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return None, "unreadable"
    return entries, "ok"


def look_label(score_dir: str | Path, stem: str, marker: str | Path | None = None) -> dict[str, Any]:
    """The label the next completed fresh-test scoring gets. Fail-closed: anything that could mean an earlier look
    (a ledger entry, a report file, an unreadable ledger, the look marker outside outputs/, a recorded hosted
    evaluation) makes it a reproduction."""
    entries, state = read_ledger(score_dir, stem)
    report_exists = _paths(Path(score_dir), stem)["report"].exists()
    if RECORDED_EVALUATION is not None:
        return {"test_look": LABEL_REPRODUCTION, "why": "a hosted evaluation of record exists (RECORDED_EVALUATION)"}
    if marker is not None and (Path(marker).exists() or Path(marker).is_symlink()):
        why = f"an earlier look in this runtime (look marker {Path(marker).as_posix()})"
        return {"test_look": LABEL_REPRODUCTION, "why": why}
    if state == "unreadable":
        return {"test_look": LABEL_REPRODUCTION, "why": "the ledger is unreadable; never promoted to the evaluation"}
    if entries or report_exists:
        return {"test_look": LABEL_REPRODUCTION, "why": "an earlier look in this runtime"}
    return {"test_look": LABEL_EVALUATION, "why": "first completed pre-registered scoring in this runtime"}


def _next_numbered(score_dir: Path, stem: str, kind: str) -> Path:
    n = 1
    while (score_dir / f"{stem}_fresh_test_{kind}_{n}.json").exists():
        n += 1
    return score_dir / f"{stem}_fresh_test_{kind}_{n}.json"


def write_exploratory(score_dir: str | Path, stem: str, payload: Mapping[str, Any], check: Mapping[str, Any]) -> Path:
    """An exploratory record (validation / exposed development test only) in its own `_exploratory_<n>` file."""
    root = Path(score_dir)
    out = _next_numbered(root, stem, "exploratory")
    return atomic_write_json(
        out, {**payload, "test_look": LABEL_EXPLORATORY, "settings_differing": list(check["settings_differing"])}
    )


def score_fresh_test(
    *,
    settings: Mapping[str, Any],
    checks: Mapping[str, bool],
    score: Callable[[], Mapping[str, Any]],
    score_dir: str | Path,
    stem: str,
    exploratory_payload: Mapping[str, Any] | None = None,
    marker: str | Path | None = None,
) -> dict[str, Any]:
    """Score the fresh test once per call, only when every pre-registered setting matches and every check passed.

    `checks`: named booleans re-checked just before scoring (grid complete and matching, model state = selected
    system, fresh-pool digest). `score()` returns {'chips', 'frozen', 'selected'} (per-chip confusions); it fetches and
    decodes the fresh chips, so a failure there raises before anything is written (not a look).

    The mask is re-read from THIS process's environment (`DIMER_MASK_FRESH_TEST`), whatever `settings['masked']`
    says. The label is fixed before scoring. Once scoring has completed it is a look: the look marker (outside
    outputs/), the report and the ledger are then written in that order, each atomically, and a failed write is
    returned in `write_errors` instead of raised, so a completed look is never reported as "not a look"."""
    root = Path(score_dir)
    root.mkdir(parents=True, exist_ok=True)
    settings = {**settings, "masked": bool(settings.get("masked")) or fresh_test_masked()}
    check = preregistration_check(settings)
    failed = sorted(k for k, ok in checks.items() if not ok)
    if not check["preregistered"]:
        path = write_exploratory(root, stem, dict(exploratory_payload or {}), check)
        return {"test_look": LABEL_EXPLORATORY, "settings_differing": check["settings_differing"], "file": str(path)}
    if failed:
        raise RuntimeError(f"fresh test not scored: pre-scoring checks failed {failed}; this is not a look")
    label = look_label(root, stem, marker)  # decided before scoring, from the state before this look
    scored = score()
    boot = paired_bootstrap(scored["frozen"], scored["selected"])
    verdict = reading_rule(boot["delta"], boot["ci95"])
    paths = _paths(root, stem)
    target = paths["report"] if label["test_look"] == LABEL_EVALUATION else _next_numbered(root, stem, "reproduction")
    report = {
        "test_look": label["test_look"],
        "label_reason": label["why"],
        "settings_differing": [],
        "settings": dict(settings),
        "endpoint": "pooled water IoU, selected minus frozen, fresh test (23 chips, 8 events, within-event)",
        "bootstrap": boot,
        "verdict": verdict,
        "reading_rule": f"improved/worse only if the 95 % interval excludes 0 and |change| >= {PRACTICAL_TOLERANCE}",
        "power": (
            "low: 23 chips from 8 events, most adjacent to training chips; "
            "the chip bootstrap understates event-level uncertainty"
        ),
        "chips": list(scored["chips"]),
        "recorded_evaluation": RECORDED_EVALUATION,
        "see": "docs/release-verification.md for the evaluation of record and its reproductions",
    }
    write_errors: list[str] = []
    if marker is not None:
        try:  # first: the look survives a deleted outputs/ and a failed report or ledger write
            marker_path = Path(marker)
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(marker_path, {"stem": stem, "test_look": label["test_look"], "report": target.name})
        except OSError as exc:
            write_errors.append(f"look marker: {type(exc).__name__}: {exc}"[:200])
    try:
        atomic_write_json(target, report)
    except OSError as exc:
        write_errors.append(f"report: {type(exc).__name__}: {exc}"[:200])
    try:
        entries, state = read_ledger(root, stem)
        if state == "unreadable":  # keep the evidence; the new ledger starts from this look
            paths["ledger"].replace(_next_numbered(root, stem, "ledger_unreadable"))
        entry = {"test_look": label["test_look"], "file": target.name, "delta": boot["delta"], "verdict": verdict}
        atomic_write_json(paths["ledger"], [*(entries or []), entry])
    except OSError as exc:
        write_errors.append(f"ledger: {type(exc).__name__}: {exc}"[:200])
    out = {**report, "file": str(target)}
    if write_errors:
        out["write_errors"] = write_errors
    return out
