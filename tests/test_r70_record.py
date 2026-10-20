"""Row-70 evaluation of record (hosted Colab T4, 2026-10-19): the recorded constants, the reproduction label and the
comparison with the record. Pure Python: no torch, no network."""

import json
from pathlib import Path

import prithvi_flood_segmentation_pipeline.scoring as SC

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "execution-evidence" / "2026-10-19-9fd2b76"
NOTEBOOK = ROOT / "tutorials" / "prithvi_flood_segmentation_colab.ipynb"


def _record_result() -> dict:
    """A score_fresh_test result shaped like the record's, with unrounded bootstrap numbers."""
    return {
        "test_look": SC.LABEL_REPRODUCTION,
        "bootstrap": {
            "n_chips": 23,
            "frozen_water_iou": 0.881512,
            "selected_water_iou": 0.881556,
            "delta": 0.0000441,
            "ci95": [-0.002412, 0.002138],
        },
        "verdict": "no detectable change",
    }


def test_recorded_evaluation_is_the_hosted_t4_record():
    rec = SC.RECORDED_EVALUATION
    assert rec is not None
    assert rec["commit"] == "9fd2b76dd1480d8a8f822e5fe2d65ac578bba559"
    assert rec["notebook_blob"] == "f030a974de5b3adbaa017e830b162710f9693a23"
    assert rec["device"] == SC.REQUIRED_DEVICE == "Tesla T4"
    assert rec["test_look"] == SC.LABEL_EVALUATION
    assert rec["grid_hash"] == "68b0a351ae2db8d39ad89bfe2818a4d7012761106a6e958f86fde09b8bd9651a"
    assert (rec["selected_system"], rec["chips"]) == ("C0", 23)
    assert (rec["frozen_water_iou"], rec["selected_water_iou"], rec["change"]) == (0.8815, 0.8816, 0.0)
    assert rec["ci95"] == [-0.0024, 0.0021]
    assert rec["verdict"] == SC.reading_rule(rec["change"], rec["ci95"]) == "no detectable change"
    assert SC.RECORDED_SELECTION == rec["selected_system"] == SC.DEFAULT_CONFIG
    assert (ROOT / rec["evidence"]).resolve() == EVIDENCE.resolve()


def test_recorded_numbers_equal_the_committed_evidence():
    printed = json.loads((EVIDENCE / "prithvi_flood_segmentation_fresh_test_report.transcribed.json").read_text(encoding="utf-8"))
    fresh = printed["source_lines"]["fresh_test"]
    grid = printed["source_lines"]["grid"]
    rec = SC.RECORDED_EVALUATION
    assert fresh["test_look"] == rec["test_look"]
    for key in ("chips", "frozen_water_iou", "selected_water_iou", "change", "ci95", "verdict"):
        assert fresh[key] == rec[key], key
    assert grid["grid_hash"] == rec["grid_hash"] and grid["selection"]["selected"] == rec["selected_system"]
    assert printed["source_lines"]["preregistration_check"] == {
        "masked": False, "preregistered": True, "will_fetch_fresh_test": True, "settings_differing": []
    }
    exposed = printed["source_lines"]["exposed_development_test"]["water_iou"]
    assert exposed == {"baseline_no_water": 0.0, "frozen": 0.7301, "adapted": 0.7459}
    log = (EVIDENCE / "exec.log").read_text(encoding="utf-8")
    assert rec["grid_hash"] in log and "'verdict': 'no detectable change'" in log
    for run in ("masked-4e29bc1-run1", "masked-4e29bc1-run3"):
        suite = (EVIDENCE / run / "suite.log").read_text(encoding="utf-8")
        assert f"'grid_hash': '{rec['grid_hash']}'" in suite and "'will_fetch_fresh_test': False" in suite
    summary = json.loads((EVIDENCE / "run_summary.json").read_text(encoding="utf-8"))
    assert (summary["sha"], summary["blob"], summary["verdict"]) == (rec["commit"], rec["notebook_blob"], "PASS")


def test_with_the_record_the_first_scoring_in_a_new_runtime_is_a_reproduction(tmp_path):
    label = SC.look_label(tmp_path / "score_once", "stem", marker=tmp_path / "marker.json")
    assert label["test_look"] == SC.LABEL_REPRODUCTION
    assert "RECORDED_EVALUATION" in label["why"]


def test_compare_with_record_reports_a_match_and_each_differing_field(monkeypatch):
    rec = SC.RECORDED_EVALUATION
    same = SC.compare_with_record(_record_result(), rec["grid_hash"], "C0")
    assert same == {"record": "2026-10-19 hosted Tesla T4, commit 9fd2b76", "matches_record": True, "differs": []}
    other = _record_result()
    other["bootstrap"]["selected_water_iou"] = 0.8901
    other["verdict"] = "improved"
    diff = SC.compare_with_record(other, "0" * 64, "C2")
    assert diff["matches_record"] is False
    assert diff["differs"] == ["selected_system", "grid_hash", "selected_water_iou", "verdict"]
    monkeypatch.setattr(SC, "RECORDED_EVALUATION", None)
    assert SC.compare_with_record(_record_result(), rec["grid_hash"], "C0") is None


def test_notebook_prints_the_comparison_and_teaches_the_record():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "compare_with_record(fresh_result, GRID_HASH, selected_name)" in code
    assert "'matches_record': comparison_to_record['matches_record']" in code
    assert "no recorded number yet" not in text and "No hosted number is recorded" not in text
    for quoted in ("0.8815", "0.8816", "[−0.0024, 0.0021]", "0.7459", "did not show up", "2026-10-19-9fd2b76"):
        assert quoted in text, quoted
