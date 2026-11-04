"""Blind formatting-audit checks; oracle data must never enter review output."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[1] / "evaluation/audit_dabench_format.py"
    spec = importlib.util.spec_from_file_location("dabench_format_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scoring(raw="The mean is 34.65.", response="@mean[34.65]", **formatter):
    return {"raw_text": raw, "formatter": {"response": response, **formatter},
            "raw": {"label_answers": {"mean": "SECRET_LABEL"}},
            "normalized": {"label_answers": {"mean": "SECRET_LABEL"}},
            "deterministic": {"label_answers": {"mean": "SECRET_LABEL"}}}


def summary(**kwargs):
    return {"case": "DA1", "repeat": 1, "arm": "on", "status": "completed",
            "answer_correct": True, "raw_answer_correct": False, "delivery_completed": True,
            "label_answers": "FORBIDDEN_SUMMARY_FIELD", **kwargs}


def test_original_explicit_values_cannot_be_changed(audit):
    result = audit.inspect_trial(scoring("@mean[999]", "@mean[34.65]"), summary())
    assert "explicit_answer_changed" in result["kinds"]
    assert result["explicit_changes"] == [{"key": "mean", "original": "999", "formatted": "34.65"}]


def test_claimed_kept_original_key_cannot_disappear(audit):
    result = audit.inspect_trial(scoring("@mean[999]", "__NO_ANSWER__", kept_original_keys=["mean"]), summary())
    assert "explicit_answer_changed" in result["kinds"]
    assert result["explicit_changes"][0]["formatted"] is None


def test_new_numeric_values_are_flagged_without_recomputing_answer(audit):
    result = audit.inspect_trial(scoring("The inputs are 3 and 4; result unknown.", "@sum[7]"), summary())
    assert result["new_numeric_values"] == [{"key": "sum", "formatted": "7", "numbers_absent_from_original": ["7"]}]
    assert "new_numeric_value" in result["kinds"]


@pytest.mark.parametrize("raw", ["均值为34.65。", "The mean is 34.65.", "均值是 **34.65**。"])
def test_existing_numbers_are_not_marked_as_new(audit, raw):
    result = audit.inspect_trial(scoring(raw), summary())
    assert "new_numeric_value" not in result["kinds"]


def test_semantic_translation_records_actual_text_without_any_labels(audit):
    result = audit.inspect_trial(scoring("差异没有达到显著水平。", "@significant[no]", semantic_review_keys=["significant"]), summary())
    assert "semantic_translation_review" in result["kinds"]
    assert result["raw_text"] == "差异没有达到显著水平。"
    assert result["formatted_response"] == "@significant[no]"
    assert "SECRET_LABEL" not in json.dumps(result)
    assert "FORBIDDEN_SUMMARY_FIELD" not in json.dumps(result)


def test_answer_success_with_core_error_is_a_separate_review_candidate(audit):
    result = audit.inspect_trial(scoring(), summary(status="error", error={"code": "invalid_final_answer"}, delivery_completed=False))
    assert "answer_passed_but_core_incomplete" in result["kinds"]


def test_formatted_success_without_explicit_raw_format_is_reviewed(audit):
    result = audit.inspect_trial(scoring(), summary())
    assert "formatted_pass_without_explicit_raw_format" in result["kinds"]
    explicit = audit.inspect_trial(scoring("@mean[34.65]"), summary(raw_answer_correct=True))
    assert "formatted_pass_without_explicit_raw_format" not in explicit["kinds"]


def test_batch_does_not_touch_inputs_or_open_oracle_and_writes_only_review_outputs(audit, tmp_path, monkeypatch):
    trial = tmp_path / "heldout/trials/DA1-1-on"
    trial.mkdir(parents=True)
    path = trial / "scoring.json"
    path.write_text(json.dumps(scoring()), encoding="utf-8")
    (trial / "summary.json").write_text(json.dumps(summary()), encoding="utf-8")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "labels.jsonl").write_text("ORACLE_MUST_NOT_BE_READ", encoding="utf-8")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in trial.iterdir()}
    original_read = audit._read_json

    def read_allowed(path):
        assert Path(path).name in {"scoring.json", "summary.json"}
        assert "oracle" not in Path(path).parts
        return original_read(path)

    monkeypatch.setattr(audit, "_read_json", read_allowed)
    result = audit.audit_batch(tmp_path)
    assert result["audited_scoring_files"] == result["review_trials"] == 1
    assert result["oracle_accessed"] is False and result["scores_changed"] is False
    for path, sha in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    for name in ["format-audit.json", "format-review.jsonl"]:
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "SECRET_LABEL" not in text
        assert "ORACLE_MUST_NOT_BE_READ" not in text
        assert "FORBIDDEN_SUMMARY_FIELD" not in text


def test_scoring_written_before_summary_is_not_treated_as_core_failure(audit, tmp_path):
    trial = tmp_path / "heldout/trials/DA1-1-on"
    trial.mkdir(parents=True)
    (trial / "scoring.json").write_text(json.dumps(scoring()), encoding="utf-8")
    result = audit.audit_batch(tmp_path)
    assert result["scoring_without_summary"] == 1
    assert result["by_kind"]["answer_passed_but_core_incomplete"] == 0


def test_audit_continues_past_malformed_records_without_rewriting_them(audit, tmp_path):
    trial = tmp_path / "heldout/trials/DA1-1-on"
    trial.mkdir(parents=True)
    (trial / "scoring.json").write_text("{broken", encoding="utf-8")
    result = audit.audit_batch(tmp_path)
    assert result["audited_scoring_files"] == 0
    assert result["issues"]
    assert (trial / "scoring.json").read_text(encoding="utf-8") == "{broken"
