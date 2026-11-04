"""Versioned formatting replay: no model calls, changed rules, or source writes."""
import copy
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


@pytest.fixture
def rescore(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "evaluation"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("dabench_rescore_test", directory / "rescore_dabench_format.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question():
    return {"question": "Calculate the correlation.", "format": "@correlation[r_value]"}


def saved(answer="@correlation[r_value] → −0.87", raw_response="@correlation[−0.87]", **extra):
    return {"raw_text": answer, "formatter": {"raw_response": raw_response, "response": "@correlation[r_value]", **extra}}


def test_exact_public_placeholder_removed_and_unicode_minus_normalized(rescore):
    result = rescore.replay_existing_response(question(), saved())
    assert result["response"] == "@correlation[-0.87]"
    assert result["removed_placeholders"] == [{"key": "correlation", "placeholder": "r_value"}]
    assert result["minus_signs_normalized"] == 2
    assert result["new_model_calls"] == result["new_tokens"] == 0


def test_an_actual_wrong_answer_is_never_replaced(rescore):
    result = rescore.replay_existing_response(question(), saved("@correlation[-0.9] → −0.87"))
    assert result["response"] == "@correlation[-0.9]"
    assert not result["removed_placeholders"]


def test_placeholder_matching_is_exact_and_key_specific(rescore):
    prepared, removed = rescore.remove_format_placeholders(
        "@correlation[r_value] @different[r_value] @correlation[r_value2]", question()["format"])
    assert prepared == "  @different[r_value] @correlation[r_value2]"
    assert removed == [{"key": "correlation", "placeholder": "r_value"}]


def test_almost_matching_placeholder_remains_an_original_answer(rescore):
    result = rescore.replay_existing_response(question(), saved("@correlation[r_value2] → −0.87"))
    assert result["response"] == "@correlation[r_value2]"


def test_unicode_minus_in_an_actual_answer_is_representation_only(rescore):
    result = rescore.replay_existing_response(question(), saved("@correlation[−0.87]"))
    assert result["response"] == "@correlation[-0.87]"
    assert not result["removed_placeholders"]


def test_replay_keeps_existing_numeric_guard_against_calculation(rescore):
    result = rescore.replay_existing_response(question(), saved("Inputs are 3 and 4, no result.", "@correlation[7]"))
    assert result["response"] == "__NO_ANSWER__"
    assert result["rejected"]


@pytest.mark.parametrize("raw_response", [None, "", " "])
def test_missing_saved_response_never_invokes_even_the_local_replay(rescore, raw_response):
    def not_called(*args, **kwargs):
        raise AssertionError("No saved response may be reconstructed")
    result = rescore.replay_existing_response(question(), saved(raw_response=raw_response), formatter=not_called)
    assert not result["replayed"]
    assert result["response"] == "@correlation[r_value]"


def test_replay_inputs_exclude_all_oracle_and_artifact_fields(rescore):
    class BlindRecord(dict):
        def get(self, key, default=None):
            assert key in {"formatter", "raw_text"}, "Formatting must not read score or label fields"
            return super().get(key, default)

    record = BlindRecord({**saved(), "normalized": {"label_answers": "SECRET"}, "artifacts": "PRIVATE"})

    def formatter(model, *, question, answer):
        assert set(question) == {"question", "format"}
        assert "SECRET" not in answer and "PRIVATE" not in answer
        message = model.invoke([SimpleNamespace(content="system"), SimpleNamespace(content=json.dumps({
            "question": question["question"], "requested_format": question["format"], "analyst_answer": answer}))])
        assert message.usage_metadata is None
        return {"response": message.content}

    result = rescore.replay_existing_response({**question(), "common_answers": "SECRET"}, record, formatter=formatter)
    assert result["response"] == "@correlation[-0.87]"


def simple_scorer():
    def compare(actual, expected):
        try:
            return actual == expected or abs(float(actual) - float(expected)) < 1e-6
        except (ValueError, TypeError):
            return False

    def score(labels, responses):
        rows = []
        for label in labels:
            response = next((r for r in responses if r["id"] == label["id"] and r["response"]), None)
            if response:
                predicted = dict(re.findall(r"@(\w+)\[(.*?)\]", response["response"]))
                rows.append({"id": label["id"], "correctness": {key: compare(predicted.get(key), value)
                                                               for key, value in label["common_answers"]}})
        return rows

    return SimpleNamespace(evaluate_responses=score,
        evaluate_accuracy_by_question=lambda results: sum(all(r["correctness"].values()) for r in results) / len(results) if results else 0,
        evaluate_accuracy_by_sub_question=lambda results: sum(sum(r["correctness"].values()) for r in results) / sum(len(r["correctness"]) for r in results) if results else 0,
        evaluate_accuracy_proportional_by_sub_question_adjusted=lambda results: sum(sum(r["correctness"].values()) / len(r["correctness"]) for r in results) / len(results) if results else 0)


def batch_fixture(rescore, tmp_path, monkeypatch, *, raw_response="@correlation[−0.87]", originally_passed=False):
    import dabench
    batch = tmp_path / "batch"
    batch.mkdir()
    evaluator = Path(rescore.__file__).parent
    dependencies = ["answer_format.py", "dabench.py", "benchmark_statistics.py"]
    source_contract = {"evaluator_files": {name: rescore.digest((evaluator / name).read_bytes()) for name in dependencies}}
    rescore.write_json(batch / "contract.json", source_contract)
    fingerprint = rescore.digest((batch / "contract.json").read_bytes())
    case = {"id": "DA1", "official_id": 1, "dataset": "data.csv", "level": "medium", "concepts": []}
    rescore.write_json(batch / "suite.json", {"development": [case], "heldout": [case], "questions": {"DA1": question()}})
    (batch / "oracle").mkdir()
    (batch / "oracle/labels.jsonl").write_text(json.dumps({"id": 1, "common_answers": [["correlation", "-0.87"]]}) + "\n", encoding="utf-8")
    scorer = simple_scorer()
    for arm in ["off", "on"]:
        directory = batch / "heldout/trials" / f"DA1-1-{arm}"
        old_score = {"id": 1, "correctness": {"correlation": originally_passed}}
        formatter = saved(raw_response=raw_response)["formatter"]
        formatter.update(usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}, called=True)
        original = {"case": "DA1", "dataset": "data.csv", "repeat": 1, "arm": arm,
                    "contract_sha256": fingerprint, "status": "completed", "delivery_completed": True,
                    "answer_correct": originally_passed, "raw_answer_correct": False, "end_to_end": originally_passed,
                    "known_total_tokens": 100, "unknown_usage_calls": 0, "skill_loads": [],
                    "formatter": formatter, "normalized_score": old_score}
        rescore.write_json(directory / "summary.json", original)
        rescore.write_json(directory / "scoring.json", {**saved(raw_response=raw_response), "formatter": formatter, "normalized": old_score})
    monkeypatch.setattr(dabench, "verify_contract", lambda *args: copy.deepcopy(source_contract))
    monkeypatch.setattr(dabench, "load_scorer", lambda *args: scorer)
    return batch


def test_versioned_rescore_keeps_originals_costs_and_full_planned_denominators(rescore, tmp_path, monkeypatch):
    batch = batch_fixture(rescore, tmp_path, monkeypatch)
    originals = {p: p.read_bytes() for p in batch.rglob("*.json")}
    report = rescore.rescore_batch(batch)
    assert not report["complete"]
    assert report["v2_original"]["planned_trials"] == report["v3_rescored"]["planned_trials"] == 6
    assert report["original_trial_records"] == 2
    assert report["upgraded_trials"] == 2 and report["downgraded_trials"] == 0
    for arm in ["off", "on"]:
        assert report["official"][arm]["v2_original"]["ABQ"] == 0
        assert report["official"][arm]["v3_rescored"]["ABQ"] == pytest.approx(1 / 3)
        assert report["v2_original"]["arms"][arm]["known_total_tokens"] == report["v3_rescored"]["arms"][arm]["known_total_tokens"] == 100
        output = rescore.read_json(batch / "rescored-format-v3/trials" / f"DA1-1-{arm}/summary.json")
        assert output["formatter"]["usage"]["total_tokens"] == 5
    assert report["new_model_calls"] == report["new_tokens"] == 0
    for path, contents in originals.items():
        assert path.read_bytes() == contents


def test_resuming_rescore_reuses_existing_replays(rescore, tmp_path, monkeypatch):
    batch = batch_fixture(rescore, tmp_path, monkeypatch)
    first = rescore.rescore_batch(batch)

    def should_not_repeat(*args, **kwargs):
        raise AssertionError("A completed formatting replay should be reused")

    monkeypatch.setattr(rescore, "replay_existing_response", should_not_repeat)
    assert rescore.rescore_batch(batch) == first


def test_rescore_refuses_source_record_changed_after_replay(rescore, tmp_path, monkeypatch):
    batch = batch_fixture(rescore, tmp_path, monkeypatch)
    rescore.rescore_batch(batch)
    path = batch / "heldout/trials/DA1-1-off/scoring.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after rescore"):
        rescore.rescore_batch(batch)


def test_rescore_refuses_tampered_cached_outputs(rescore, tmp_path, monkeypatch):
    batch = batch_fixture(rescore, tmp_path, monkeypatch)
    rescore.rescore_batch(batch)
    path = batch / "rescored-format-v3/trials/DA1-1-off/summary.json"
    value = rescore.read_json(path)
    value["answer_correct"] = False
    rescore.write_json(path, value)
    with pytest.raises(ValueError, match="output changed"):
        rescore.rescore_batch(batch)


def test_rescore_refuses_rule_changes_in_existing_destination(rescore, tmp_path, monkeypatch):
    batch = batch_fixture(rescore, tmp_path, monkeypatch)
    rescore.rescore_batch(batch)
    monkeypatch.setattr(rescore, "RULES", {**rescore.RULES, "minus": "A different rule"})
    with pytest.raises(ValueError, match="rules or source contract changed"):
        rescore.rescore_batch(batch)


@pytest.mark.parametrize("originally_passed", [True, False])
def test_missing_formatter_response_preserves_original_score(rescore, tmp_path, monkeypatch, originally_passed):
    batch = batch_fixture(rescore, tmp_path, monkeypatch, raw_response=None, originally_passed=originally_passed)
    report = rescore.rescore_batch(batch)
    assert report["upgraded_trials"] == report["downgraded_trials"] == 0
    for arm in ["off", "on"]:
        assert report["official"][arm]["v2_original"] == report["official"][arm]["v3_rescored"]


def test_partial_scoring_file_cannot_upgrade_worker_failure_without_formatter_response(rescore, tmp_path, monkeypatch):
    batch = batch_fixture(rescore, tmp_path, monkeypatch, raw_response=None)
    for path in (batch / "heldout/trials").glob("*/summary.json"):
        value = rescore.read_json(path)
        value.pop("normalized_score")
        value.update(status="error", delivery_completed=False, answer_correct=False, end_to_end=False)
        rescore.write_json(path, value)
        scoring_path = path.with_name("scoring.json")
        scoring_value = rescore.read_json(scoring_path)
        scoring_value["normalized"] = {"id": 1, "correctness": {"correlation": True}}
        rescore.write_json(scoring_path, scoring_value)
    report = rescore.rescore_batch(batch)
    assert report["upgraded_trials"] == 0
    assert report["v3_rescored"]["arms"]["off"]["answer_correct"] == 0


def test_format_rule_phase_precedes_reading_labels(rescore, tmp_path, monkeypatch):
    batch = batch_fixture(rescore, tmp_path, monkeypatch)
    original_read = Path.read_text
    labels_read = []

    def read(path, *args, **kwargs):
        if path.name == "labels.jsonl":
            labels_read.append(True)
        return original_read(path, *args, **kwargs)

    original_replay = rescore.replay_existing_response

    def replay(*args, **kwargs):
        assert not labels_read
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(rescore, "replay_existing_response", replay)
    rescore.rescore_batch(batch)
    assert labels_read
