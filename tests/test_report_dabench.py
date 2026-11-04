"""Public benchmark reports exclude answers, keys, local paths and raw logs."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "evaluation"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("report_dabench_test", root / "report_dabench.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def batch(module, tmp_path):
    from benchmark_statistics import summarize_trials
    root = tmp_path / "batch"
    suite = {"heldout": [{"id": "DA1", "dataset": "public.csv", "level": "medium",
                           "concepts": ["Summary Statistics"], "task": "DO_NOT_PUBLISH_TASK"}]}
    contract = {"implementation_sha256": "a" * 64, "model": {"model_id": "public-model", "api_key": "SECRET_KEY_SENTINEL"},
                "options": {"temperature": 0, "main_max_tokens": 4096},
                "private_path": "C:/DO_NOT_PUBLISH_PATH"}
    write_json(root / "suite.json", suite)
    write_json(root / "contract.json", contract)
    fingerprint = hashlib.sha256((root / "contract.json").read_bytes()).hexdigest()
    rows = []
    for arm in ("off", "on"):
        row = dict(case="DA1", dataset="public.csv", arm=arm, repeat=1, contract_sha256=fingerprint,
                   answer_correct=arm == "on", raw_answer_correct=False, delivery_completed=True,
                   end_to_end=arm == "on", status="completed", known_total_tokens=100,
                   unknown_usage_calls=0, elapsed_seconds=5, skill_loads=[],
                   record_path="C:/DO_NOT_PUBLISH_RECORD", normalized_score={"label_answers": "DO_NOT_PUBLISH_LABEL"},
                   formatter={"called": True, "response": "@answer[DO_NOT_PUBLISH_ANSWER]", "usage": {"total_tokens": 20}})
        rows.append(row)
        write_json(root / f"heldout/trials/DA1-1-{arm}/summary.json", row)
    aggregate = summarize_trials(rows, planned_cases=suite["heldout"], repetitions=1)
    aggregate.update(split="heldout", contract_sha256=fingerprint, dataset_revision="b" * 40, scorer_revision="c" * 40,
                     private_field="DO_NOT_PUBLISH_AGGREGATE")
    for arm in ("off", "on"):
        value = aggregate["arms"][arm]
        value.update(raw_answer_correct=0, correct_subanswers=int(arm == "on"), raw_correct_subanswers=0,
                     total_subanswers=1, no_internal_repair_attained=0, code_attempts=1,
                     official_raw=dict(ABQ=0, PASQ=0, UASQ=0),
                     official_blind_reformat=dict(ABQ=int(arm == "on"), PASQ=int(arm == "on"), UASQ=int(arm == "on")),
                     formatter_known_total_tokens=20, formatter_unknown_usage_calls=0, formatter_calls=1)
    write_json(root / "heldout/report.json", aggregate)
    return root


def test_publish_whitelists_only_safe_numbers_and_official_metadata(module, batch, tmp_path):
    destination = tmp_path / "publish"
    report = module.publish(batch, destination)
    assert report["complete"]
    assert report["arms"]["off"]["answer_correct"] == 0
    assert report["arms"]["on"]["end_to_end"] == 1
    assert report["arms"]["on"]["format_rescued"] == 1
    assert report["by_case"][0]["dataset"] == "public.csv"
    assert report["arms"]["off"]["failure_categories"]["answer_mismatch"] == 1
    for name in ("skills-dabench.json", "skills-dabench.md"):
        text = (destination / name).read_text(encoding="utf-8")
        assert "DO_NOT_PUBLISH" not in text
        assert "SECRET_KEY_SENTINEL" not in text
        assert str(batch) not in text
    markdown = (destination / "skills-dabench.md").read_text(encoding="utf-8")
    for section in ("OFF 原始", "ON 原始", "OFF 盲整理", "ON 盲整理", "PASQ", "UASQ", "1e-6", "未全面覆盖"):
        assert section in markdown


def test_report_rejects_new_trial_since_last_aggregate(module, batch):
    path = batch / "heldout/trials/DA1-1-on/summary.json"
    row = json.loads(path.read_text())
    row.update(answer_correct=False, end_to_end=False)
    write_json(path, row)
    with pytest.raises(ValueError, match="stale or inconsistent"):
        module.build_public_report(batch)


def test_mixed_contract_is_rejected(module, batch):
    path = batch / "heldout/trials/DA1-1-on/summary.json"
    row = json.loads(path.read_text())
    row["contract_sha256"] = "wrong"
    write_json(path, row)
    with pytest.raises(ValueError, match="mixed contracts"):
        module.build_public_report(batch)


def test_incomplete_batch_cannot_be_published_as_final(module, batch):
    from benchmark_statistics import summarize_trials
    (batch / "heldout/trials/DA1-1-on/summary.json").unlink()
    off = json.loads((batch / "heldout/trials/DA1-1-off/summary.json").read_text())
    cases = json.loads((batch / "suite.json").read_text())["heldout"]
    aggregate = json.loads((batch / "heldout/report.json").read_text())
    partial = summarize_trials([off], planned_cases=cases, repetitions=1)
    for arm in ("off", "on"):
        extra = aggregate["arms"][arm]
        partial["arms"][arm] = {**extra, **partial["arms"][arm]}
    aggregate.update(partial)
    write_json(batch / "heldout/report.json", aggregate)
    with pytest.raises(ValueError, match="incomplete"):
        module.build_public_report(batch)
    intermediate = module.build_public_report(batch, allow_incomplete=True)
    assert not intermediate["complete"]
    assert "尚未完成，不作最终效果结论" in module.render_markdown(intermediate)


def test_only_official_basename_metadata_can_be_published(module):
    with pytest.raises(ValueError, match="basenames"):
        module._case_metadata(dict(id="DA1", dataset="C:/private.csv", level="easy", concepts=[]))


def test_observable_failure_tags_do_not_waive_failed_tasks(module):
    base = dict(end_to_end=False, answer_correct=False, delivery_completed=False)
    assert module.failure_categories({**base, "status": "awaiting_input"}) == ["clarification"]
    assert module.failure_categories({**base, "errors": ["worker_timeout"]}) == ["worker"]
    assert module.failure_categories({**base, "status": "error", "formatter": {"called": True, "response": "__NO_ANSWER__"}}) == ["protocol", "format"]


def test_blind_audit_preserves_risks_but_exports_no_raw_text(module, batch, tmp_path):
    suite_raw = (batch / "suite.json").read_bytes()
    audit = {"suite_sha256": hashlib.sha256(suite_raw).hexdigest(), "tasks": [
        {"id": "DA1", "severity": "high", "reason": "DO_NOT_PUBLISH_REASON", "source_text": "DO_NOT_PUBLISH_TEXT"}]}
    path = tmp_path / "audit.json"
    write_json(path, audit)
    report = module.build_public_report(batch, audit_path=path)
    summary = report["statement_audit"]
    assert summary["risk_counts"]["high"] == 1
    assert summary["high_risk_case_ids"] == ["DA1"]
    assert "DO_NOT_PUBLISH" not in json.dumps(report)


def test_audit_requires_proof_that_changed_suite_has_identical_tasks(module, batch, tmp_path):
    current = json.loads((batch / "suite.json").read_text())
    original = copy.deepcopy(current)
    original["old_protocol"] = True
    original_path = tmp_path / "audited-suite.json"
    write_json(original_path, original)
    audit = {"suite_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
             "tasks": [{"id": "DA1", "severity": "high"}]}
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, audit)
    with pytest.raises(ValueError, match="provide the original"):
        module.build_public_report(batch, audit_path=audit_path)
    report = module.build_public_report(batch, audit_path=audit_path, audited_suite_path=original_path)
    assert report["statement_audit"]["heldout_definitions_unchanged"]
    original["heldout"][0]["task"] = "Changed task"
    write_json(original_path, original)
    audit["suite_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
    write_json(audit_path, audit)
    with pytest.raises(ValueError, match="definitions changed"):
        module.build_public_report(batch, audit_path=audit_path, audited_suite_path=original_path)


@pytest.fixture
def rescore(batch):
    from benchmark_statistics import summarize_trials
    cases = json.loads((batch / "suite.json").read_text())["heldout"]
    original = [json.loads(p.read_text()) for p in sorted((batch / "heldout/trials").glob("*/summary.json"))]
    revised = copy.deepcopy(original)
    revised[0].update(answer_correct=True, end_to_end=True)
    aggregate = json.loads((batch / "heldout/report.json").read_text())
    destination = batch / "rescored-format-v3"
    contract = dict(version=3, source_contract_sha256=aggregate["contract_sha256"],
                    source_suite_sha256=hashlib.sha256((batch / "suite.json").read_bytes()).hexdigest(),
                    script_sha256="f" * 64, new_model_calls=0, new_tokens=0)
    write_json(destination / "contract.json", contract)
    report = dict(version=3, split="heldout", complete=True, original_trial_records=2,
                  source_contract_sha256=aggregate["contract_sha256"],
                  rescore_contract_sha256=hashlib.sha256((destination / "contract.json").read_bytes()).hexdigest(),
                  v2_original=summarize_trials(original, planned_cases=cases, repetitions=1),
                  v3_rescored=summarize_trials(revised, planned_cases=cases, repetitions=1),
                  official={arm: dict(v2_original=aggregate["arms"][arm]["official_blind_reformat"],
                                       raw_unchanged=aggregate["arms"][arm]["official_raw"],
                                       v3_rescored=dict(ABQ=1, PASQ=1, UASQ=1)) for arm in ("off", "on")},
                  changed_trials=[dict(case="DA1", repeat=1, arm="off", v2_answer_correct=False, v3_answer_correct=True)],
                  upgraded_trials=1, downgraded_trials=0, new_model_calls=0, new_tokens=0,
                  output_directory="DO_NOT_PUBLISH_RESCORE_PATH")
    write_json(destination / "report.json", report)
    return destination / "report.json"


def test_representation_rescore_is_separate_and_cost_not_counted_twice(module, batch, rescore):
    report = module.build_public_report(batch, rescore_path=rescore)
    assert report["arms"]["off"]["answer_correct"] == 0
    assert report["representation_rescore"]["arms"]["off"]["answer_correct"] == 1
    assert report["representation_rescore"]["new_model_calls"] == 0
    assert report["arms"]["off"]["known_total_tokens"] == 100
    text = module.render_markdown(report)
    assert "表示修正版 v3" in text and "不重新计算或选择性重跑任务" in text
    assert "DO_NOT_PUBLISH" not in json.dumps(report)


def test_rescore_cannot_silently_change_original_scores(module, batch, rescore):
    report = json.loads(rescore.read_text())
    report["official"]["off"]["raw_unchanged"]["ABQ"] = 1
    write_json(rescore, report)
    with pytest.raises(ValueError, match="raw official scores changed"):
        module.build_public_report(batch, rescore_path=rescore)


def test_rescore_with_new_inference_or_wrong_source_is_rejected(module, batch, rescore):
    report = json.loads(rescore.read_text())
    report["new_model_calls"] = 1
    write_json(rescore, report)
    with pytest.raises(ValueError, match="must not incur"):
        module.build_public_report(batch, rescore_path=rescore)
    report["new_model_calls"] = 0
    report["source_contract_sha256"] = "a" * 64
    write_json(rescore, report)
    with pytest.raises(ValueError, match="contract does not match"):
        module.build_public_report(batch, rescore_path=rescore)


@pytest.fixture
def interruption(batch, tmp_path):
    root = tmp_path / "skills-dabench-v2"
    for name in ("suite.json", "contract.json"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((batch / name).read_bytes())
    row = json.loads((batch / "heldout/trials/DA1-1-off/summary.json").read_text())
    write_json(root / "heldout/trials/DA1-1-off/summary.json", row)
    unknown = dict(case="DA1", dataset="public.csv", arm="off", repeat=2,
                   contract_sha256=row["contract_sha256"], answer_correct=False,
                   delivery_completed=False, end_to_end=False, status="error",
                   errors=["OpenAIConnectionError"])
    write_json(root / "heldout/trials/DA1-2-off/summary.json", unknown)
    marker = dict(status="stopped_environment_interruption", original_contract_sha256=row["contract_sha256"],
                  recorded_trials=2, planned_trials=6, longest_recorded_seconds=23385,
                  reason="wall-time jumps, OpenAIConnectionError, Docker preflight timeout, DO_NOT_PUBLISH_ERROR")
    write_json(root / "interruption.json", marker)
    return root


def test_environment_interrupted_batch_has_costs_but_no_success_rate(module, batch, interruption):
    report = module.build_public_report(batch, interrupted_batch_paths=[interruption])
    assert report["design"]["recorded_trials"] == 2
    assert report["arms"]["off"]["known_total_tokens"] == 100
    stopped = report["interrupted_batches"][0]
    assert stopped["recorded_trials"] == 2 and stopped["planned_trials"] == 6
    assert stopped["excluded_from_effect_comparison"]
    assert stopped["known_total_tokens"] == 100
    assert stopped["usage_call_counts_unmeasured_trials"] == 1
    assert stopped["transport_failure_trials"] is None
    assert stopped["transport_diagnostics_unmeasured_trials"] == 1
    assert "answer_correct" not in stopped and "end_to_end" not in stopped
    assert "DO_NOT_PUBLISH" not in json.dumps(report)
    assert "不按得分选择最佳批次" in module.render_markdown(report)


def test_transport_failure_counts_require_recorded_diagnostics(module, batch, interruption):
    path = interruption / "heldout/trials/DA1-1-off/summary.json"
    row = json.loads(path.read_text())
    row["errors"] = []
    write_json(path, row)
    stopped = module.build_public_report(batch, interrupted_batch_paths=[interruption])["interrupted_batches"][0]
    assert stopped["transport_failure_trials"] == 1
    assert stopped["transport_diagnostics_unmeasured_trials"] == 0


def test_interrupted_batch_must_have_same_pipeline_and_cannot_be_double_counted(module, batch, interruption):
    with pytest.raises(ValueError, match="distinct"):
        module.build_public_report(batch, interrupted_batch_paths=[interruption, interruption])
    source = json.loads((interruption / "contract.json").read_text())
    source["implementation_sha256"] = "b" * 64
    write_json(interruption / "contract.json", source)
    marker = json.loads((interruption / "interruption.json").read_text())
    marker["original_contract_sha256"] = hashlib.sha256((interruption / "contract.json").read_bytes()).hexdigest()
    write_json(interruption / "interruption.json", marker)
    with pytest.raises(ValueError, match="same frozen evaluation"):
        module.build_public_report(batch, interrupted_batch_paths=[interruption])


def test_report_describes_distinct_csv_files_without_claiming_family_independence(module, batch):
    report = module.build_public_report(batch)
    text = module.render_markdown(report)
    assert "份不同CSV路径" in text
    assert "份独立CSV" not in text and "按CSV隔离开发" not in text
    assert "CSV级划分不能保证底层数据家族完全隔离" in text
    assert "预训练污染" in text
    assert not report["dataset_family_boundary"]["complete_family_isolation_verified"]
    assert "CSV filename" in report["confidence_intervals"]["end_to_end"]["cluster_definition"]


def test_known_titanic_family_disclosure_includes_development_and_heldout_names(module):
    boundary = module._family_disclosure({
        "development": [{"dataset": "test_ave.csv"}],
        "heldout": [{"dataset": name} for name in ("titanic_train.csv", "titanic.csv", "test_x.csv", "other.csv")],
    })
    assert boundary["known_overlap_examples"][0]["family"] == "Titanic"
    assert boundary["known_overlap_examples"][0]["development_csv"] == "test_ave.csv"
    assert len(boundary["known_overlap_examples"][0]["heldout_csv"]) == 3
    assert not boundary["family_aware_split"]


def test_byte_and_known_family_sensitivity_does_not_replace_primary_scores(module, tmp_path):
    from benchmark_statistics import summarize_trials
    cases = [{"id": f"DA{i}", "dataset": name} for i, name in enumerate(
        ("titanic_train.csv", "titanic.csv", "test_x.csv", "other.csv"), 1)]
    suite = {"development": [{"dataset": "test_ave.csv"}], "heldout": cases}
    suite_raw = json.dumps(suite, sort_keys=True).encode()
    rows = [dict(case=case["id"], dataset=case["dataset"], repeat=1, arm=arm,
                 answer_correct=(arm == "on") if i < 2 else (arm == "off"),
                 delivery_completed=True, end_to_end=(arm == "on") if i < 2 else (arm == "off"))
            for i, case in enumerate(cases) for arm in ("off", "on")]
    original = copy.deepcopy(rows)
    baseline = summarize_trials(rows, planned_cases=cases, repetitions=1)
    audit = dict(suite_sha256=hashlib.sha256(suite_raw).hexdigest(), heldout_csv_paths=4,
                 heldout_unique_byte_hashes=3,
                 byte_identical_groups=[dict(sha256="a" * 64, csv_files=["titanic_train.csv", "titanic.csv"])],
                 known_family_groups=[dict(family="Titanic", development=["test_ave.csv"],
                                           heldout=["titanic_train.csv", "titanic.csv", "test_x.csv"])])
    path = tmp_path / "data-audit.json"
    write_json(path, audit)
    public, sensitivity = module._data_audit_summary(path, suite_raw, suite, rows, 1)
    assert public["heldout_csv_paths"] == 4 and public["heldout_unique_byte_hashes"] == 3
    assert sensitivity["identical_content"]["clusters"] == 3
    assert sensitivity["known_families"]["clusters"] == 2
    for name in ("identical_content", "known_families"):
        assert sensitivity[name]["confidence_intervals"]["end_to_end"]["on_minus_off"] == baseline["confidence_intervals"]["end_to_end"]["on_minus_off"]
    assert rows == original
    assert baseline == summarize_trials(rows, planned_cases=cases, repetitions=1)
    audit["heldout_unique_byte_hashes"] = 4
    write_json(path, audit)
    with pytest.raises(ValueError, match="unique-content count"):
        module._data_audit_summary(path, suite_raw, suite, rows, 1)
