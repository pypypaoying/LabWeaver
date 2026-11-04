"""Publish safe aggregate Skills/DABench results after the frozen batch finishes.

Read a completed heldout/report.json plus its matching trial summaries. Never
invoke models, read CSV contents, export answers, or change the frozen runner.
The explicit allowlist is intentionally independent of raw summary schemas.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from benchmark_statistics import summarize_trials


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("off", "on")
METRICS = ("answer_correct", "delivery_completed", "end_to_end")
SKILLS = ("data-exploration", "statistical-analysis", "data-visualization", "analysis-validation")
CATEGORIES = ("protocol", "answer_mismatch", "clarification", "format", "worker", "unclassified")
CATEGORY_LABELS = {
    "protocol": "运行协议或交付失败",
    "answer_mismatch": "已交付但官方答案不符",
    "clarification": "等待用户澄清",
    "format": "未形成可评分答案／整理失败",
    "worker": "工作进程或基础设施异常",
    "unclassified": "尚未归因",
}
ARM_COUNTERS = (
    "planned", "recorded", "missing", *METRICS, "raw_answer_correct",
    "correct_subanswers", "raw_correct_subanswers", "total_subanswers",
    "no_internal_repair_attained", "code_attempts", "known_total_tokens",
    "unknown_usage_calls", "usage_unmeasured_trials", "usage_call_counts_unmeasured_trials",
    "timing_unmeasured_trials", "skill_reads", "trials_reading_skills",
    "skill_reads_unmeasured_trials", "formatter_known_total_tokens",
    "formatter_unknown_usage_calls", "formatter_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value, *, integer=False):
    if value is None:
        return None
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or (integer and value < 0)):
        raise ValueError("Public aggregate contains an invalid numeric measurement")
    return value


def _fingerprint(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid public source fingerprint")
    return value


def _revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Invalid upstream revision")
    return value


def _csv_basename(dataset):
    if (not isinstance(dataset, str) or len(dataset) > 255 or not dataset.endswith(".csv")
            or any(character in dataset for character in ("/", "\\", ":", "\x00", "\n", "\r", "|"))):
        raise ValueError("Only official CSV basenames may appear in a public report")
    return dataset


def _case_metadata(case):
    if not re.fullmatch(r"DA[0-9]+", case["id"]):
        raise ValueError("Unexpected official case identifier")
    dataset = _csv_basename(case["dataset"])
    if case["level"] not in {"easy", "medium", "hard"}:
        raise ValueError("Unexpected official difficulty")
    concepts = case["concepts"]
    if not isinstance(concepts, list) or any(
        not isinstance(c, str) or not re.fullmatch(r"[A-Za-z0-9 ,()/&.\-]{1,100}", c)
        for c in concepts
    ):
        raise ValueError("Unexpected official concept metadata")
    return dict(id=case["id"], dataset=dataset, level=case["level"], concepts=concepts)


def failure_categories(row):
    """Observable diagnostic tags, not an automatic proof of a root cause."""
    if row["end_to_end"]:
        return []
    categories = []
    if row.get("errors"):
        categories.append("worker")
    if row.get("status") == "awaiting_input":
        categories.append("clarification")
    elif not row["delivery_completed"] and not row.get("errors"):
        categories.append("protocol")
    formatter = row.get("formatter") or {}
    response = formatter.get("response") or ""
    if not row["answer_correct"]:
        if formatter.get("error") or (formatter.get("called") and not re.search(r"@\w+\[", response)):
            categories.append("format")
        elif row["delivery_completed"]:
            categories.append("answer_mismatch")
    return categories or ["unclassified"]


def _safe_arms(aggregate, rows):
    result = {}
    for arm in ARMS:
        source = aggregate["arms"][arm]
        item = {key: _number(source.get(key), integer=True) for key in ARM_COUNTERS}
        item["median_seconds"] = _number(source.get("median_seconds"))
        for name in ("rates_observed", "rates_planned"):
            item[name] = {metric: _number(source[name][metric]) for metric in METRICS}
        for name in ("official_raw", "official_blind_reformat"):
            item[name] = {metric: _number(source[name][metric]) for metric in ("ABQ", "PASQ", "UASQ")}
        selected = [r for r in rows if r["arm"] == arm]
        item["format_rescued"] = sum(r["answer_correct"] and not r.get("raw_answer_correct", False) for r in selected)
        item["raw_pass_but_formatted_fail"] = sum(r.get("raw_answer_correct", False) and not r["answer_correct"] for r in selected)
        item["formatter_status_unmeasured_trials"] = sum("formatter" not in r for r in selected)
        item["failure_categories"] = {category: sum(category in failure_categories(r) for r in selected) for category in CATEGORIES}
        item["skill_reads_by_id"] = {skill: sum(load.get("skill_id") == skill for r in selected for load in (r.get("skill_loads") or [])) for skill in SKILLS}
        item["skill_reads_by_role"] = {role: sum(load.get("role") == role for r in selected for load in (r.get("skill_loads") or [])) for role in ("main", "analysis")}
        result[arm] = item
    return result


def _safe_paired(aggregate):
    return {metric: {key: _number(aggregate["paired"][metric][key], integer=True)
                     for key in ("off_only", "on_only", "both_pass", "both_fail", "incomplete")}
            for metric in METRICS}


def _safe_intervals(aggregate, *, cluster_definition="CSV filename; shared underlying data families are not excluded."):
    result = {}
    for metric in METRICS:
        source = aggregate["confidence_intervals"][metric]
        item = {key: _number(source[key]) for key in ("on_minus_off", "lower", "upper", "confidence_level")}
        item.update({key: _number(source[key], integer=True) for key in (
            "complete_pairs", "excluded_incomplete_pairs", "dataset_clusters", "bootstrap_samples", "seed")})
        item.update(method="dataset_cluster_percentile_bootstrap", resampling_unit="dataset",
                    cluster_definition=cluster_definition)
        result[metric] = item
    return result


def _audit_summary(audit_path, audited_suite_path, suite_raw, suite):
    if audit_path is None:
        return {"available": False}
    audit = read_json(audit_path)
    current_sha = hashlib.sha256(suite_raw).hexdigest()
    audited_sha = _fingerprint(audit["suite_sha256"])
    if audited_sha != current_sha:
        if audited_suite_path is None:
            raise ValueError("Audit refers to a different suite; provide the original audited suite")
        original_raw = Path(audited_suite_path).read_bytes()
        if hashlib.sha256(original_raw).hexdigest() != audited_sha:
            raise ValueError("Original audited suite fingerprint does not match")
        if json.loads(original_raw)["heldout"] != suite["heldout"]:
            raise ValueError("Heldout task definitions changed after the audit")
    ids = {c["id"] for c in suite["heldout"]}
    tasks = audit["tasks"]
    if len(tasks) != len(ids) or {t["id"] for t in tasks} != ids:
        raise ValueError("Audit does not cover the same unique heldout tasks")
    levels = ("none", "low", "medium", "high")
    if any(t["severity"] not in levels for t in tasks):
        raise ValueError("Unknown audit severity")
    return {
        "available": True,
        "audited_suite_sha256": audited_sha,
        "current_suite_sha256": current_sha,
        "heldout_definitions_unchanged": True,
        "risk_counts": {level: sum(t["severity"] == level for t in tasks) for level in levels},
        "high_risk_case_ids": sorted(t["id"] for t in tasks if t["severity"] == "high"),
        "policy": "Statement-only audit; no question changed, failure waived, or score increased.",
    }


def _batch_label(path):
    name = Path(path).name
    return name if re.fullmatch(r"skills-dabench-v[0-9]+", name) else "benchmark-batch"


def _family_disclosure(suite):
    development = {case["dataset"] for case in suite.get("development", [])}
    heldout = {case["dataset"] for case in suite["heldout"]}
    related = sorted(heldout & {"titanic_train.csv", "titanic.csv", "test_x.csv"})
    examples = []
    if "test_ave.csv" in development and related:
        examples.append({
            "family": "Titanic",
            "development_csv": "test_ave.csv",
            "heldout_csv": related,
            "basis": "Original column headers and dataset identity reviewed; same-source or derived CSVs.",
        })
    return {
        "split_unit": "CSV file",
        "family_aware_split": False,
        "complete_family_isolation_verified": False,
        "pretraining_contamination_excluded": False,
        "development_use": "Answer-format adapter development only; Agent prompts were not tuned on development tasks.",
        "known_overlap_examples": examples,
    }


def _data_audit_summary(path, suite_raw, suite, rows, repetitions):
    """Optional outcome-blind grouping sensitivity; leave primary CIs intact."""
    if path is None:
        return {"available": False}, {"available": False}
    raw = Path(path).read_bytes()
    audit = json.loads(raw)
    if audit.get("suite_sha256") != hashlib.sha256(suite_raw).hexdigest():
        raise ValueError("Data-family audit refers to a different frozen suite")
    cases = suite["heldout"]
    filenames = {c["dataset"] for c in cases}
    development = {c["dataset"] for c in suite.get("development", [])}
    if audit.get("heldout_csv_paths") != len(filenames):
        raise ValueError("Data-family audit has a different CSV path count")
    byte_groups, seen = [], set()
    byte_public = []
    for group in audit.get("byte_identical_groups", []):
        members = group["csv_files"]
        if (not isinstance(members, list) or len(members) < 2 or len(set(members)) != len(members)
                or not set(members) <= filenames or seen & set(members)):
            raise ValueError("Invalid byte-identical CSV group")
        seen.update(members)
        byte_groups.append(members)
        byte_public.append({"sha256": _fingerprint(group["sha256"]), "csv_files": sorted(members)})
    family_groups, family_public = [], []
    for group in audit.get("known_family_groups", []):
        name = group["family"]
        members = group["heldout"]
        prior = group.get("development", [])
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9 _-]{1,80}", name)
                or not isinstance(members, list) or not members or len(set(members)) != len(members)
                or not set(members) <= filenames or not isinstance(prior, list) or not set(prior) <= development):
            raise ValueError("Invalid known data-family group")
        for filename in prior:
            _csv_basename(filename)
        family_groups.append(members)
        family_public.append({"family": name, "development": sorted(prior), "heldout": sorted(members)})

    def grouping(groups):
        parents = {name: name for name in filenames}
        def find(name):
            while parents[name] != name:
                name = parents[name]
            return name
        for members in groups:
            representative = find(min(members))
            for member in members:
                parents[find(member)] = representative
        return {name: find(name) for name in filenames}

    content_map = grouping(byte_groups)
    family_map = grouping([*byte_groups, *family_groups])
    unique_content = len(set(content_map.values()))
    if audit.get("heldout_unique_byte_hashes") != unique_content:
        raise ValueError("Data-family audit has an inconsistent unique-content count")
    sensitivities = {"available": True, "scoring_basis": "Original frozen main-table scores only; no primary interval or outcome is replaced."}
    for name, mapping, definition in (
        ("identical_content", content_map, "Exact CSV byte identity; further semantic overlap remains possible."),
        ("known_families", family_map, "Exact byte identity plus explicitly identified data families; not exhaustive."),
    ):
        modified_cases = [{**case, "dataset": mapping[case["dataset"]]} for case in cases]
        modified_rows = [{**row, "dataset": mapping[row["dataset"]]} for row in rows]
        result = summarize_trials(modified_rows, planned_cases=modified_cases, repetitions=repetitions)
        sensitivities[name] = {"clusters": len(set(mapping.values())),
                               "confidence_intervals": _safe_intervals(result, cluster_definition=definition)}
    return {
        "available": True, "audit_sha256": hashlib.sha256(raw).hexdigest(),
        "suite_sha256": audit["suite_sha256"], "heldout_csv_paths": len(filenames),
        "heldout_unique_byte_hashes": unique_content,
        "byte_identical_groups": byte_public, "known_family_groups": family_public,
        "scope": "CSV bytes and headers only; no labels or model outcomes used for grouping. Not an exhaustive family audit.",
    }, sensitivities


def _rescore_summary(path, aggregate, batch, contract_sha):
    if path is None:
        return {"available": False}
    path = Path(path)
    rescored = read_json(path)
    contract_raw = path.with_name("contract.json").read_bytes()
    contract = json.loads(contract_raw)
    if (rescored.get("version") != 3 or rescored.get("split") != "heldout"
            or rescored.get("source_contract_sha256") != contract_sha
            or rescored.get("rescore_contract_sha256") != hashlib.sha256(contract_raw).hexdigest()
            or contract.get("source_contract_sha256") != contract_sha
            or contract.get("source_suite_sha256") != hashlib.sha256((batch / "suite.json").read_bytes()).hexdigest()):
        raise ValueError("Rescore source or versioned contract does not match the main batch")
    if any(source.get(key) != 0 for source in (rescored, contract) for key in ("new_model_calls", "new_tokens")):
        raise ValueError("Representation rescore must not incur new model calls or tokens")
    for key in ("complete", "planned_trials", "recorded_trials", "unique_cases", "unique_datasets", "repetitions", "paired"):
        if rescored["v2_original"].get(key) != aggregate.get(key):
            raise ValueError("Rescore original accounting differs from the main batch")
    if rescored["complete"] != aggregate["complete"] or rescored["original_trial_records"] != aggregate["recorded_trials"]:
        raise ValueError("Rescore does not cover the same recorded trials")
    revised = rescored["v3_rescored"]
    for key in ("complete", "planned_trials", "recorded_trials", "unique_cases", "unique_datasets", "repetitions"):
        if revised.get(key) != aggregate.get(key):
            raise ValueError("Rescore changed the planned task population")
    arms = {}
    official = {}
    for arm in ARMS:
        if rescored["official"][arm]["v2_original"] != aggregate["arms"][arm]["official_blind_reformat"]:
            raise ValueError("Original official scores changed in rescore report")
        if rescored["official"][arm]["raw_unchanged"] != aggregate["arms"][arm]["official_raw"]:
            raise ValueError("Original raw official scores changed in rescore report")
        original = rescored["v2_original"]["arms"][arm]
        for key in ("planned", "recorded", "answer_correct", "delivery_completed", "end_to_end"):
            if original[key] != aggregate["arms"][arm][key]:
                raise ValueError("Original outcome counts changed in rescore report")
        source = revised["arms"][arm]
        if source["delivery_completed"] != original["delivery_completed"]:
            raise ValueError("Representation rescore cannot change Agent delivery outcomes")
        arms[arm] = {key: _number(source[key], integer=True) for key in ("planned", "recorded", *METRICS)}
        official[arm] = {name: {metric: _number(rescored["official"][arm][name][metric]) for metric in ("ABQ", "PASQ", "UASQ")}
                         for name in ("v2_original", "v3_rescored", "raw_unchanged")}
    changes = rescored["changed_trials"]
    if len({(r["case"], r["repeat"], r["arm"]) for r in changes}) != len(changes):
        raise ValueError("Duplicate changed trial in representation rescore")
    known_cases = {c["id"] for c in read_json(batch / "suite.json")["heldout"]}
    for change in changes:
        if (change["case"] not in known_cases or change["arm"] not in ARMS
                or type(change["repeat"]) is not int or not 1 <= change["repeat"] <= aggregate["repetitions"]
                or type(change["v2_answer_correct"]) is not bool or type(change["v3_answer_correct"]) is not bool
                or change["v2_answer_correct"] == change["v3_answer_correct"]):
            raise ValueError("Invalid changed trial identity or outcome")
    upgraded = sum(c["v3_answer_correct"] for c in changes)
    downgraded = len(changes) - upgraded
    if (rescored["upgraded_trials"], rescored["downgraded_trials"]) != (upgraded, downgraded):
        raise ValueError("Representation rescore change counts disagree")
    return {
        "available": True, "version": 3, "complete": rescored["complete"],
        "source_contract_sha256": contract_sha,
        "rescore_contract_sha256": _fingerprint(rescored["rescore_contract_sha256"]),
        "script_sha256": _fingerprint(contract["script_sha256"]),
        "new_model_calls": 0, "new_tokens": 0,
        "upgraded_trials": upgraded, "downgraded_trials": downgraded,
        "arms": arms, "official": official,
        "paired": _safe_paired(revised), "confidence_intervals": _safe_intervals(revised),
        "policy": "Uniform no-model representation correction derived from development cases; original scores and Agent outcomes retained, original usage counted once.",
    }


def _interrupted_summaries(paths, batch, contract):
    results, seen = [], set()
    for value in paths or []:
        path = Path(value).resolve()
        if path == batch.resolve() or path in seen:
            raise ValueError("Interrupted batches must be distinct from the main batch and each other")
        seen.add(path)
        marker = read_json(path / "interruption.json")
        if marker.get("status") != "stopped_environment_interruption":
            raise ValueError("Interrupted batch lacks a confirmed environment interruption")
        raw = (path / "contract.json").read_bytes()
        source = json.loads(raw)
        fingerprint = hashlib.sha256(raw).hexdigest()
        if marker.get("original_contract_sha256") != fingerprint:
            raise ValueError("Interruption marker has a different original contract")
        for key in ("implementation_sha256", "inputs_sha256", "model", "options", "image", "evaluator_files", "upstream_dataset_revision", "upstream_scorer_revision"):
            if source.get(key) != contract.get(key):
                raise ValueError("Interrupted batch does not use the same frozen evaluation: " + key)
        rows = [read_json(p) for p in sorted((path / "heldout/trials").glob("*/summary.json"))]
        if any(r.get("contract_sha256") != fingerprint for r in rows):
            raise ValueError("Interrupted batch contains mixed contracts")
        cases = read_json(path / "suite.json")["heldout"]
        counts = summarize_trials(rows, planned_cases=cases, repetitions=3)
        if counts["complete"] or len(rows) != marker.get("recorded_trials") or counts["planned_trials"] != marker.get("planned_trials"):
            raise ValueError("Interrupted batch counts disagree with the preserved interruption record")
        known_usage = [r for r in rows if r.get("known_total_tokens") is not None]
        known_counts = [r for r in rows if r.get("unknown_usage_calls") is not None]
        formatters = [r["formatter"] for r in rows if isinstance(r.get("formatter"), dict)]
        called = [f for f in formatters if f.get("called")]
        measured_formatter = [f for f in called if (f.get("usage") or {}).get("total_tokens") is not None]
        reason = str(marker.get("reason", "")).lower()
        reason_codes = []
        if "wall-time" in reason or "sleep" in reason:
            reason_codes.append("host_time_jump_or_sleep")
        if "connectionerror" in reason or "transport" in reason:
            reason_codes.append("model_transport_error")
        if "docker" in reason:
            reason_codes.append("docker_preflight_timeout")
        results.append({
            "batch": _batch_label(path), "status": "stopped_environment_interruption",
            "contract_sha256": fingerprint, "excluded_from_effect_comparison": True,
            "planned_trials": counts["planned_trials"], "recorded_trials": len(rows),
            "known_total_tokens": sum(_number(r["known_total_tokens"], integer=True) for r in known_usage),
            "usage_unmeasured_trials": len(rows) - len(known_usage),
            "unknown_usage_calls": sum(_number(r["unknown_usage_calls"], integer=True) for r in known_counts),
            "usage_call_counts_unmeasured_trials": len(rows) - len(known_counts),
            "formatter_calls": len(called),
            "formatter_known_total_tokens": sum(_number(f["usage"]["total_tokens"], integer=True) for f in measured_formatter),
            "formatter_unknown_usage_calls": len(called) - len(measured_formatter),
            "formatter_status_unmeasured_trials": len(rows) - len(formatters),
            # Older workers put SDK exceptions in the private core report,
            # not in summary.errors. An absent diagnostic is unknown, not zero.
            "transport_failure_trials": (sum("OpenAIConnectionError" in r["errors"] for r in rows)
                                         if all(isinstance(r.get("errors"), list) for r in rows) else None),
            "transport_diagnostics_unmeasured_trials": sum(not isinstance(r.get("errors"), list) for r in rows),
            "longest_recorded_seconds": _number(marker.get("longest_recorded_seconds")),
            "reason_codes": reason_codes or ["unclassified_environment_interruption"],
        })
    return results


def build_public_report(batch, *, allow_incomplete=False, audit_path=None, audited_suite_path=None,
                        rescore_path=None, interrupted_batch_paths=None, data_audit_path=None):
    """Select only safe metadata/counts; reject stale or mixed batch reports."""
    batch = Path(batch)
    aggregate = read_json(batch / "heldout/report.json")
    contract_raw = (batch / "contract.json").read_bytes()
    contract = json.loads(contract_raw)
    fingerprint = hashlib.sha256(contract_raw).hexdigest()
    if aggregate.get("split") != "heldout" or aggregate.get("contract_sha256") != fingerprint:
        raise ValueError("Heldout aggregate has a different evaluation contract")
    suite_raw = (batch / "suite.json").read_bytes()
    suite = json.loads(suite_raw)
    cases = suite["heldout"]
    rows = [read_json(path) for path in sorted((batch / "heldout/trials").glob("*/summary.json"))]
    if any(row.get("contract_sha256") != fingerprint for row in rows):
        raise ValueError("Trial summaries contain mixed contracts")
    expected = summarize_trials(rows, planned_cases=cases, repetitions=aggregate["repetitions"])
    # All original accounting must agree with current summaries. Extra raw
    # fields such as answers or formatter responses are deliberately ignored.
    for key in ("complete", "planned_trials", "recorded_trials", "unique_cases", "unique_datasets", "paired", "confidence_intervals", "strict_all_repeats"):
        if aggregate.get(key) != expected[key]:
            raise ValueError("Heldout aggregate is stale or inconsistent: " + key)
    for arm in ARMS:
        for key, value in expected["arms"][arm].items():
            if aggregate["arms"][arm].get(key) != value:
                raise ValueError("Heldout arm aggregate is stale: " + key)
    if not aggregate["complete"] and not allow_incomplete:
        raise ValueError("Batch is incomplete; refuse to publish final results")
    safe_cases = []
    for case in sorted(cases, key=lambda c: c["id"]):
        metadata = _case_metadata(case)
        for arm in ARMS:
            selected = [r for r in rows if r["case"] == case["id"] and r["arm"] == arm]
            metadata[arm] = {
                "recorded": len(selected),
                **{metric: sum(r[metric] for r in selected) for metric in METRICS},
                "raw_answer_correct": sum(bool(r.get("raw_answer_correct")) for r in selected),
                "skill_reads": sum(len(r.get("skill_loads") or []) for r in selected),
                "failure_categories": {category: sum(category in failure_categories(r) for r in selected) for category in CATEGORIES},
            }
        safe_cases.append(metadata)
    model_id = contract.get("model", {}).get("model_id")
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", model_id):
        raise ValueError("Unexpected public model identifier")
    options = contract.get("options", {})
    data_audit, sensitivities = _data_audit_summary(data_audit_path, suite_raw, suite, rows, aggregate["repetitions"])
    return {
        "report_version": 1,
        "benchmark": "InfiAgent-DABench public validation compatible subset",
        "complete": aggregate["complete"],
        "batch": _batch_label(batch),
        "selection_policy": "First complete uninterrupted batch; incomplete environment-interrupted batches retained and excluded, never select the best score.",
        "contract_sha256": fingerprint,
        "dataset_revision": _revision(aggregate["dataset_revision"]),
        "scorer_revision": _revision(aggregate["scorer_revision"]),
        "implementation_sha256": _fingerprint(contract["implementation_sha256"]),
        "configuration": {"agent_model": model_id, "formatter_model": model_id,
                          **{key: _number(options.get(key)) for key in (
                              "temperature", "main_max_tokens", "analysis_max_tokens", "formatter_max_tokens",
                              "max_retries", "formatter_retries", "execution_timeout", "worker_timeout")}},
        "design": {**{key: aggregate[key] for key in ("planned_trials", "recorded_trials", "unique_cases", "unique_datasets", "repetitions")},
                   "dataset_count_unit": "distinct CSV files, not proven independent data families"},
        "dataset_family_boundary": _family_disclosure(suite),
        "data_audit": data_audit,
        "grouping_sensitivity": sensitivities,
        "difficulty_counts": dict(sorted(Counter(c["level"] for c in cases).items())),
        "arms": _safe_arms(aggregate, rows),
        "paired": _safe_paired(aggregate),
        "confidence_intervals": _safe_intervals(aggregate),
        "strict_all_repeats": {metric: {arm: {key: _number(aggregate["strict_all_repeats"][metric][arm][key])
            for key in ("passed_cases", "complete_cases", "planned_cases", "rate_planned")} for arm in ARMS} for metric in METRICS},
        "by_case": safe_cases,
        "statement_audit": _audit_summary(audit_path, audited_suite_path, suite_raw, suite),
        "representation_rescore": _rescore_summary(rescore_path, aggregate, batch, fingerprint),
        "interrupted_batches": _interrupted_summaries(interrupted_batch_paths, batch, contract),
        "failure_category_labels": CATEGORY_LABELS,
        "limits": [
            "公开验证集的预先冻结兼容子集，不是官方隐藏测试或完整排行榜成绩。",
            "同一CSV的重复运行相关；区间按CSV文件整簇重采样，是探索性区间，未排除不同CSV之间的同源相关性。",
            "不同CSV文件的数量不等于相同数量的独立数据家族；CSV级开发/留出划分不保证底层数据家族完全隔离，也不能排除预训练污染。",
            "开发题仅用于格式适配，没有调整Agent提示词；已知数据家族重叠仍限制效果结论的外推。",
            "ON比较目录、读取引导和实际正文的整体集成，不是只有正文变化的纯文本消融。",
            "盲格式整理器只接收题目、格式要求与最终回答，不接收CSV、产物、标准答案或工具。",
            "整理器沿用本轮固定模型，不是原论文GPT-3.5整理器的完全复现；原始分与整理后分并列。",
            "题目方法、舍入和解释可能有歧义；盲审发现的问题不改题、不豁免、不提高分数。",
            "失败分类是运行状态和评分字段的诊断标签，不是自动证明的根本原因。",
            "未全面覆盖专业报告质量、RAG、分类/回归建模和实际业务任务；不能外推任意分析正确性。",
            "技能读取次数代表实际加载，不证明遵循了指导或结果正确；未加载的ON运行仍计入全部结果。",
        ],
    }


def _percent(value):
    return "未知" if value is None else f"{100 * value:.2f}%"


def _points(value):
    return "未知" if value is None else f"{100 * value:+.2f}"


def _display(value):
    if value is None:
        return "未知"
    return f"{value:,}" if isinstance(value, int) else f"{value:.2f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(report):
    off, on = report["arms"]["off"], report["arms"]["on"]
    design = report["design"]
    data_audit = report["data_audit"]
    csv_description = f"{design['unique_datasets']}份不同CSV路径"
    if data_audit["available"]:
        csv_description += f"、{data_audit['heldout_unique_byte_hashes']}种唯一文件内容"
    lines = ["# Skills 外部基准配对评测", "",
             "## 结果概览", "",
             f"批次{'完整' if report['complete'] else '尚未完成（中间结果）'}：{design['unique_cases']}道题、{csv_description}，每题每组{design['repetitions']}次；计划{design['planned_trials']}次，已记录{design['recorded_trials']}次。",
             "", "| 指标 | Skills OFF | Skills ON |", "|---|---:|---:|"]
    for metric, label in (("answer_correct", "官方参考答案匹配（盲整理后）"), ("delivery_completed", "产品交付完成"), ("end_to_end", "参考答案匹配且交付完成")):
        lines.append(f"| {label} | {off[metric]}/{off['planned']}（{_percent(off['rates_planned'][metric])}） | {on[metric]}/{on['planned']}（{_percent(on['rates_planned'][metric])}） |")
    ci = report["confidence_intervals"]["end_to_end"]
    lines += ["", f"端到端配对差值 ON−OFF：{_points(ci['on_minus_off'])}个百分点；CSV整簇 bootstrap 95%探索性区间 [{_points(ci['lower'])}, {_points(ci['upper'])}] 个百分点。"]
    lines += ["", f"主比较采用 `{report['batch']}`。按预先声明的规则采用首个完整且未发生环境中断的批次，不按得分选择最佳批次；所有中断批次保留并单列成本。主表保留冻结评分流程的原成绩，表示修正另列。"]
    if not report["complete"]:
        lines += ["", "**本批次尚未完成，不作最终效果结论。**"]
    elif ci["lower"] is not None and ci["lower"] <= 0 <= ci["upper"]:
        lines += ["", "区间覆盖零差值；当前样本不足以确认稳定收益。结论仅针对该固定子集，不能据此断定Skills普遍有效或无效。"]
    else:
        lines += ["", "该差值只描述当前固定子集及配置，不能外推为任意任务的普遍改善或退步。"]
    lines += ["", "## 方法与评分", "",
              "使用官方公开CSV、原始英文问题、constraints及format；按不同CSV文件划分开发与留出集，正式题目在运行前冻结。难度分布："
              + "、".join(f"{level} {report['difficulty_counts'].get(level, 0)}题" for level in ("easy", "medium", "hard")) + "。",
              "", "两组使用相同代码、模型、输入、沙箱与预算；每次创建新进程和会话，配对顺序交错。ON整体包含技能目录、读取引导与按需正文，不能把差值仅归因于正文。",
              "", f"Agent和盲整理器均为 `{_cell(report['configuration']['agent_model'])}`，配置固定。整理器没有数据、工具和标准答案，只可转写原回答已陈述的答案；未陈述内容不得补算。数值来源过滤不能完全证明字段映射和语义转写正确。",
              "", "官方比较器使用字符串严格相等，或数值绝对差小于 `1e-6`。原评分器会跳过空回答；本轮为全部计划试次补齐失败记录，空答、超时、崩溃与未完成均保留在分母中。",
              "", "ABQ为整题全部子答案正确率；PASQ为逐题子答案正确比例的均值；UASQ为全部子答案的总体正确率。三者按官方函数计算，保留原始回答分与盲整理后分。", "",
              "| 官方指标 | OFF 原始 | ON 原始 | OFF 盲整理 | ON 盲整理 |", "|---|---:|---:|---:|---:|"]
    # Append the family limitation outside the metric table so the prose cannot
    # accidentally be rendered as malformed table rows.
    for metric in ("ABQ", "PASQ", "UASQ"):
        lines.append(f"| {metric} | {_percent(off['official_raw'][metric])} | {_percent(on['official_raw'][metric])} | {_percent(off['official_blind_reformat'][metric])} | {_percent(on['official_blind_reformat'][metric])} |")
    lines += ["", "开发题仅用于回答格式适配，没有调整Agent提示词。CSV级划分不能保证底层数据家族完全隔离；以下区间也不能排除同源相关性或预训练污染。"]
    for example in report["dataset_family_boundary"]["known_overlap_examples"]:
        names = "、".join(f"`{name}`" for name in example["heldout_csv"])
        lines += ["", f"已核对原CSV表头：开发集 `{example['development_csv']}` 与留出集 {names} 属于Titanic同源或派生数据。因此这些文件不能视作完全独立的数据家族。题目、冻结计划和全部成绩均保留，没有为消除此局限补跑新题。"]
    if data_audit["available"]:
        for group in data_audit["byte_identical_groups"]:
            lines += ["", "字节审计还确认 " + "、".join(f"`{name}`" for name in group["csv_files"]) + " 的SHA-256完全相同；不同文件路径并不代表不同内容。重复内容对应的题目仍保留在全部原始成绩中。"]
    lines += ["", f"仅因盲整理从原始未通过变为通过：OFF {off['format_rescued']}次，ON {on['format_rescued']}次。整理不作为Agent内部修复，也不修改原始报告。",
              "", "## 配对与重复稳定性", "", "| 维度 | 双方通过 | 仅OFF通过 | 仅ON通过 | 双方失败 | 缺失配对 |", "|---|---:|---:|---:|---:|---:|"]
    for metric, label in (("answer_correct", "答案"), ("delivery_completed", "交付"), ("end_to_end", "端到端")):
        paired = report["paired"][metric]
        lines.append(f"| {label} | " + " | ".join(str(paired[k]) for k in ("both_pass", "off_only", "on_only", "both_fail", "incomplete")) + " |")
    lines += ["", "| 维度 | ON−OFF（百分点） | 95%探索性区间（百分点） | CSV簇 | 完整配对／排除配对 |", "|---|---:|---|---:|---|"]
    for metric in METRICS:
        item = report["confidence_intervals"][metric]
        lines.append(f"| {metric} | {_points(item['on_minus_off'])} | [{_points(item['lower'])}, {_points(item['upper'])}] | {item['dataset_clusters']} | {item['complete_pairs']}／{item['excluded_incomplete_pairs']} |")
    lines += ["", "整簇抽样将同一CSV的全部重复一起抽取；区间仅使用完整OFF/ON配对，缺失数量如上。重复次数不能当成同等数量的独立问题；不同CSV之间也可能共享数据家族，此区间未消除该相关性。", "",
              "| 全部重复均通过的题目 | OFF | ON |", "|---|---:|---:|"]
    for metric in METRICS:
        item = report["strict_all_repeats"][metric]
        lines.append(f"| {metric} | {item['off']['passed_cases']}/{item['off']['planned_cases']} | {item['on']['passed_cases']}/{item['on']['planned_cases']} |")
    lines += ["", "## 效率与实际技能读取", "", "| 指标 | OFF | ON |", "|---|---:|---:|"]
    for key, label in (("known_total_tokens", "Agent已知token"), ("unknown_usage_calls", "Agent用量未知请求"),
                       ("usage_unmeasured_trials", "Agent未取得计量的试次"), ("formatter_calls", "额外盲整理调用"),
                       ("formatter_known_total_tokens", "盲整理已知token"), ("formatter_unknown_usage_calls", "盲整理用量未知请求"),
                       ("formatter_status_unmeasured_trials", "盲整理状态未取得的试次"),
                       ("median_seconds", "Agent耗时中位数（秒）"), ("code_attempts", "代码执行尝试"),
                       ("no_internal_repair_attained", "无需内部修正即端到端完成"),
                       ("skill_reads", "实际技能读取"), ("trials_reading_skills", "发生读取的试次")):
        lines.append(f"| {label} | {_display(off[key])} | {_display(on[key])} |")
    lines.append(f"| 经内部反馈或修正后端到端完成 | {off['end_to_end'] - off['no_internal_repair_attained']} | {on['end_to_end'] - on['no_internal_repair_attained']} |")
    lines += ["", "Token仅汇总接口实际返回的用量；未知不填零、不估算价格。耗时是本机实际运行记录，不是隔离的性能基准。", "",
              "| 技能 | OFF读取 | ON读取 |", "|---|---:|---:|"]
    for skill in SKILLS:
        lines.append(f"| {skill} | {off['skill_reads_by_id'][skill]} | {on['skill_reads_by_id'][skill]} |")
    lines += ["", "## 失败诊断", "", "以下标签来自状态与评分字段，可重叠，数量不能相加作为失败总数。‘答案不符’不直接等同于计算代码错误；题意、方法选择、标准答案和语义转写都可能影响得分。", "",
              "| 诊断标签 | OFF | ON |", "|---|---:|---:|"]
    for category in CATEGORIES:
        lines.append(f"| {CATEGORY_LABELS[category]} | {off['failure_categories'][category]} | {on['failure_categories'][category]} |")
    lines += ["", "## 逐题结果", "", "每项为通过次数／已记录次数。原始CSV、题目答案、聊天、代码、运行路径和日志不随此报告发布。", "",
              "| 题目 | CSV | 难度／概念 | OFF答案／交付／端到端 | ON答案／交付／端到端 | OFF诊断 | ON诊断 |", "|---|---|---|---|---|---|---|"]
    for case in report["by_case"]:
        values, categories = [], []
        for arm in ARMS:
            item = case[arm]
            values.append("／".join(f"{item[metric]}/{item['recorded']}" for metric in METRICS))
            categories.append("；".join(f"{CATEGORY_LABELS[key]} {value}" for key, value in item["failure_categories"].items() if value) or "—")
        lines.append("| " + " | ".join(_cell(v) for v in (
            case["id"], case["dataset"], case["level"] + "；" + ", ".join(case["concepts"]), *values, *categories)) + " |")
    sensitivity = report["grouping_sensitivity"]
    if sensitivity["available"]:
        lines += ["", "## 分组相关性的敏感性分析", "",
                  "以下仅对前文主表原评分调整重采样分组，不改变任何题目、通过次数、主表原CSV簇区间或模型执行。分组规则只依据CSV字节及表头审计，未使用标准答案或模型成绩。两个附加区间仍为探索性估计，已识别家族并非穷尽审计。", "",
                  "| 重采样分组 | 簇数 | 端到端ON−OFF（百分点） | 95%探索性区间（百分点） |", "|---|---:|---:|---|"]
        for name, label in (("identical_content", "完全相同字节合簇"), ("known_families", "相同字节＋已识别家族合簇")):
            item = sensitivity[name]
            interval = item["confidence_intervals"]["end_to_end"]
            lines.append(f"| {label} | {item['clusters']} | {_points(interval['on_minus_off'])} | [{_points(interval['lower'])}, {_points(interval['upper'])}] |")
        lines += ["", "相同内容与已知家族合簇能够体现部分相关性，但不能证明其他CSV彼此独立，也不能排除训练污染；不将这些附表当作显著性结论。表示修正版v3在下节单列，未以敏感性区间替换其原CSV簇区间。"]
    rescore = report["representation_rescore"]
    if rescore["available"]:
        lines += ["", "## 表示修正版 v3：全样本无模型重评分", "",
                  "开发题暴露了两项回答传输问题：原回答复述公共format占位符时被当成最终值保留，以及Unicode减号不被数值解析识别。v3对所有计划样本使用同一纯表示规则，只移除与原题format完全相同的占位值、统一减号，并重放已有整理器回答。真实原答案（包括错误数值）仍保留；不新增模型调用，不重新计算或选择性重跑任务。",
                  "", "这是开发题导出的事后适配器修正，不能解释为Agent分析能力提升；原始评分、summary及前文主表均保留。", "",
                  "| 官方指标 | OFF 原盲整理 | ON 原盲整理 | OFF 表示v3 | ON 表示v3 |", "|---|---:|---:|---:|---:|"]
        for metric in ("ABQ", "PASQ", "UASQ"):
            official = rescore["official"]
            lines.append(f"| {metric} | {_percent(official['off']['v2_original'][metric])} | {_percent(official['on']['v2_original'][metric])} | {_percent(official['off']['v3_rescored'][metric])} | {_percent(official['on']['v3_rescored'][metric])} |")
        lines += ["", "| v3维度 | OFF | ON |", "|---|---:|---:|"]
        for metric in METRICS:
            arms = rescore["arms"]
            lines.append(f"| {metric} | {arms['off'][metric]}/{arms['off']['planned']} | {arms['on'][metric]}/{arms['on']['planned']} |")
        revised_ci = rescore["confidence_intervals"]["end_to_end"]
        lines += ["", f"v3端到端差值 ON−OFF：{_points(revised_ci['on_minus_off'])}个百分点，95%CSV簇探索性区间 [{_points(revised_ci['lower'])}, {_points(revised_ci['upper'])}] 个百分点；同源数据家族相关性仍未排除。",
                  "", f"变更试次：通过增加{rescore['upgraded_trials']}次，通过减少{rescore['downgraded_trials']}次；新增模型调用0次、新增token为0。原Agent与整理器用量只计一次，不能把原分和v3成本相加。",
                  "", f"重评分合同：`{rescore['rescore_contract_sha256']}`；重评分脚本：`{rescore['script_sha256']}`。"]
    if report["interrupted_batches"]:
        lines += ["", "## 环境中断批次：保留记录，不混入效果比较", "",
                  "以下不完整批次整批保留，不按题目成绩筛除或覆盖失败。外部主机异常与传输故障影响了运行；恢复环境后使用相同冻结代码、题目、模型和预算创建完整新批次。此表只披露运行与成本，不汇入主表成功率、配对或耗时估计。", "",
                  "| 批次 | 已记录／计划 | Agent已知token | Agent用量未知调用 | 未取得调用计量的试次 | 整理器已知token | 整理器未知调用 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for item in report["interrupted_batches"]:
            lines.append(f"| {item['batch']} | {item['recorded_trials']}/{item['planned_trials']} | {_display(item['known_total_tokens'])} | {item['unknown_usage_calls']} | {item['usage_call_counts_unmeasured_trials']} | {_display(item['formatter_known_total_tokens'])} | {item['formatter_unknown_usage_calls']} |")
            lines += ["", f"`{item['batch']}`：摘要可计量的模型连接错误试次{_display(item['transport_failure_trials'])}（{item['transport_diagnostics_unmeasured_trials']}份摘要缺少该诊断字段）；最长已记录耗时{_display(item['longest_recorded_seconds'])}秒。原因类别：" + "、".join(item["reason_codes"]) + "。",
                      f"其Agent token未计量试次{item['usage_unmeasured_trials']}，整理器状态未取得的试次{item['formatter_status_unmeasured_trials']}；这些未知量不能当作已知零成本。原合同指纹：`{item['contract_sha256']}`。", ""]
    lines += ["", "## 边界与可追溯性", ""]
    audit = report["statement_audit"]
    if audit["available"]:
        lines += ["题目盲审风险计数：" + "、".join(f"{level} {audit['risk_counts'][level]}题" for level in ("high", "medium", "low", "none"))
                  + "。高风险题：" + "、".join(audit["high_risk_case_ids"]) + "。盲审未查看标准答案或模型结果；保留所有题目和原评分，不作豁免。", ""]
        if audit["audited_suite_sha256"] != audit["current_suite_sha256"]:
            lines += ["盲审后评测元数据曾变化；已核对原审计文件的suite指纹，以及两版heldout原文/约束/格式完全相同。两个原始指纹均保留，未覆盖原审计记录。", ""]
    else:
        lines += ["此导出未附独立题目盲审汇总；不据此假定题目或标准答案完全无歧义。", ""]
    lines.extend("- " + item for item in report["limits"])
    lines += ["", f"- 数据版本：`{report['dataset_revision']}`。", f"- 评分器版本：`{report['scorer_revision']}`。",
              f"- 实现指纹：`{report['implementation_sha256']}`。", f"- 本轮合同指纹：`{report['contract_sha256']}`。", "",
              "来源：[InfiAgent-DABench](https://infiagent.github.io/)、[官方公开数据](https://huggingface.co/datasets/infiagent/DABench)、[官方评分代码](https://github.com/InfiAgent/InfiAgent/blob/"
              + report["scorer_revision"] + "/examples/DA-Agent/eval_closed_form.py)。", ""]
    return "\n".join(lines)


def publish(batch, publish_dir=ROOT / "evaluation/results", *, allow_incomplete=False, audit_path=None, audited_suite_path=None,
            rescore_path=None, interrupted_batch_paths=None, data_audit_path=None):
    report = build_public_report(batch, allow_incomplete=allow_incomplete, audit_path=audit_path, audited_suite_path=audited_suite_path,
                                 rescore_path=rescore_path, interrupted_batch_paths=interrupted_batch_paths, data_audit_path=data_audit_path)
    destination = Path(publish_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "skills-dabench.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "skills-dabench.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, default=ROOT / "evaluation/results")
    parser.add_argument("--allow-incomplete", action="store_true", help="Publish explicitly marked intermediate results")
    parser.add_argument("--audit", type=Path, help="Statement-only audit summary; raw audit text is not exported")
    parser.add_argument("--audited-suite", type=Path, help="Original audited suite when only evaluation metadata later changed")
    parser.add_argument("--rescore", type=Path, help="Independent no-model representation rescore report.json; original scores remain")
    parser.add_argument("--interrupted-batch", type=Path, action="append", default=[], help="Preserved incomplete environment-interrupted batch; repeatable")
    parser.add_argument("--data-audit", type=Path, help="Outcome-blind CSV-byte/family audit; adds separate exploratory grouping sensitivity")
    args = parser.parse_args()
    report = publish(args.batch, args.publish_dir, allow_incomplete=args.allow_incomplete,
                     audit_path=args.audit, audited_suite_path=args.audited_suite,
                     rescore_path=args.rescore, interrupted_batch_paths=args.interrupted_batch, data_audit_path=args.data_audit)
    print(json.dumps({key: report[key] for key in ("complete", "contract_sha256", "design")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
