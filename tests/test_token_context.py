"""Model context budgets preserve real output, provenance, and tool evidence.

Executor fixtures verify orchestration only; they do not establish numerical
correctness of generated analysis code.
"""

import base64
import copy
import json

import pytest
from langchain_core.messages import AIMessage

from labweaver.agent import create_session
from labweaver.runtime.artifacts import (
    artifact_view,
    model_artifact_receipts,
    model_artifact_view,
    validate_payload,
)
from labweaver.modeling import PLANNING_POLICY
from labweaver.runtime.harness import validate_evidence
from labweaver.skills import SkillLibrary
from test_agent import answer, call, csv_path, inject_executor, paired, plan
from test_complex_analysis import (
    ReceiptAfterRepairModel, ScriptedAnalysisModel, code_call, figure_payload, no_network,
)


SOURCE = {"name": "selected.csv", "sha256": "b" * 64}
HOST_FIELDS = {"files", "execution_id", "image_id", "session_id", "task_id"}


def _registered(item):
    metadata = item["metadata"]
    assets = validate_payload(
        [item],
        deliverables=[{
            "id": metadata["deliverable_id"],
            "kind": metadata["kind"],
            "description": "Requested complete result",
        }],
        source=SOURCE,
        execution_id="execution-1",
        image_id="sha256:" + "c" * 64,
    )
    asset = next(iter(assets.values()))
    asset["metadata"].update(session_id="session-1", task_id="task-1")
    return asset


def _table(rows=280, width=1):
    raw = "group,value\n" + "".join(
        f"{i:03},{'x' * width}\n" for i in range(rows)
    )
    return _registered({
        "metadata": {
            "kind": "table", "deliverable_id": "table", "label": "Complete table",
            "row_count": rows, "columns": ["group", "value"],
        },
        "files": {"table.csv": base64.b64encode(raw.encode()).decode()},
    })


def _metric(value, deliverable_id="metric"):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    return _registered({
        "metadata": {
            "kind": "metric", "deliverable_id": deliverable_id,
            "label": "Requested metric",
        },
        "files": {"metric.json": base64.b64encode(raw).decode()},
    })


def test_table_receipt_keeps_full_count_and_cursor_without_host_metadata():
    asset = _table()
    before = copy.deepcopy(asset)
    preview = model_artifact_receipts([asset])[0]
    assert preview["source"] == SOURCE
    assert preview["id"] == asset["metadata"]["id"]
    assert preview["deliverable_id"] == "table"
    assert preview["row_count"] == preview["available_row_count"] == 280
    assert len(preview["rows"]) == 5
    assert preview["preview_only"] is True
    assert preview["page"] == {"offset": 0, "row_count": 5, "next_offset": 5}
    assert not HOST_FIELDS.intersection(preview)

    later = model_artifact_view(asset, offset=270, limit=20)
    assert len(later["rows"]) == 10
    assert later["rows"][-1]["group"] == "279"
    assert later["page"]["next_offset"] is None
    assert asset == before
    assert len(asset["files"]["table.csv"].splitlines()) == 281


@pytest.mark.parametrize("kind", ["table", "metric", "figure"])
def test_model_projection_preserves_values_and_leaves_host_evidence_unchanged(kind):
    asset = (_table(rows=2) if kind == "table" else
             _metric({"total": 6, "labels": ["A", "B"]}) if kind == "metric" else
             _registered(figure_payload("chart")))
    before = copy.deepcopy(asset)
    host_view = artifact_view(asset)
    projected = model_artifact_view(asset)
    assert not HOST_FIELDS.intersection(projected)
    assert projected["source"] == SOURCE
    assert projected["id"] == asset["metadata"]["id"]
    assert projected["kind"] == kind
    if kind == "metric":
        assert projected["value"] == {"total": 6, "labels": ["A", "B"]}
    else:
        assert projected["rows"] == host_view["rows"]
        assert projected["columns"] == host_view["columns"]
    assert asset == before
    assert artifact_view(asset) == host_view
    assert HOST_FIELDS.issubset(host_view)


def test_large_default_metric_receipt_points_to_full_explicit_read():
    value = {"parameters": "measurement-" * 900, "expected_count": 280}
    asset = _metric(value)
    before = copy.deepcopy(asset)
    receipt = model_artifact_receipts([asset])[0]
    assert receipt["preview_only"] is True
    assert "value" not in receipt
    assert receipt["source"] == SOURCE
    assert receipt["id"] == asset["metadata"]["id"]
    assert "ID" in receipt["notice"]
    explicit = model_artifact_view(asset)
    assert explicit["value"] == value
    assert not explicit.get("preview_only", False)
    assert asset == before


def test_wide_table_receipt_does_not_turn_preview_into_complete_result():
    asset = _table(rows=10, width=1500)
    receipt = model_artifact_receipts([asset])[0]
    assert receipt["preview_only"] is True
    assert receipt["row_count"] == 10
    assert receipt["source"] == SOURCE
    assert "rows" not in receipt
    assert "ID" in receipt["notice"]
    explicit = model_artifact_view(asset, offset=5, limit=5)
    assert explicit["available_row_count"] == 10
    assert len(explicit["rows"]) == 5
    assert explicit["rows"][-1] == {"group": "009", "value": "x" * 1500}
    assert explicit["page"]["next_offset"] is None


def test_combined_receipt_budget_keeps_references_to_every_registered_result():
    assets = [_metric({"values": "x" * 2500}, f"metric_{i}") for i in range(4)]
    receipts = model_artifact_receipts(assets)
    assert len(receipts) == len(assets)
    assert "value" in receipts[0]
    omitted = [receipt for receipt in receipts if receipt.get("preview_only")]
    assert omitted
    by_id = {asset["metadata"]["id"]: asset for asset in assets}
    assert {receipt["id"] for receipt in receipts} == set(by_id)
    for receipt in omitted:
        assert "value" not in receipt
        assert receipt["source"] == SOURCE
        assert "ID" in receipt["notice"]
        assert model_artifact_view(by_id[receipt["id"]])["value"] == {"values": "x" * 2500}
    assert len(json.dumps(receipts, ensure_ascii=False).encode()) < 8192


def test_metric_beyond_explicit_preview_limit_retains_identity_and_omission_notice():
    asset = _metric({"values": "x" * (70 * 1024)})
    before = copy.deepcopy(asset)
    for view in (model_artifact_view(asset), model_artifact_receipts([asset])[0]):
        assert view["preview_only"] is True
        assert "value" not in view
        assert view["id"] == asset["metadata"]["id"]
        assert view["deliverable_id"] == "metric"
        assert view["source"] == SOURCE
        assert view.get("view_error") or view.get("notice")
    assert asset == before
    assert json.loads(asset["files"]["metric.json"])["values"] == "x" * (70 * 1024)


def _delegated_previous_code(csv_path, monkeypatch, previous_report):
    """Inspect real delegation transport using explicit prior-report fixtures."""
    main = ScriptedAnalysisModel(responses=[
        call(), call("set_task_plan", "plan", {"deliverables": plan()}),
        call("delegate_analysis", "delegate", {
            "instruction": "Recompute the confirmed complete table", "deliverable_ids": ["table"],
        }),
        call("ask_user", "scope", {"question": "请确认后续计算范围"}),
    ])
    child = ScriptedAnalysisModel(responses=[AIMessage(content=json.dumps({
        "status": "needs_clarification", "question": "请确认后续计算范围",
    }))])
    session = create_session(csv_path, main, analysis_model=child)
    before = copy.deepcopy(previous_report)
    session._last = previous_report
    monkeypatch.setattr(session.executor, "execute", lambda *a, **k: pytest.fail("Context inspection must not execute"))

    report = session.invoke("按已确认口径重新计算并导出完整表")

    assert report["status"] == "awaiting_input", report.get("error")
    assert report["analysis_runs"][0]["status"] == "needs_clarification"
    assert not report["code_executions"]
    assert previous_report == before
    assert validate_evidence(report["trace"], report["execution_ledger"], pending=True) is None
    packet = json.loads(next(message.content for message in child.received[0]
                             if message.type == "human"))
    return packet["context"]["previous_code"]


@pytest.mark.parametrize("previous_report", [
    None,
    {"code_executions": [{"status": "error", "code": "raise ValueError('failed only')"}]},
])
def test_previous_code_without_success_sends_no_script(csv_path, monkeypatch, previous_report):
    assert _delegated_previous_code(csv_path, monkeypatch, previous_report) == []


def test_previous_code_sends_only_latest_successful_complete_script(csv_path, monkeypatch):
    latest = "from helper import load_dataset\ndf = load_dataset()\nassert len(df) >= 0\n"
    previous_report = {"code_executions": [
        {"status": "completed", "code": "print('older success')"},
        {"status": "completed", "code": "# older oversized script\n#" + "x" * 32768},
        {"status": "error", "code": "raise ValueError('before latest success')"},
        {"status": "completed", "code": latest},
        {"status": "error", "code": "raise ValueError('after latest success')"},
    ]}
    assert _delegated_previous_code(csv_path, monkeypatch, previous_report) == [latest]


@pytest.mark.parametrize("character,byte_count", [
    ("x", 32768), ("x", 32769), ("中", 32768), ("中", 32769),
])
def test_previous_code_budget_uses_utf8_bytes_and_never_truncates_or_falls_back(
    csv_path, monkeypatch, character, byte_count,
):
    prefix, suffix = "# latest_complete_script\n#", "\nprint('complete latest script')\n"
    available = byte_count - len((prefix + suffix).encode("utf-8"))
    count, padding = divmod(available, len(character.encode("utf-8")))
    latest = prefix + character * count + "x" * padding + suffix
    assert len(latest.encode("utf-8")) == byte_count
    if character == "中":
        assert len(latest) < 32768
    older = "print('older valid script must not substitute for latest')"
    previous_report = {"code_executions": [
        {"status": "completed", "code": older},
        {"status": "completed", "code": latest},
        {"status": "error", "code": "raise ValueError('newer failed attempt')"},
    ]}

    previous = _delegated_previous_code(csv_path, monkeypatch, previous_report)

    if byte_count <= 32768:
        assert previous == [latest]
    else:
        assert isinstance(previous, dict) and set(previous) == {"omitted"}
        assert "context budget" in previous["omitted"]
        assert "regenerate" in previous["omitted"]
        assert "latest_complete_script" not in json.dumps(previous)
        assert older not in json.dumps(previous)


def test_accepted_plan_schema_is_removed_and_new_task_can_plan_again(csv_path, monkeypatch):
    deliveries = [{"id": "overview", "kind": "explanation", "description": "Actual fields"}]
    model = ScriptedAnalysisModel(responses=[
        call(), call("set_task_plan", "plan-1", {"deliverables": deliveries}), answer(),
        call("set_task_plan", "plan-2", {"deliverables": deliveries}), answer(),
    ])
    session = create_session(csv_path, model)
    monkeypatch.setattr(session.executor, "execute", lambda *a, **k: pytest.fail("Overview must not execute"))
    first = session.invoke("仅概览实际字段，不需要计算或绘图")
    assert first["status"] == "completed", first.get("error")
    assert "set_task_plan" in first["tool_exposure"][1]
    assert "set_task_plan" not in first["tool_exposure"][2]
    assert "set_task_plan" not in model.bound_tool_sets[2]
    paired(first)

    second = session.invoke("再次概览实际字段，不需要计算或绘图")
    assert second["status"] == "completed", second.get("error")
    assert second["task_id"] != first["task_id"]
    assert "set_task_plan" in second["tool_exposure"][0]
    assert "set_task_plan" not in second["tool_exposure"][1]
    assert "set_task_plan" in model.bound_tool_sets[3]
    assert "set_task_plan" not in model.bound_tool_sets[4]
    paired(second)


@pytest.mark.parametrize("task", [
    "概述数据规模和字段",
    "训练一个回归模型",
    "Train a model to predict score",
    "预测score",
])
def test_planning_keeps_modeling_defaults_for_all_task_wordings(csv_path, task):
    model = ScriptedAnalysisModel(responses=[
        call(), call("ask_user", "scope", {"question": "需要采用哪些字段？"}),
    ])
    session = create_session(csv_path, model)
    report = session.invoke(task)
    assert report["status"] == "awaiting_input", report.get("error")
    initial = "\n".join(str(message.content) for message in model.received[0])
    assert PLANNING_POLICY in initial
    assert model.bound_tool_sets[0] == ["profile_csv"]


@pytest.mark.parametrize("modeling", [False, True])
def test_frozen_plan_keeps_modeling_policy_only_when_modeling_is_recorded(csv_path, modeling):
    task = "Train a model to predict score" if modeling else "统计各组分数"
    deliveries = ([
        {"id": ident, "kind": kind, "description": ident}
        for ident, kind in [("pred", "table"), ("meta", "metric"), ("metrics", "metric")]
    ] if modeling else plan())
    spec = {"goal": task}
    if modeling:
        spec["modeling"] = {
            "task_type": "regression", "target": "c2", "features": ["c1"],
            "prediction_delivery": "pred", "metadata_delivery": "meta",
            "metrics_delivery": "metrics",
        }
    model = ScriptedAnalysisModel(responses=[
        call(), call("set_task_plan", "plan", {"deliverables": deliveries, "task_spec": spec}),
        call("ask_user", "scope", {"question": "还需补充哪些范围？"}),
    ])
    session = create_session(csv_path, model)
    report = session.invoke(task)
    assert report["status"] == "awaiting_input", report.get("error")
    assert session.deliverables == deliveries
    for messages in model.received[:2]:
        assert PLANNING_POLICY in "\n".join(str(message.content) for message in messages)
    frozen_context = "\n".join(str(message.content) for message in model.received[2])
    assert (PLANNING_POLICY in frozen_context) is modeling
    assert "set_task_plan" not in model.bound_tool_sets[2]


def test_full_catalog_is_available_to_main_and_child_without_eager_skill_read(csv_path, monkeypatch):
    main = ScriptedAnalysisModel(responses=[
        call(), call("set_task_plan", "plan", {"deliverables": plan()}),
        call("delegate_analysis", "delegate", {
            "instruction": "Compute all groups", "deliverable_ids": ["table"],
        }), answer("calculation"),
    ])
    child = ReceiptAfterRepairModel(responses=[
        call("read_skill", "child-skill", {"skill_id": "statistical-analysis"}),
        code_call(),
    ])
    session = create_session(csv_path, main, analysis_model=child, skills_enabled=True)
    executed = inject_executor(session, monkeypatch)
    report = session.invoke("统计各组并导出完整结果表，不绘图")
    assert report["status"] == "completed", report.get("error")
    assert len(executed) == 1
    main_initial = "\n".join(str(message.content) for message in main.received[0])
    child_initial = "\n".join(str(message.content) for message in child.received[0])
    library = SkillLibrary()
    assert all(item["id"] in main_initial for item in library.catalog())
    assert all(item["id"] in child_initial for item in library.catalog())
    assert all(library.read(item["id"])["content"] not in main_initial
               for item in library.catalog())
    assert all(library.read(item["id"])["content"] not in child_initial
               for item in library.catalog())
    assert report["tool_counts"]["read_skill"] == 0
    assert [(load["role"], load["skill_id"]) for load in report["skills"]["loads"]] == [
        ("analysis", "statistical-analysis"),
    ]
    paired(report)
