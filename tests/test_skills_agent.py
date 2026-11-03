"""Reviewed skills through real Agent graphs; fixture execution is not numerical proof."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from labweaver.agent import create_session
from labweaver.runtime.harness import ExecutionHarness, validate_evidence
from labweaver.runtime.skills import read_skill_tool
from labweaver.skills import SkillLibrary
from test_agent import (
    ScriptedModel, answer, call, csv_path, inject_executor, paired, plan,
)
from test_complex_analysis import (
    ReceiptAfterRepairModel, ScriptedAnalysisModel, code_call, no_network,
)


def skill_call(ident="skill", skill_id="data-exploration"):
    return call("read_skill", ident, {"skill_id": skill_id})


def coordinator():
    return [
        call(),
        call("set_task_plan", "plan", {"deliverables": plan()}),
        call("delegate_analysis", "delegate", {
            "instruction": "按真实字段统计各组", "deliverable_ids": ["table"],
        }),
        answer("calculation"),
    ]


def test_skill_guidance_cannot_replace_user_material_retrieval(csv_path, tmp_path):
    material=tmp_path/'rules.md'
    material.write_text('本任务只采用明确说明的字段定义。',encoding='utf-8')
    model=ScriptedModel(responses=[call(),skill_call(skill_id='analysis-validation'),
                                  answer(text='已按资料解释字段 [D1-C1]。')])
    report=create_session(csv_path,model,material_paths=[material],skills_enabled=True).invoke('基于资料解释字段')
    assert report['status']=='error'
    assert report['retrieval_status']=='not_used'
    assert report['skills']['loads'] and not report['retrieved_chunks']


def test_catalog_is_progressive_and_actual_read_is_paired(csv_path, monkeypatch):
    library = SkillLibrary()
    expected = library.read("data-exploration")
    model = ScriptedAnalysisModel(responses=[call(), skill_call(), answer()])
    session = create_session(csv_path, model, skills_enabled=True)
    monkeypatch.setattr(session.executor, "execute", lambda *a, **k: pytest.fail("Overview must not execute"))
    before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    events = []
    report = session.invoke("概述实际字段", on_event=events.append)
    assert report["status"] == "completed", report.get("error")
    initial = "\n".join(str(m.content) for m in model.received[0])
    assert all(item["id"] in initial for item in library.catalog())
    assert expected["content"] not in initial
    tool_messages = [m for history in model.received for m in history
                     if isinstance(m, ToolMessage) and m.name == "read_skill"]
    assert tool_messages and tool_messages[-1].tool_call_id == "skill"
    assert json.loads(tool_messages[-1].content)["content"] == expected["content"]
    assert report["skills"]["loads"] == [{
        "role": "main", "tool_call_id": "skill", "skill_id": "data-exploration",
        "sha256": expected["sha256"],
    }]
    assert len([e for e in events if e["type"] == "skill_loaded"]) == 1
    assert report["tool_exposure"][0] == ["profile_csv"]
    assert "read_skill" in report["tool_exposure"][1]
    assert not report["code_executions"]
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == before
    paired(report)


@pytest.mark.parametrize("enabled,before_profile", [(False, False), (True, True)])
def test_skill_is_not_exposed_when_disabled_or_before_profile(csv_path, enabled, before_profile):
    responses = ([] if before_profile else [call()]) + [skill_call()]
    model = ScriptedModel(responses=responses)
    report = create_session(csv_path, model, skills_enabled=enabled).invoke("概述实际字段")
    assert report["status"] == "error" and report["error"]["code"] == "tool_not_allowed"
    assert report["skills"]["loads"] == []
    if not enabled:
        assert report["skills"]["catalog"] == []
        assert all("read_skill" not in names for names in report["tool_exposure"])


@pytest.mark.parametrize("skill_id", ["../../.env", "C:/Users/private.txt", "https://example.com/SKILL.md", "missing-skill"])
def test_model_cannot_use_skill_reader_for_paths_or_urls(csv_path, skill_id):
    model = ScriptedModel(responses=[call(), skill_call(skill_id=skill_id), answer()])
    report = create_session(csv_path, model, skills_enabled=True).invoke("概述字段")
    result = next(e["result"] for e in report["execution_ledger"] if e["name"] == "read_skill")
    assert result == {"status": "error", "error": {"code": "skill_not_available"}}
    assert report["skills"]["loads"] == []
    assert not report["code_executions"]
    paired(report)


def test_reading_guidance_cannot_complete_a_computation(csv_path, monkeypatch):
    model = ScriptedModel(responses=[call(), skill_call(),
        call("set_task_plan", "plan", {"deliverables": plan()}),
        answer("calculation"), answer("calculation")])
    session = create_session(csv_path, model, skills_enabled=True)
    monkeypatch.setattr(session.executor, "execute", lambda *a, **k: pytest.fail("No execution was requested"))
    report = session.invoke("统计各组并导出完整结果表")
    assert report["status"] == "error" and report["error"]["code"] == "missing_deliverables"
    assert len(report["skills"]["loads"]) == 1
    assert not report["code_executions"] and report["missing_deliverables"] == ["table"]


def test_code_agent_reads_own_skill_and_keeps_scoped_tools(csv_path, monkeypatch):
    main = ScriptedModel(responses=coordinator())
    child = ReceiptAfterRepairModel(responses=[
        skill_call("child-skill", "statistical-analysis"), code_call(),
    ])
    session = create_session(csv_path, main, analysis_model=child, skills_enabled=True)
    calls = inject_executor(session, monkeypatch)
    report = session.invoke("统计各组并导出完整表，不绘图")
    assert report["status"] == "completed", report.get("error")
    assert len(calls) == 1 and report["analysis_model_calls"] == 3
    assert report["skills"]["loads"] == [{
        "role": "analysis", "tool_call_id": "child-skill", "skill_id": "statistical-analysis",
        "parent_call_id": "delegate",
        "sha256": SkillLibrary().read("statistical-analysis")["sha256"],
    }]
    run = report["analysis_runs"][0]
    assert [{**s, "parent_call_id": run["parent_call_id"]} for s in run["skill_loads"]] == report["skills"]["loads"]
    assert all(set(names) == {"read_skill", "execute_python", "read_artifact"}
               for names in run["tool_exposure"])
    assert validate_evidence(run["trace"], run["execution_ledger"]) is None
    paired(report)
    assert session.harness.model_limit == 12
    assert session.analyst.model_calls == 3


@pytest.mark.parametrize("name,args", [
    ("read_file", {"file_path": ".env"}),
    ("execute", {"command": "whoami"}),
    ("run_command", {"command": "whoami"}),
])
def test_skills_do_not_open_filesystem_or_shell_tools(csv_path, name, args):
    model = ScriptedModel(responses=[call(), skill_call(), call(name, "forbidden", args)])
    report = create_session(csv_path, model, skills_enabled=True).invoke("概述字段")
    assert report["status"] == "error" and report["error"]["code"] == "tool_not_allowed"
    assert all(name not in exposed for exposed in report["tool_exposure"])


def test_actual_skill_tool_replay_does_not_read_or_charge_twice(monkeypatch):
    library = SkillLibrary()
    reads, events = [], []
    original = library.read
    def tracked(skill_id):
        reads.append(skill_id)
        return original(skill_id)
    monkeypatch.setattr(library, "read", tracked)
    harness = ExecutionHarness({"read_skill": 3})
    harness._advertised = {"read_skill"}
    actual = read_skill_tool(library, harness, events.append, "main")
    invocation = {"name": "read_skill", "id": "same", "type": "tool_call",
                  "args": {"skill_id": "data-exploration"}}
    handler = lambda request: actual.invoke(request.tool_call)
    for _ in range(2):
        response = harness.wrap_tool_call(SimpleNamespace(tool_call=invocation), handler)
        assert isinstance(response, ToolMessage)
        assert json.loads(response.content)["skill_id"] == "data-exploration"
    assert reads == ["data-exploration"]
    assert harness.tool_counts["read_skill"] == len(harness.execution_ledger) == len(events) == 1
    altered = {**copy.deepcopy(invocation), "args": {"skill_id": "data-visualization"}}
    result = harness.wrap_tool_call(SimpleNamespace(tool_call=altered), handler)
    assert result.status == "error" and harness.failure == "reused_tool_call_id"
    assert reads == ["data-exploration"]


def test_resume_keeps_skill_budget_and_new_task_starts_fresh(csv_path):
    model = ScriptedModel(responses=[
        call(), skill_call("before"),
        call("ask_user", "question", {"question": "需要解释哪些字段？"}),
        skill_call("after", "analysis-validation"), answer(),
        skill_call("next", "data-visualization"), answer(),
    ])
    session = create_session(csv_path, model, skills_enabled=True)
    pending = session.invoke("概述实际字段")
    assert pending["status"] == "awaiting_input"
    assert session.harness.tool_counts["read_skill"] == 1
    finished = session.resume("全部字段")
    assert finished["status"] == "completed", finished.get("error")
    assert session.harness.tool_counts["read_skill"] == 2
    assert [s["tool_call_id"] for s in finished["skills"]["loads"]] == ["before", "after"]
    assert finished["replies"] == ["全部字段"]
    next_task = session.invoke("换一个角度概述字段")
    assert next_task["status"] == "completed", next_task.get("error")
    assert session.harness.tool_counts["read_skill"] == 1
    assert [s["tool_call_id"] for s in next_task["skills"]["loads"]] == ["next"]


def test_skill_budget_does_not_reset_after_clarification(csv_path):
    model = ScriptedModel(responses=[
        call(), skill_call("one"),
        call("ask_user", "question", {"question": "需要解释哪些字段？"}),
        skill_call("two"), skill_call("three"), skill_call("four"),
    ])
    session = create_session(csv_path, model, skills_enabled=True)
    assert session.invoke("概述实际字段")["status"] == "awaiting_input"
    report = session.resume("全部字段")
    assert report["status"] == "error" and report["error"]["code"] == "tool_budget_exhausted"
    assert len(report["skills"]["loads"]) == 3
    assert session.harness.tool_counts["read_skill"] == 4
