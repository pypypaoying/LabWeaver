"""Real LangGraph/Deep Agents flow; injected transport fixtures never execute Python on host."""

import copy
import hashlib
import json
import socket
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr
import pytest
from labweaver.agent import create_session, run_intake
from labweaver.offline import OfflineIntakeModel
from labweaver.runtime.artifacts import validate_payload


class ScriptedModel(OfflineIntakeModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    def _generate(self, messages, **kwargs):
        if self._cursor >= len(self.responses):
            raise AssertionError("Unexpected extra model call")
        message = copy.deepcopy(self.responses[self._cursor])
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def call(name="profile_csv", cid="p1", args=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": cid, "type": "tool_call"}],
    )


def answer(kind="summary", text="已按实际数据完成概览。"):
    return AIMessage(
        content=json.dumps(
            {
                "task_kind": kind,
                "retrieval_reason": "CSV 足够完成本任务",
                "answer": text,
            },
            ensure_ascii=False,
        )
    )


def paired(report):
    calls = [v for v in report["trace"] if v["kind"] == "tool_call"]
    results = [v for v in report["trace"] if v["kind"] == "tool_result"]
    assert len(calls) == len(results) == len(report["execution_ledger"])
    for call in calls:
        result = next(v for v in results if v["tool_call_id"] == call["id"])
        ledger = next(
            v for v in report["execution_ledger"] if v["tool_call_id"] == call["id"]
        )
        assert call["name"] == result["name"] == ledger["name"]
        assert call["args"] == ledger["arguments"]
        assert json.loads(result["content"]) == ledger["result"]


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "study.csv"
    path.write_text("group,score\nA,2\nB,4\nA,\n", encoding="utf-8")
    return path


def test_summary_no_network_or_container(csv_path, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Offline connected")

    monkeypatch.setattr(socket.socket, "connect", deny)
    before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    session = create_session(csv_path, OfflineIntakeModel())
    monkeypatch.setattr(session.executor, "execute", deny)
    result = session.invoke("概览字段与缺失情况")
    assert result["status"] == "completed", result.get("error")
    assert result["profile_result"]["row_count"] == 3
    assert result["tool_exposure"][0] == ["profile_csv"]
    assert result["artifacts"] == [] and result["analysis_status"] == "not_needed"
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == before
    paired(result)


def test_negated_calculation_and_chart_words_do_not_force_execution(csv_path):
    report = run_intake(
        "仅概述数据规模、字段与缺失情况，不需要计算或绘图。",
        csv_path,
        ScriptedModel(responses=[call(), answer()]),
    )
    assert report["status"] == "completed", report.get("error")


def test_overview_plan_is_delivered_in_report_without_code(csv_path, monkeypatch):
    session = create_session(
        csv_path,
        ScriptedModel(
            responses=[
                call(),
                call(
                    "set_task_plan",
                    "plan",
                    {
                        "deliverables": [
                            {
                                "id": "overview",
                                "kind": "explanation",
                                "description": "概览实际字段",
                            }
                        ]
                    },
                ),
                answer(),
            ]
        ),
    )
    monkeypatch.setattr(
        session.executor,
        "execute",
        lambda *a, **k: pytest.fail("Unnecessary execution"),
    )
    report = session.invoke("仅概述字段，不需要计算或绘图。")
    assert report["status"] == "completed", report.get("error")
    assert report["fulfilled_deliverable_ids"] == ["overview"]
    assert (
        report["answer_deliverables"][0]["source"] == report["profile_result"]["source"]
    )
    assert not report["code_executions"] and report["analysis_status"] == "not_needed"
    paired(report)


@pytest.mark.parametrize(
    "kind,task", [("table", "概览"), ("explanation", "统计各组平均值")]
)
def test_final_text_cannot_replace_computed_outputs(csv_path, kind, task):
    report = run_intake(
        task,
        csv_path,
        ScriptedModel(
            responses=[
                call(),
                call(
                    "set_task_plan",
                    "plan",
                    {
                        "deliverables": [
                            {"id": "result", "kind": kind, "description": "真实结果"}
                        ]
                    },
                ),
                answer(),
            ]
        ),
    )
    assert (
        report["status"] == "error"
        and report["error"]["code"] == "missing_deliverables"
    )
    assert report["answer_deliverables"] == []


@pytest.mark.parametrize(
    "responses,code",
    [
        ([answer()], "profile_failed"),
        ([call("execute", "bad", {"command": "whoami"})], "tool_not_allowed"),
        ([call(), AIMessage(content="not JSON")], "invalid_final_answer"),
        ([call(), answer("calculation")], "missing_task_plan"),
    ],
)
def test_refuses_false_completion(csv_path, responses, code):
    report = run_intake("概览", csv_path, ScriptedModel(responses=responses))
    assert report["status"] == "error"
    assert report["error"]["code"] == code


def plan():
    return [{"id": "table", "kind": "table", "description": "完整统计表"}]


class ReceiptModel(OfflineIntakeModel):
    """Execute real tool dispatch, then return only IDs from actual ToolMessage."""

    code: str = "fixture code"
    fail_first: bool = False
    invented: bool = False

    def _generate(self, messages, **kwargs):
        results = [
            json.loads(m.content)
            for m in messages
            if isinstance(m, ToolMessage) and m.name == "execute_python"
        ]
        if not results or self.fail_first and len(results) == 1:
            return self._call(
                "execute_python", {"code": self.code + (" fixed" if results else "")}
            )
        ids = [a for r in results for a in r.get("artifact_ids", [])]
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "artifact_ids": ["invented"] if self.invented else ids,
                                "summary": "真实工具成果",
                            }
                        )
                    )
                )
            ]
        )


def inject_executor(session, monkeypatch, *, failures=0, empty=False):
    import base64

    calls = []

    def execute(code, snapshot, deliverables, **kwargs):
        calls.append(code)
        record = {
            "id": f"execution-{len(calls)}",
            "code": code,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "source": snapshot.source,
            "status": "completed",
            "image_id": "sha256:" + "a" * 64,
            "image": "fixture-image",
            "exit_code": 0,
            "artifact_ids": [],
            "elapsed_seconds": 0.1,
        }
        if len(calls) <= failures:
            return {
                **record,
                "status": "error",
                "exit_code": 1,
                "stderr": {"text": "NameError: fixture", "truncated": False},
                "error": {"code": "python_failed"},
            }, {}
        payload = (
            []
            if empty
            else [
                {
                    "metadata": {
                        "kind": "table",
                        "deliverable_id": "table",
                        "label": "结果",
                        "row_count": 2,
                        "columns": ["group", "value"],
                    },
                    "files": {
                        "table.csv": base64.b64encode(
                            b"group,value\nA,2\nB,4\n"
                        ).decode()
                    },
                }
            ]
        )
        assets = validate_payload(
            payload,
            deliverables=deliverables,
            source=snapshot.source,
            execution_id=record["id"],
            image_id=record["image_id"],
        )
        record["artifact_ids"] = list(assets)
        return record, assets

    monkeypatch.setattr(session.executor, "execute", execute)
    return calls


def coordinator_responses():
    return [
        call(),
        call("set_task_plan", "plan", {"deliverables": plan()}),
        call(
            "delegate_analysis",
            "delegate",
            {"instruction": "分组统计", "deliverable_ids": ["table"]},
        ),
        answer("calculation"),
    ]


def test_two_agents_repair_real_dispatch_and_export(csv_path, monkeypatch, tmp_path):
    session = create_session(
        csv_path,
        ScriptedModel(responses=coordinator_responses()),
        analysis_model=ReceiptModel(fail_first=True),
    )
    calls = inject_executor(session, monkeypatch, failures=1)
    events = []
    result = session.invoke("分组统计", on_event=events.append)
    assert result["status"] == "completed", result.get("error")
    assert len(calls) == 2 and calls[1].endswith("fixed")
    assert [e["status"] for e in result["code_executions"]] == ["error", "completed"]
    assert result["fulfilled_deliverable_ids"] == ["table"]
    assert result["analysis_model_calls"] == 3
    assert any(e["type"] == "plan" for e in events)
    assert [e["attempt"] for e in events if e["type"] == "execution_start"] == [1, 2]
    assert all(
        set(t) == {"execute_python", "read_artifact"}
        for t in result["analysis_runs"][0]["tool_exposure"]
    )
    paired(result)
    saved = session.save(tmp_path / "runs")
    assert len(saved["result_csv_paths"]) == 1
    assert Path(saved["result_csv_paths"][0]).read_text(
        encoding="utf-8-sig"
    ).splitlines() == ["group,value", "A,2", "B,4"]


@pytest.mark.parametrize("empty,invented", [(True, False), (False, True)])
def test_child_cannot_claim_missing_or_fabricated_artifact(
    csv_path, monkeypatch, empty, invented
):
    session = create_session(
        csv_path,
        ScriptedModel(responses=coordinator_responses()),
        analysis_model=ReceiptModel(invented=invented),
    )
    inject_executor(session, monkeypatch, empty=empty)
    result = session.invoke("分组统计")
    assert result["status"] == "error" and result["missing_deliverables"] == ["table"]
    if invented:
        assert result[
            "artifacts"
        ]  # Preserve valid partial files without marking complete.


def test_clarification_followup_preserves_budget_and_new_scope(csv_path, monkeypatch):
    responses = [
        call(),
        call("ask_user", "ask", {"question": "哪些范围？"}),
        *coordinator_responses()[1:],
        call("set_task_plan", "plan2", {"deliverables": plan()}),
        call(
            "delegate_analysis",
            "delegate2",
            {"instruction": "新范围", "deliverable_ids": ["table"]},
        ),
        answer("calculation"),
    ]
    session = create_session(
        csv_path, ScriptedModel(responses=responses), analysis_model=ReceiptModel()
    )
    calls = inject_executor(session, monkeypatch)
    pending = session.invoke("统计")
    assert pending["status"] == "awaiting_input"
    finished = session.resume("全部年份")
    assert finished["status"] == "completed", finished.get("error")
    assert finished["tool_counts"]["ask_user"] == 1 and finished["replies"] == [
        "全部年份"
    ]
    assert finished["model_calls"] == 5 and len(calls) == 1
    next_task = session.invoke("新范围统计")
    assert next_task["status"] == "completed"
    assert next_task["task_id"] != finished["task_id"]
    assert next_task["model_calls"] == 3 and len(calls) == 2
    assert next_task["tool_counts"]["profile_csv"] == 0


def test_cancel_and_pending_invalid_transitions(csv_path):
    session = create_session(
        csv_path,
        ScriptedModel(
            responses=[call(), call("ask_user", "ask", {"question": "范围？"})]
        ),
    )
    assert session.invoke("统计")["status"] == "awaiting_input"
    with pytest.raises(ValueError):
        session.invoke("另一个任务")
    assert session.cancel()["status"] == "cancelled"
    with pytest.raises(ValueError):
        session.resume("全部")


from pathlib import Path
