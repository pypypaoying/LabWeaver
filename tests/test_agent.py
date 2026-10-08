"""Task-level acceptance through real Deep Agents and LangGraph interrupts."""

import copy
import hashlib
import json
import socket
from types import SimpleNamespace
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr
from labweaver.agent import (
    create_session,
    run_intake,
    _IntakeHarness,
    _IntakeRejected,
    _validate_evidence,
)
from labweaver.offline import OfflineIntakeModel


class ScriptedModel(OfflineIntakeModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        position = min(self._cursor, len(self.responses) - 1)
        self._cursor += 1
        return ChatResult(
            generations=[ChatGeneration(message=self.responses[position])]
        )


def call(name="profile_csv", cid="p1", args=None):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {} if args is None else args,
                "id": cid,
                "type": "tool_call",
            }
        ],
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


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "study.csv"
    p.write_text("group,score\nA,2\nB,4\nA,\n", encoding="utf8")
    return p


def paired(report):
    calls = [v for v in report["trace"] if v["kind"] == "tool_call"]
    results = [v for v in report["trace"] if v["kind"] == "tool_result"]
    assert len(calls) == len(results) == len(report["execution_ledger"])
    for c in calls:
        r = next(v for v in results if v["tool_call_id"] == c["id"])
        e = next(v for v in report["execution_ledger"] if v["tool_call_id"] == c["id"])
        assert c["name"] == r["name"] == e["name"]
        assert c["args"] == e["arguments"]
        assert json.loads(r["content"]) == e["result"]


def test_offline_summary_real_tools_no_network(csv_path, monkeypatch):
    connections = []

    def deny(*a, **k):
        connections.append(True)
        raise AssertionError("offline connected")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    model = OfflineIntakeModel()
    r = run_intake("概览字段与缺失情况", csv_path, model)
    assert r["status"] == "completed"
    assert r["profile"]["row_count"] == 3 and r["profile"]["column_count"] == 2
    assert r["model_calls"] == 2 and r["tool_counts"]["profile_csv"] == 1
    assert set(model.bound_tool_sets[0]) == {"profile_csv"}
    assert set(model.bound_tool_sets[-1]) == {"analyze_csv", "ask_user", "prepare_distribution"}
    assert "3 行、2 列" in r["final_answer"]
    assert r["retrieval_status"] == "unavailable" and not connections
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == before
    paired(r)
    json.dumps(r, allow_nan=False)


def test_medal_clarify_execute_followup_and_budget():
    p = Path("tests/fixtures/medals.csv")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    s = create_session(p, OfflineIntakeModel())
    graph = s.graph
    r = s.invoke("统计出获得最多奖牌的前5个国家")
    assert r["status"] == "awaiting_input" and r["question"]
    assert r["tool_counts"] == {
        "profile_csv": 1,
        "search_materials": 0,
        "analyze_csv": 0,
        "ask_user": 1,
        "prepare_distribution": 0,
        "delegate_visualization": 0,
    }
    assert r["model_calls"] == 2
    task_id = r["task_id"]
    r = s.resume("全部年份、按 Total 累计、保留原始 NOC")
    assert r["status"] == "completed" and r["task_id"] == task_id
    assert r["tool_counts"]["ask_user"] == 1 and r["tool_counts"]["profile_csv"] == 1
    assert r["analysis_results"][0]["rows"] == [
        {"column_2": "Alpha", "total_medals": 22},
        {"column_2": "Beta", "total_medals": 17},
        {"column_2": "Gamma", "total_medals": 11},
        {"column_2": "Delta", "total_medals": 7},
        {"column_2": "Epsilon", "total_medals": 5},
    ]
    assert "Alpha | 22" in r["final_answer"]
    assert r["replies"] == ["全部年份、按 Total 累计、保留原始 NOC"]
    paired(r)
    r = s.invoke("改为2024年")
    assert s.graph is graph
    assert r["status"] == "completed" and r["task_id"] != task_id
    assert r["tool_counts"]["profile_csv"] == r["tool_counts"]["ask_user"] == 0
    assert r["model_calls"] == 2 and r["tool_counts"]["analyze_csv"] == 1
    assert r["analysis_results"][0]["rows"] == [
        {"column_2": "Alpha", "total_medals": 12},
        {"column_2": "Beta", "total_medals": 9},
        {"column_2": "Gamma", "total_medals": 6},
        {"column_2": "Delta", "total_medals": 4},
        {"column_2": "Epsilon", "total_medals": 3},
    ]
    paired(r)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


def test_no_fake_success_without_profile(csv_path):
    r = run_intake("概览", csv_path, ScriptedModel(responses=[answer()]))
    assert r["status"] == "error" and r["error"]["code"] == "missing_tool_call"


@pytest.mark.parametrize(
    "name,args",
    [
        ("write_file", {"file_path": "/bad", "content": "bad"}),
        ("task", {"description": "delegate", "subagent_type": "general-purpose"}),
        ("unknown", {}),
    ],
)
def test_forbidden_dispatch(csv_path, name, args):
    r = run_intake(
        "概览", csv_path, ScriptedModel(responses=[call(name, args=args), answer()])
    )
    assert r["status"] == "error" and r["error"]["code"] == "tool_not_allowed"
    assert r["execution_ledger"] == []


def test_path_injection(csv_path):
    r = run_intake(
        "概览",
        csv_path,
        ScriptedModel(responses=[call(args={"path": "other.csv"}), answer()]),
    )
    assert r["status"] == "error" and r["error"]["code"] == "invalid_tool_arguments"
    assert r["execution_ledger"] == []


def test_readonly_calculation_must_execute(csv_path):
    r = run_intake(
        "统计各组得分均值", csv_path, ScriptedModel(responses=[call(), answer()])
    )
    assert r["status"] == "error" and r["error"]["code"] == "missing_analysis_result"
    assert r["analysis_results"] == []


def test_model_declared_calculation_needs_execution(csv_path):
    r = run_intake(
        "查看 score", csv_path, ScriptedModel(responses=[call(), answer("calculation")])
    )
    assert r["error"]["code"] == "missing_analysis_result"


def test_failed_analysis_never_completed_as_summary(csv_path):
    r = run_intake(
        "查看 score",
        csv_path,
        ScriptedModel(
            responses=[call(), call("analyze_csv", "a1", {"spec": {}}), answer()]
        ),
    )
    assert r["status"] == "error" and r["error"]["code"] == "missing_analysis_result"
    assert r["execution_ledger"][-1]["result"]["status"] == "error"


def test_analyze_before_profile_rejected(csv_path):
    r = run_intake(
        "概览",
        csv_path,
        ScriptedModel(responses=[call("analyze_csv", args={"spec": {}}), answer()]),
    )
    assert r["error"]["code"] == "tool_before_profile"
    assert r["execution_ledger"] == []


def test_duplicate_profile_and_parallel_budgets(csv_path):
    for responses in (
        [call(), call(cid="p2"), answer()],
        [
            AIMessage(
                content="", tool_calls=call().tool_calls + call(cid="p2").tool_calls
            ),
            answer(),
        ],
    ):
        r = run_intake("概览", csv_path, ScriptedModel(responses=responses))
        assert r["error"]["code"] == "tool_budget_exhausted"
        assert len(r["execution_ledger"]) == 1 and r["profile_completed"]


def test_id_and_content_tampering_detected(csv_path):
    r = run_intake("概览", csv_path, OfflineIntakeModel())
    for mutation in ("id", "content", "args"):
        trace = copy.deepcopy(r["trace"])
        if mutation == "id":
            trace[1]["tool_call_id"] = "fake"
        elif mutation == "content":
            trace[1]["content"] = "{}"
        else:
            trace[0]["args"] = {"path": "fake.csv"}
        assert (
            _validate_evidence(trace, r["execution_ledger"])[1]
            == "unmatched_tool_evidence"
        )
    assert _validate_evidence(r["trace"], [])[1] == "missing_execution_evidence"


def test_model_budget_12():
    h = _IntakeHarness()

    class Request:
        tools = [SimpleNamespace(name="profile_csv")]

        def override(self, **values):
            return self

    count = []

    def handler(r):
        count.append(1)
        return SimpleNamespace(result=[])

    for _ in range(12):
        h.wrap_model_call(Request(), handler)
    with pytest.raises(_IntakeRejected, match="model_budget_exhausted"):
        h.wrap_model_call(Request(), handler)
    assert len(count) == h.model_calls == 12


def test_replay_dispatch_deduplication():
    h = _IntakeHarness()
    h.profile_ready = True
    h._advertised = {"analyze_csv", "ask_user"}
    request = SimpleNamespace(
        tool_call={"name": "analyze_csv", "args": {"spec": {}}, "id": "a1"}
    )
    runs = []

    def handler(req):
        runs.append(1)
        out = {"status": "completed", "rows": []}
        h.record_execution(out, "analyze_csv")
        return __import__(
            "langchain_core.messages", fromlist=["ToolMessage"]
        ).ToolMessage(content=json.dumps(out), tool_call_id="a1", name="analyze_csv")

    h.wrap_tool_call(request, handler)
    h.wrap_tool_call(request, handler)
    assert len(runs) == h.tool_counts["analyze_csv"] == len(h.execution_ledger) == 1
    request.tool_call["args"] = {"spec": {"top_k": 2}}
    assert h.wrap_tool_call(request, handler).status == "error"
    assert h.failure == "reused_tool_call_id"


def test_budget_resume_not_reset(csv_path):
    responses = (
        [call()]
        + [call("ask_user", f"q{i}", {"question": f"关键口径{i}?"}) for i in range(4)]
        + [answer()]
    )
    s = create_session(csv_path, ScriptedModel(responses=responses))
    r = s.invoke("查看数据")
    for i in range(3):
        assert r["status"] == "awaiting_input"
        assert r["tool_counts"]["ask_user"] == i + 1
        r = s.resume("回答")
    assert (
        r["status"] == "error"
        and r["error"]["code"] == "clarification_budget_exhausted"
    )
    assert len([e for e in r["execution_ledger"] if e["name"] == "ask_user"]) == 3


def test_cancel_and_invalid_transitions():
    s = create_session("tests/fixtures/medals.csv", OfflineIntakeModel())
    with pytest.raises(ValueError):
        s.resume("x")
    s.invoke("前5个国家")
    with pytest.raises(ValueError):
        s.invoke("另一个任务")
    with pytest.raises(ValueError):
        s.resume("  ")
    assert s.cancel()["status"] == "cancelled"
    with pytest.raises(ValueError):
        s.resume("回答")


def test_file_error_is_evidenced(tmp_path):
    r = run_intake("概览", tmp_path / "missing.csv", OfflineIntakeModel())
    assert r["status"] == "error" and r["error"]["code"] == "profile_failed"
    assert len(r["execution_ledger"]) == 1 and not r["profile_completed"]


def test_oversized_profile_bound_and_retry(tmp_path):
    p = tmp_path / "long.csv"
    p.write_text("a,b\n" + "测" * 20000 + "," + "测" * 20000 + "\n", encoding="utf8")
    r = run_intake("概览", p, OfflineIntakeModel())
    assert r["error"]["code"] == "profile_output_too_large"
    assert len(json.dumps(r, ensure_ascii=False).encode()) < 6000
    assert (
        run_intake("概览", p, OfflineIntakeModel(), sample_rows=0)["status"]
        == "completed"
    )


def test_upstream_errors_sanitized(csv_path):
    class Bad(OfflineIntakeModel):
        def _generate(self, *a, **k):
            raise RuntimeError("api_key=private-secret")

    r = run_intake("概览", csv_path, Bad())
    assert r["error"]["code"] == "agent_failed" and r["diagnostics"] == {
        "exception_type": "RuntimeError"
    }
    assert "private-secret" not in json.dumps(r)


def test_profile_survives_later_model_error(csv_path):
    class Bad(OfflineIntakeModel):
        def _generate(self, messages, *a, **k):
            if any(m.type == "tool" for m in messages):
                raise RuntimeError("bad")
            return super()._generate(messages, *a, **k)

    r = run_intake("概览", csv_path, Bad())
    assert (
        r["status"] == "error"
        and r["profile_completed"]
        and len(r["execution_ledger"]) == 1
    )


def test_empty_task_no_model_calls(csv_path):
    r = run_intake(" ", csv_path, OfflineIntakeModel())
    assert (
        r["error"]["code"] == "invalid_task"
        and r["model_calls"] == r["tool_attempts"] == 0
    )


def test_malformed_final_json(csv_path):
    r = run_intake(
        "概览",
        csv_path,
        ScriptedModel(responses=[call(), AIMessage(content="finished")]),
    )
    assert r["error"]["code"] == "invalid_final_answer"
