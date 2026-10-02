"""Offline checks of the real Agent loop and its evidence/budget boundaries."""

from __future__ import annotations

import copy
import json
import socket
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from labweaver.agent import (
    _IntakeHarness,
    _IntakeRejected,
    _validate_evidence,
    run_intake,
)
from labweaver.offline import OfflineIntakeModel
from labweaver.tools.csv_profile import profile_csv


class ScriptedModel(OfflineIntakeModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        position = min(self._cursor, len(self.responses) - 1)
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[position])])


def _call(name="profile_csv", call_id="test-call-1", args=None):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": {} if args is None else args,
        "id": call_id, "type": "tool_call",
    }])


@pytest.fixture(autouse=True)
def _disable_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "study.csv"
    path.write_text("group,score\nA,2\nB,4\nA,\n", encoding="utf-8")
    return path


def test_offline_runs_real_tool_loop_without_network(csv_path, monkeypatch):
    attempts = []

    def deny_network(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Offline Agent attempted a network connection.")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    model = OfflineIntakeModel()
    report = run_intake("比较各组得分，先给出概览。", csv_path, model)
    assert report["status"] == "awaiting_confirmation"
    assert report["stage"] == "intake"
    assert report["profile_observed"] is True
    assert report["profile_completed"] is True
    assert report["profile"]["row_count"] == 3
    assert report["source"] == report["profile"]["source"]
    assert report["source"]["name"] == "study.csv"
    assert str(csv_path.parent) not in json.dumps(report)
    assert report["profile"]["column_count"] == 2
    assert report["model_calls"] == 2
    assert report["tool_attempts"] == 1
    assert len(report["execution_ledger"]) == 1
    assert model.bound_tool_sets == [["profile_csv"], ["profile_csv"]]
    assert model.seen_tool_results == ["offline-profile-1"]
    call, result = report["trace"]
    assert call["name"] == result["name"] == "profile_csv"
    assert call["id"] == result["tool_call_id"] == "offline-profile-1"
    assert "3 行、2 列" in report["final_answer"]
    assert "等待用户确认" in report["final_answer"]
    assert attempts == []
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_offline_summary_changes_with_actual_dataset(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("name,cost\na,1\nb,2\n", encoding="utf-8")
    second.write_text("temperature\n20\n", encoding="utf-8")
    report_one = run_intake("理解成本数据", first, OfflineIntakeModel())
    report_two = run_intake("理解实验测量", second, OfflineIntakeModel())
    assert report_one["status"] == report_two["status"] == "awaiting_confirmation"
    assert "2 行、2 列" in report_one["final_answer"]
    assert "name, cost" in report_one["final_answer"]
    assert "1 行、1 列" in report_two["final_answer"]
    assert "temperature" in report_two["final_answer"]
    assert report_one["profile"]["source"]["sha256"] != report_two["profile"]["source"]["sha256"]


def test_no_tool_call_cannot_be_counted_as_profile_success(csv_path):
    report = run_intake("检查数据", csv_path,
                        ScriptedModel(responses=[AIMessage(content="数据已检查，全部正常。")]))
    assert report["status"] == "error"
    assert report["error"]["code"] == "missing_tool_call"
    assert report["profile_observed"] is False
    assert report["profile_completed"] is False
    assert report["execution_ledger"] == []


@pytest.mark.parametrize("name,args", [
    ("write_file", {"file_path": "/forbidden.txt", "content": "do not write"}),
    ("task", {"description": "Delegate the analysis", "subagent_type": "general-purpose"}),
    ("nonexistent_tool", {}),
])
def test_tools_outside_allowlist_cannot_execute(csv_path, name, args):
    model = ScriptedModel(responses=[_call(name, args=args), AIMessage(content="完成。")])
    report = run_intake("检查数据", csv_path, model)
    assert report["status"] == "error"
    assert report["error"]["code"] == "tool_not_allowed"
    assert report["profile_observed"] is False
    assert report["execution_ledger"] == []
    assert all(names == ["profile_csv"] for names in model.bound_tool_sets)


def test_harness_rejects_actual_dispatch_of_nonprofile_tool():
    harness = _IntakeHarness()
    called = []
    request = SimpleNamespace(tool_call={"name": "write_file", "args": {}, "id": "bad-1"})
    response = harness.wrap_tool_call(request, lambda value: called.append(value))
    assert called == []
    assert response.status == "error"
    assert response.tool_call_id == "bad-1"
    assert harness.failure == "tool_not_allowed"


def test_profile_tool_cannot_be_redirected_to_another_path(csv_path):
    model = ScriptedModel(responses=[_call(args={"path": "other.csv"}), AIMessage(content="完成。")])
    report = run_intake("检查数据", csv_path, model)
    assert report["status"] == "error"
    assert report["error"]["code"] == "invalid_tool_arguments"
    assert report["execution_ledger"] == []


def test_repeated_profile_calls_do_not_bypass_execution_budget(csv_path):
    model = ScriptedModel(responses=[
        _call(call_id="first"), _call(call_id="second"), AIMessage(content="完成。"),
    ])
    report = run_intake("检查数据", csv_path, model)
    assert report["status"] == "error"
    assert report["error"]["code"] == "tool_budget_exhausted"
    assert len(report["execution_ledger"]) == 1
    assert report["tool_attempts"] == 2
    assert report["model_calls"] <= 3
    assert report["profile_completed"] is False


def test_parallel_tool_calls_cannot_bypass_execution_budget(csv_path):
    calls = _call(call_id="first").tool_calls + _call(call_id="second").tool_calls
    model = ScriptedModel(responses=[AIMessage(content="", tool_calls=calls), AIMessage(content="完成。")])
    report = run_intake("检查数据", csv_path, model)
    assert report["status"] == "error"
    assert report["error"]["code"] == "tool_budget_exhausted"
    assert len(report["execution_ledger"]) == 1


def test_model_budget_blocks_fourth_handler_call():
    harness = _IntakeHarness()
    handler_calls = []

    class Request:
        tools = [SimpleNamespace(name="profile_csv")]

        def override(self, **values):
            assert [tool.name for tool in values["tools"]] == ["profile_csv"]
            return self

    def handler(request):
        handler_calls.append(request)
        return SimpleNamespace(result=[])

    for _ in range(3):
        harness.wrap_model_call(Request(), handler)
    with pytest.raises(_IntakeRejected, match="model_budget_exhausted"):
        harness.wrap_model_call(Request(), handler)
    assert len(handler_calls) == harness.model_calls == 3


def test_call_result_and_execution_ids_must_match(csv_path):
    successful = run_intake("检查数据", csv_path, OfflineIntakeModel())
    trace = copy.deepcopy(successful["trace"])
    trace[1]["tool_call_id"] = "invented-result-id"
    profile, error = _validate_evidence(trace, successful["execution_ledger"])
    assert profile is None
    assert error == "unmatched_tool_evidence"


def test_result_content_must_match_actual_execution(csv_path):
    successful = run_intake("检查数据", csv_path, OfflineIntakeModel())
    trace = copy.deepcopy(successful["trace"])
    altered = json.loads(trace[1]["content"])
    altered["row_count"] = 9999
    trace[1]["content"] = json.dumps(altered)
    assert _validate_evidence(trace, successful["execution_ledger"])[1] == "unmatched_tool_evidence"


def test_fabricated_tool_message_without_actual_execution_is_not_evidence():
    trace = [{"kind": "tool_call", "name": "profile_csv", "id": "fake"},
             {"kind": "tool_result", "name": "profile_csv", "tool_call_id": "fake",
              "content": '{"status":"completed"}', "status": "success"}]
    assert _validate_evidence(trace, [])[1] == "missing_execution_evidence"


def test_parse_or_file_error_is_not_success(tmp_path):
    report = run_intake("检查数据", tmp_path / "missing.csv", OfflineIntakeModel())
    assert report["status"] == "error"
    assert report["error"]["code"] == "profile_failed"
    assert report["profile_observed"] is False
    assert report["profile_completed"] is False
    assert len(report["execution_ledger"]) == 1
    assert report["source"] == {"name": "missing.csv", "sha256": None}


def test_large_profile_is_a_bounded_error_and_no_samples_allows_retry(tmp_path):
    path = tmp_path / "long-sample.csv"
    long_cell = "测" * 20_000
    path.write_text(f"note_a,note_b\n{long_cell},{long_cell}\n", encoding="utf-8")
    standalone = profile_csv(path)
    assert standalone["status"] == "completed"
    assert len(standalone["sample_rows"][0][0]) == 20_000
    oversized = run_intake("理解数据", path, OfflineIntakeModel())
    assert oversized["status"] == "error"
    assert oversized["error"]["code"] == "profile_output_too_large"
    assert "--sample-rows 0" in oversized["error"]["message"]
    assert oversized["source"] == standalone["source"]
    assert len(json.dumps(oversized, ensure_ascii=False).encode("utf-8")) < 5000
    retry = run_intake("理解数据", path, OfflineIntakeModel(), sample_rows=0)
    assert retry["status"] == "awaiting_confirmation"
    assert retry["profile"]["sample_rows"] == []


def test_model_failure_after_profile_preserves_source_and_execution(csv_path):
    class FailsAfterToolModel(OfflineIntakeModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if any(message.type == "tool" for message in messages):
                raise RuntimeError("upstream model failed")
            return super()._generate(messages, stop, run_manager, **kwargs)

    report = run_intake("检查数据", csv_path, FailsAfterToolModel())
    assert report["status"] == "error"
    assert report["error"]["code"] == "agent_failed"
    assert report["profile_completed"] is False
    assert len(report["execution_ledger"]) == 1
    assert report["source"] == report["execution_ledger"][0]["result"]["source"]
    assert len(report["source"]["sha256"]) == 64
    assert [item["kind"] for item in report["trace"]] == ["tool_call", "tool_result"]


def test_terminal_tool_message_cannot_be_a_final_summary(csv_path, monkeypatch):
    import labweaver.agent as agent_module

    original_builder = agent_module.create_deep_agent

    def build_without_final_message(**kwargs):
        graph = original_builder(**kwargs)

        def invoke(*args, **invoke_kwargs):
            result = graph.invoke(*args, **invoke_kwargs)
            result["messages"] = result["messages"][:-1]
            return result

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(agent_module, "create_deep_agent", build_without_final_message)
    report = run_intake("检查数据", csv_path, OfflineIntakeModel())
    assert report["status"] == "error"
    assert report["error"]["code"] == "missing_final_answer"
    assert report["profile_completed"] is False


def test_agent_exception_details_do_not_leak_credentials(csv_path):
    class FailingModel(OfflineIntakeModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("api_key=secret-test-credential")

    report = run_intake("检查数据", csv_path, FailingModel())
    assert report["status"] == "error"
    assert report["error"]["code"] == "agent_failed"
    assert report["diagnostics"] == {"exception_type": "RuntimeError"}
    assert "secret-test-credential" not in json.dumps(report)


def test_http_error_diagnostics_keep_only_type_and_integer_status(csv_path):
    class UpstreamError(RuntimeError):
        status_code = 401

    class FailingModel(OfflineIntakeModel):
        def _generate(self, *args, **kwargs):
            raise UpstreamError("body contains secret-test-token at https://private.example")

    report = run_intake("检查数据", csv_path, FailingModel())
    assert report["status"] == "error"
    assert report["diagnostics"] == {"exception_type": "UpstreamError", "http_status": 401}
    serialized = json.dumps(report)
    assert "secret-test-token" not in serialized
    assert "private.example" not in serialized


def test_empty_task_is_rejected_without_model_or_tool_calls(csv_path):
    model = OfflineIntakeModel()
    report = run_intake(" ", csv_path, model)
    assert report["error"]["code"] == "invalid_task"
    assert report["model_calls"] == report["tool_attempts"] == 0
    assert model.bound_tool_sets == []
