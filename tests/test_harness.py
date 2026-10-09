"""Replay and dispatch proof without duplicate execution or budget charges."""

import json
from types import SimpleNamespace
import pytest
from langchain_core.messages import ToolMessage
from labweaver.runtime.harness import (
    ExecutionHarness,
    HarnessRejected,
    validate_evidence,
)


def test_tool_call_id_replay_runs_handler_once():
    harness = ExecutionHarness({"execute_python": 3}, reject_invalid=False)
    harness._advertised = {"execute_python"}
    call = {"name": "execute_python", "id": "same", "args": {"code": "full script"}}
    calls = []

    def handler(request):
        calls.append(True)
        output = harness.record(
            {"status": "completed", "artifact_ids": ["registered"]}, "execute_python"
        )
        return ToolMessage(
            content=json.dumps(output), name="execute_python", tool_call_id="same"
        )

    for _ in range(2):
        response = harness.wrap_tool_call(SimpleNamespace(tool_call=call), handler)
        assert json.loads(response.content)["artifact_ids"] == ["registered"]
    assert (
        len(calls)
        == harness.tool_counts["execute_python"]
        == len(harness.execution_ledger)
        == 1
    )
    altered = {**call, "args": {"code": "different script"}}
    assert (
        harness.wrap_tool_call(SimpleNamespace(tool_call=altered), handler).status
        == "error"
    )
    assert harness.failure == "reused_tool_call_id"


def test_execute_attempt_budget_rejects_fourth_call():
    harness = ExecutionHarness({"execute_python": 3})
    harness._advertised = {"execute_python"}

    def handler(request):
        output = harness.record(
            {"status": "error", "error": {"code": "python_failed"}}, "execute_python"
        )
        return ToolMessage(
            content=json.dumps(output),
            name="execute_python",
            tool_call_id=request.tool_call["id"],
        )

    for index in range(4):
        response = harness.wrap_tool_call(
            SimpleNamespace(
                tool_call={
                    "id": str(index),
                    "name": "execute_python",
                    "args": {"code": "script"},
                }
            ),
            handler,
        )
    assert response.status == "error" and len(harness.execution_ledger) == 3


def test_model_budget_counter_is_shared_between_delegations():
    from labweaver.analysis import AnalysisRunner
    from unittest.mock import Mock

    runner = AnalysisRunner(
        Mock(), Mock(), get_snapshot=Mock(), register=Mock(), read_asset=Mock()
    )
    for _ in range(12):
        runner._charge()
    with pytest.raises(HarnessRejected, match="analysis_model_budget_exhausted"):
        runner._charge()
    assert runner.model_calls == 12


def test_evidence_does_not_accept_result_without_execution():
    trace = [
        {"kind": "tool_call", "id": "a", "name": "execute_python", "args": {}},
        {
            "kind": "tool_result",
            "tool_call_id": "a",
            "name": "execute_python",
            "content": "{}",
            "status": "success",
        },
    ]
    assert validate_evidence(trace, []) == "missing_execution_evidence"
