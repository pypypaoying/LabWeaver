"""Scope model transport by skills mode while retaining one execution path.

Frozen-source comparisons run in separate interpreters so module caching cannot
silently turn a baseline comparison into a current-versus-current comparison.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from labweaver import analysis as analysis_module
from labweaver.analysis import ANALYSIS_PROMPT, AnalysisRunner
from labweaver.modeling import ANALYSIS_POLICY
from labweaver.offline import OfflineIntakeModel
from labweaver.runtime.skills import skill_context
from test_complex_analysis import PLAN, code_call, no_network, runner_factory

FULL_ANALYSIS_PROMPT = getattr(analysis_module, "FULL_ANALYSIS_PROMPT", None)


class FixtureSkills:
    def catalog(self):
        return [{"id": "fixture", "description": "确定性技能资料"}]

    def read(self, skill_id):
        assert skill_id == "fixture"
        return {"status": "completed", "skill_id": skill_id,
                "sha256": "b" * 64, "content": "使用实际字段和授权交付项。"}


class CapturingModel(OfflineIntakeModel):
    received: list = Field(default_factory=list)
    tool_schemas: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tool_schemas.append([convert_to_openai_tool(t) for t in tools])
        return super().bind_tools(tools, **kwargs)

    def _generate(self, messages, **kwargs):
        # Only LangChain-assigned message IDs are omitted. Prompt strings,
        # serialized packet strings, tool arguments/results and schemas remain exact.
        self.received.append([m.model_dump(exclude={"id"}) for m in messages])
        results = [m for m in messages if isinstance(m, ToolMessage)]
        names = [m.name for m in results]
        if "read_skill" in self._available_tools and "read_skill" not in names:
            message = AIMessage(content="", tool_calls=[{
                "name": "read_skill", "args": {"skill_id": "fixture"},
                "id": "read-fixture", "type": "tool_call",
            }])
        elif "execute_python" not in names:
            message = code_call("print('complete deterministic fixture')")
        else:
            receipt = json.loads(next(m.content for m in reversed(results)
                                      if m.name == "execute_python"))
            if "read_artifact" not in names:
                message = AIMessage(content="", tool_calls=[{
                    "name": "read_artifact", "args": {"artifact_id": receipt["artifact_ids"][0]},
                    "id": "read-result", "type": "tool_call",
                }])
            else:
                message = AIMessage(content=json.dumps({
                    "status": "completed", "artifact_ids": receipt["artifact_ids"],
                    "summary": "使用真实成果回执完成。",
                }, ensure_ascii=False))
        return ChatResult(generations=[ChatGeneration(message=message)])


def capture_payloads(tmp_path, skills_enabled, compact_model_context=None):
    """Exercise failure, provisional repair, successful execution and explicit read."""
    from labweaver.runtime import artifacts, verification

    model = CapturingModel()
    original, assets, calls, events = runner_factory.__wrapped__(tmp_path)(model)
    library = FixtureSkills() if skills_enabled else None
    kwargs = {} if compact_model_context is None else {"compact_model_context": compact_model_context}
    runner = AnalysisRunner(
        model, original.executor, get_snapshot=original.get_snapshot,
        get_task_spec=lambda: {
            "goal": "计算全部记录", "original_task": "原始要求须完整保留。",
            "confirmed_replies": ["保留零值"],
        },
        register=original.register, read_asset=original.read_asset,
        progress=events.append, skill_library=library, **kwargs,
    )
    context = json.dumps({
        "task_spec": runner.get_task_spec(), "original_task": "原始要求须完整保留。",
        "replies": ["保留零值"], "retrieved_chunks": [{"id": "evidence-1", "text": "真实资料"}],
    }, ensure_ascii=False)
    old_uuid, old_verify = artifacts.uuid, verification.verify_artifacts
    verification_results = iter([False, True])
    artifact_ids = iter(range(1, 10))
    artifacts.uuid = SimpleNamespace(uuid4=lambda: uuid.UUID(int=next(artifact_ids)))
    verification.verify_artifacts = lambda *args: {
        "passed": next(verification_results),
        "results": [{"check": "fixture", "status": "completed", "evidence": {"actual": 6}}],
    }
    try:
        first = runner.delegate("计算完整结果", PLAN, "initial", context=context)
        second = runner.delegate("修复相同授权交付", PLAN, "repair", context=context)
    finally:
        artifacts.uuid, verification.verify_artifacts = old_uuid, old_verify
    assert first["status"] == "error", first
    assert second["status"] == "completed", second
    assert len(calls) == 2 and len(runner.attempts) == 2
    return {
        "messages": model.received, "tool_schemas": model.tool_schemas,
        "results": [first, second], "calls": calls,
        "host_assets": [{"metadata": asset["metadata"],
                         "files": {k: raw.hex() for k, raw in asset["files"].items()}}
                        for asset in assets.values()],
    }


@pytest.mark.parametrize("skills_enabled", [False, True])
def test_default_mode_preserves_complete_transport_and_host_evidence(tmp_path, skills_enabled):
    captured = capture_payloads(tmp_path, skills_enabled)
    prompt = captured["messages"][0][0]["content"]
    expected = (FULL_ANALYSIS_PROMPT + ANALYSIS_POLICY if skills_enabled else ANALYSIS_PROMPT)
    library = FixtureSkills() if skills_enabled else None
    names = "execute_python, read_artifact" + (", read_skill" if skills_enabled else "")
    assert prompt == expected + skill_context(library) + "\n当前开放工具：" + names
    packets = [json.loads(m[1]["content"]) for m in captured["messages"]]
    first, repair = packets[0], packets[-1]
    assert first["task_spec"]["original_task"] == "原始要求须完整保留。"
    assert (isinstance(first["context"], str)) is skills_enabled
    assert ("name" in first["profile"]["columns"][0]) is skills_enabled
    assert ("position" in first["profile"]["columns"][0]) is skills_enabled
    if not skills_enabled:
        assert first["context"] == {"retrieved_chunks": [{"id": "evidence-1", "text": "真实资料"}]}
    else:
        assert json.loads(first["context"])["task_spec"] == first["task_spec"]
    assert repair["previous_failed_attempt"]["code"] == captured["calls"][0]
    assert repair["previous_failed_attempt"]["verification"]["passed"] is False
    receipt = captured["results"][1]["artifacts"][0]
    assert ("files" in receipt) is skills_enabled
    assert ("execution_id" in receipt) is skills_enabled
    assert ("image_id" in receipt) is skills_enabled
    assert ("files" in repair["provisional_artifacts"][0]) is skills_enabled
    assert receipt["source"] == captured["host_assets"][0]["metadata"]["source"]
    assert captured["host_assets"][0]["metadata"]["files"]["table.csv"]["sha256"]
    assert captured["host_assets"][0]["files"]["table.csv"] == b"group,value\nA,2\nB,4\n".hex()
    # Explicit artifact reads retain full provenance in either mode.
    explicit_reads = [json.loads(m["content"]) for request in captured["messages"]
                      for m in request if m["type"] == "tool" and m["name"] == "read_artifact"]
    assert explicit_reads and all("files" in r and "execution_id" in r for r in explicit_reads)


@pytest.mark.parametrize("skills_enabled, compact", [(False, False), (True, True)])
def test_explicit_internal_override_selects_only_transport(tmp_path, skills_enabled, compact):
    captured = capture_payloads(tmp_path, skills_enabled, compact)
    packet = json.loads(captured["messages"][0][1]["content"])
    assert (isinstance(packet["context"], str)) is not compact
    assert ("files" in captured["results"][1]["artifacts"][0]) is not compact
    assert ("read_skill" in captured["messages"][0][0]["content"]) is skills_enabled


def isolated_capture(source, fixture_path, skills_enabled):
    root = Path(__file__).resolve().parents[1]
    # Import the requested implementation before the shared fixture module.
    script = """
import json, pathlib, socket, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import labweaver.analysis as analysis
assert pathlib.Path(analysis.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
from test_scoped_analysis_context import capture_payloads
def deny(*args, **kwargs):
    raise AssertionError('Frozen transport comparison attempted networking')
socket.socket.connect = deny
print(json.dumps(capture_payloads(pathlib.Path(sys.argv[3]), sys.argv[4] == '1'), ensure_ascii=False))
"""
    env = {**os.environ, "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
           "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(root / "tests"),
         str(fixture_path), "1" if skills_enabled else "0"],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("skills_enabled, frozen", [
    (False, "token-efficiency-off-v3/candidate/implementation/src"),
    (True, "token-efficiency-v1/baseline/implementation/src"),
])
def test_actual_model_payload_equals_frozen_mode(tmp_path, skills_enabled, frozen):
    root = Path(__file__).resolve().parents[1]
    source = root / ".cache/evaluation" / frozen
    if not source.is_dir():
        pytest.skip("Local frozen experiment implementation is not present")
    fixture = tmp_path / "shared"
    fixture.mkdir()
    expected = isolated_capture(source, fixture, skills_enabled)
    actual = isolated_capture(root / "src", fixture, skills_enabled)
    assert actual == expected
