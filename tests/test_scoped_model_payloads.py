"""Exact model-boundary regressions for the public, finite recipe below.

These tests do not establish equivalence for arbitrary tasks or model responses.
They run genuine coordinator and child graphs, with deterministic model responses,
CSV bytes, reviewed-skill fixtures and an artifact-producing executor transport.
No model endpoint, Docker execution, or network connection is allowed.

Recipe: two tasks in one session; profile -> plan -> two delegations in task one,
then plan -> one delegation in task two. Both agents explicitly read actual
registered artifacts; Skills ON also reads a public synthetic skill. This covers
planning before/after admission, bound schemas, full serialized delegation
packets, successful execution/read receipts and prior-task code context.

The committed hashes come from the frozen OFF-v3 and ON-baseline implementations,
never from the implementation under test. Regenerate only with both references:
    python tests/test_scoped_model_payloads.py --write-reference-hashes
Local snapshot comparisons additionally compare the complete captured payloads
in independent interpreters. CI without the ignored snapshots still checks every
current request and binding against the committed reference hashes.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from test_agent import ScriptedModel, answer, call, inject_executor, plan
from test_complex_analysis import ReceiptAfterRepairModel, code_call


ROOT = Path(__file__).resolve().parents[1]
HASH_FIXTURE = Path(__file__).with_name("fixtures") / "scoped_model_payload_hashes.json"
REFERENCES = {
    False: "token-efficiency-off-v3/candidate/implementation/src",
    True: "token-efficiency-v1/baseline/implementation/src",
}
RECIPE = "scoped-model-payloads/public-two-task-three-delegation/v1"
UUID = re.compile(
    r"(?<![0-9a-f])(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32})(?![0-9a-f])"
)


class PublicSkills:
    """A fixed public catalog keeps hashes independent of packaged skill edits."""

    def catalog(self):
        return [{"id": "public-fixture", "description": "Use actual fields and receipts."}]

    def read(self, skill_id):
        assert skill_id == "public-fixture"
        content = "Read the actual CSV. Preserve the authorized deliverables."
        return {
            "status": "completed", "skill_id": skill_id,
            "sha256": hashlib.sha256(content.encode()).hexdigest(), "content": content,
        }


def _request(messages, stop, kwargs):
    # Do not parse/re-serialize message content: JSON whitespace, key order,
    # duplicated facts, policy text and source/provenance fields must all match.
    return {"messages": [m.model_dump() for m in messages], "stop": stop,
            "kwargs": copy.deepcopy(kwargs)}


class CapturingCoordinator(ScriptedModel):
    requests: list = Field(default_factory=list)
    bindings: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bindings.append({"tools": [convert_to_openai_tool(t) for t in tools],
                              "kwargs": copy.deepcopy(kwargs)})
        return super().bind_tools(tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(_request(messages, stop, kwargs))
        result = super()._generate(messages, **kwargs)
        message = result.generations[0].message
        if message.tool_calls and message.tool_calls[0]["name"] == "read_artifact":
            receipt = next(json.loads(m.content) for m in reversed(messages)
                           if isinstance(m, ToolMessage) and m.name == "delegate_analysis")
            message.tool_calls[0]["args"]["artifact_id"] = receipt["artifact_ids"][0]
        return result


class CapturingAnalysis(ReceiptAfterRepairModel):
    requests: list = Field(default_factory=list)
    bindings: list = Field(default_factory=list)
    _executions: int = PrivateAttr(default=0)

    def bind_tools(self, tools, **kwargs):
        self.bindings.append({"tools": [convert_to_openai_tool(t) for t in tools],
                              "kwargs": copy.deepcopy(kwargs)})
        return super().bind_tools(tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(_request(messages, stop, kwargs))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        names = [m.name for m in results]
        if "read_skill" in self._available_tools and "read_skill" not in names:
            return self._call("read_skill", {"skill_id": "public-fixture"})
        if "execute_python" not in names:
            self._executions += 1
            self.responses = [code_call(f"print('public transport fixture {self._executions}')")]
            self._cursor = 0
        elif "read_artifact" not in names:
            receipt = json.loads(next(m.content for m in reversed(results)
                                      if m.name == "execute_python"))
            return self._call("read_artifact", {"artifact_id": receipt["artifact_ids"][0]})
        # The existing helper finishes only from IDs in a genuine ToolMessage.
        return super()._generate(messages, **kwargs)


def normalize_payload(payload):
    """Replace only volatile UUIDs, preserving aliases and occurrence order.

    Our CSV basename and execution IDs are already constant fixture values. All
    other strings (including JSON strings) and mapping/list order remain intact.
    UUIDs are normalized in message IDs, registered artifact/session/task IDs,
    and their subsequent references. Fixed synthetic call IDs remain unchanged.
    """
    identities = {}

    def replace(match):
        value = match.group().replace("-", "")
        return identities.setdefault(value, f"<uuid-{len(identities) + 1}>")

    def visit(value):
        if isinstance(value, str):
            return UUID.sub(replace, value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {visit(key): visit(item) for key, item in value.items()}
        return value

    return visit(payload)


def capture_recipe(directory, skills_enabled):
    """Runs only in an isolated child interpreter with the selected src first."""
    from labweaver.agent import create_session
    import labweaver.skills as skills

    def deny(*args, **kwargs):
        raise AssertionError("Public model-boundary recipe attempted external execution")

    socket.socket.connect = deny
    # Block process creation before constructing a session. The only executor
    # allowed below is inject_executor's public validated artifact transport.
    subprocess.Popen = deny
    directory.mkdir(parents=True, exist_ok=True)
    csv = directory / "public-payload-fixture.csv"
    csv.write_bytes(b"group,score\nA,2\nB,4\nA,\n")
    skill_read = lambda cid: call("read_skill", cid, {"skill_id": "public-fixture"})
    delegate = lambda cid: call("delegate_analysis", cid, {
        "instruction": "Compute the full public result table.", "deliverable_ids": ["table"],
    })
    set_plan = lambda cid: call("set_task_plan", cid, {"deliverables": plan()})
    read = lambda cid: call("read_artifact", cid, {"artifact_id": "replace-with-real-receipt"})
    coordinator = CapturingCoordinator(responses=[
        call("profile_csv", "profile-public"),
        *([skill_read("skill-task-one")] if skills_enabled else []),
        set_plan("plan-task-one"), delegate("delegate-one"), read("read-main-one"),
        delegate("delegate-two"), answer("calculation", "Public fixture receipts checked."),
        *([skill_read("skill-task-two")] if skills_enabled else []),
        set_plan("plan-task-two"), delegate("delegate-three"), read("read-main-two"),
        answer("calculation", "Public follow-up receipts checked."),
    ])
    child = CapturingAnalysis()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(skills, "SkillLibrary", PublicSkills)
        session = create_session(csv, coordinator, analysis_model=child,
                                 skills_enabled=skills_enabled,
                                 history_context="Public prior scope: preserve zero values.")
        patch.setattr(session.executor, "_run", deny, raising=False)
        calls = inject_executor(session, patch)
        reports = [session.invoke("Compute the public result table."),
                   session.invoke("Recompute the same public table from the original rows.")]
        assert all(r["status"] == "completed" for r in reports), [r.get("error") for r in reports]
        assert len(calls) == 3
        assert [len(r["code_executions"]) for r in reports] == [2, 1]
        assert all(r["fulfilled_deliverable_ids"] == ["table"] for r in reports)
        assert all(a["files"]["table.csv"] == b"group,value\nA,2\nB,4\n"
                   for a in session.artifact_assets.values())
    payload = normalize_payload({
        "main": {"bindings": coordinator.bindings, "requests": coordinator.requests},
        "analysis": {"bindings": child.bindings, "requests": child.requests},
    })
    # Side-channel evidence is not part of the model-input equality comparison.
    observations = {
        "statuses": [r["status"] for r in reports], "executions": calls,
        "fulfilled": [r["fulfilled_deliverable_ids"] for r in reports],
        "policy": [r.get("model_context_policy") for r in reports],
        "usage": [r.get("token_usage") for r in reports],
    }
    return {"payload": payload, "observations": observations}


def isolated_capture(source, directory, skills_enabled):
    script = """
import json, pathlib, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import labweaver.agent as agent
import labweaver.analysis as analysis
source = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(agent.__file__).resolve().is_relative_to(source)
assert pathlib.Path(analysis.__file__).resolve().is_relative_to(source)
from test_scoped_model_payloads import capture_recipe
print(json.dumps(capture_recipe(pathlib.Path(sys.argv[3]), sys.argv[4] == '1'), ensure_ascii=False))
"""
    env = {**os.environ, "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
           "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(source), str(ROOT / "tests"),
         str(directory), "1" if skills_enabled else "0"],
        cwd=ROOT, env=env, capture_output=True, encoding="utf-8", timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def fingerprints(payload):
    def digest(value):
        # sort_keys deliberately omitted: no JSON text is canonicalized away.
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return {f"{role}.{kind}": [digest(item) for item in values]
            for role, streams in payload.items() for kind, values in streams.items()}


@pytest.fixture(scope="module", params=[False, True], ids=["skills-off-v3", "skills-on-baseline"])
def captured(request, tmp_path_factory):
    enabled = request.param
    directory = tmp_path_factory.mktemp("public-model-payloads")
    return enabled, isolated_capture(ROOT / "src", directory, enabled)


def test_current_payload_matches_public_reference_hashes(captured):
    enabled, current = captured
    fixture = json.loads(HASH_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["recipe"] == RECIPE
    key = "skills_on" if enabled else "skills_off"
    assert fingerprints(current["payload"]) == fixture["references"][key]["sha256"]


def _tool_results(requests, name):
    return [json.loads(m["content"]) for request in requests for m in request["messages"]
            if m["type"] == "tool" and m["name"] == name]


def test_recipe_covers_phase_schemas_and_actual_receipts(captured):
    enabled, current = captured
    main, analysis = (current["payload"][role] for role in ("main", "analysis"))
    assert len(main["requests"]) == (12 if enabled else 10)
    assert len(analysis["requests"]) == (12 if enabled else 9)
    assert len(main["bindings"]) == len(main["requests"])
    assert len(analysis["bindings"]) == len(analysis["requests"])
    schemas = [[tool["function"]["name"] for tool in bound["tools"]]
               for bound in main["bindings"]]
    assert schemas[0] == ["profile_csv"]
    assert "set_task_plan" in schemas[1]
    # First request following accepted planning: a direct model-boundary check.
    post_plan = 3 if enabled else 2
    assert ("set_task_plan" in schemas[post_plan]) is enabled
    assert "delegate_analysis" in schemas[post_plan]
    assert all(("read_skill" in names) is enabled for names in schemas[1:])
    plan_schemas = [t["function"] for b in main["bindings"] for t in b["tools"]
                    if t["function"]["name"] == "set_task_plan"]
    assert all("task_spec" in schema["parameters"]["properties"] for schema in plan_schemas)
    assert _tool_results(main["requests"], "profile_csv")[0]["row_count"] == 3
    assert _tool_results(main["requests"], "set_task_plan")[0]["status"] == "completed"
    delegates = _tool_results(main["requests"], "delegate_analysis")
    assert delegates and all(r["status"] == "completed" and r["artifact_ids"] for r in delegates)
    executions = _tool_results(analysis["requests"], "execute_python")
    assert executions and all(r["status"] == "completed" and r["artifact_ids"] for r in executions)
    for role in (main, analysis):
        reads = _tool_results(role["requests"], "read_artifact")
        assert reads and all(r["status"] == "completed" and r["rows"] for r in reads)
        assert all(("files" in r) is (enabled or role is analysis) for r in reads)
    if enabled:
        for role in (main, analysis):
            assert _tool_results(role["requests"], "read_skill")[0]["skill_id"] == "public-fixture"
    packets = [json.loads(next(m["content"] for m in r["messages"] if m["type"] == "human"))
               for r in analysis["requests"]]
    follow_up = packets[-1]
    context = follow_up["context"]
    assert isinstance(context, str) is enabled
    if enabled:
        context = json.loads(context)
    assert context["previous_code"] == current["observations"]["executions"][:2][-(2 if enabled else 1):]
    assert ("position" in follow_up["profile"]["columns"][0]) is enabled


def test_bypass_telemetry_is_retained(captured):
    enabled, current = captured
    observed = current["observations"]
    expected_policy = "standard_skills_on" if enabled else "compact_skills_off"
    assert observed["policy"] == [expected_policy, expected_policy]
    for role in ("main", "analysis"):
        usage = [entry for task in observed["usage"] for entry in task[role]]
        assert len(usage) == len(current["payload"][role]["requests"])
        assert all("context_bytes" in entry and "cache_usage" in entry for entry in usage)
        assert all(entry["role"] == role for entry in usage)


def test_complete_payload_matches_local_frozen_implementation(captured, tmp_path):
    enabled, current = captured
    source = ROOT / ".cache/evaluation" / REFERENCES[enabled]
    if not source.is_dir():
        pytest.skip("Optional local snapshot absent; committed full-payload hashes still run")
    reference = isolated_capture(source, tmp_path, enabled)
    assert current["payload"] == reference["payload"]


def test_normalizer_preserves_json_text_and_identifier_aliases():
    first, second = "1" * 32, "2" * 32
    packet = {"content": '{"second": 2, "first": 1}', "other": '{"first":1,"second":2}',
              "id": first, "refs": [first, second, first], "sha256": "a" * 64}
    normalized = normalize_payload(packet)
    assert normalized["content"] == packet["content"]
    assert normalized["other"] == packet["other"]
    assert normalized["id"] == normalized["refs"][0] == normalized["refs"][2]
    assert normalized["refs"][1] != normalized["id"]
    assert normalized["sha256"] == packet["sha256"]


def write_reference_hashes(directory):
    """Explicit maintainer operation requiring genuine frozen source directories."""
    fixture = {
        "recipe": RECIPE,
        "scope": "Exact messages and bind_tools schemas for this finite public recipe only.",
        "normalization": "UUID identities in occurrence order; fixed CSV basename and execution IDs; JSON text unchanged.",
        "references": {},
    }
    for enabled, relative in REFERENCES.items():
        source = ROOT / ".cache/evaluation" / relative
        assert source.is_dir(), f"Cannot regenerate references without {source}"
        capture = isolated_capture(source, directory / str(enabled), enabled)
        fixture["references"]["skills_on" if enabled else "skills_off"] = {
            "source": relative,
            "implementation_sha256": {
                name: hashlib.sha256((source / "labweaver" / name).read_bytes()).hexdigest()
                for name in ("agent.py", "analysis.py")
            },
            "sha256": fingerprints(capture["payload"]),
        }
    HASH_FIXTURE.parent.mkdir(exist_ok=True)
    HASH_FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--write-reference-hashes"]
    write_reference_hashes(ROOT / ".cache" / "public-payload-reference-recipe")
