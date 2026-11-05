"""Compact analysis transport through the real child graph, without live models."""

import copy
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from labweaver.analysis import ANALYSIS_PROMPT
from labweaver.modeling import ANALYSIS_POLICY, artifact_contracts
from labweaver.offline import OfflineIntakeModel, demo_code
from labweaver.tools.csv_profile import load_csv_snapshot
from test_complex_analysis import (
    PLAN,
    ReceiptAfterRepairModel,
    code_call,
    no_network,
    runner_factory,
)


class ClarificationModel(OfflineIntakeModel):
    received: list = Field(default_factory=list)

    def _generate(self, messages, **kwargs):
        self.received.append(copy.deepcopy(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=json.dumps({"status": "needs_clarification", "question": "请确认新增口径"})
        ))])


def packet_from(model):
    return json.loads(next(m.content for m in model.received[0] if m.type == "human"))


def test_packet_keeps_one_contract_and_distinct_latest_replies(runner_factory):
    model = ClarificationModel()
    runner, _, calls, _ = runner_factory(model)
    original = "完整保留用户要求和约束。" * 100
    spec = {"goal": "分组汇总", "original_task": original, "confirmed_replies": ["按全量记录"]}
    runner.get_task_spec = lambda: spec
    context = {"task_spec": copy.deepcopy(spec), "original_task": original,
               "replies": ["按全量记录", "还需保留零值组"],
               "retrieved_chunks": [{"id": "chunk-1", "text": "资料口径"}],
               "history_context": "历史仅作背景"}
    profile = copy.deepcopy(runner.get_snapshot().profile)

    result = runner.delegate("Compute confirmed results", PLAN, "compact", context=json.dumps(context))

    assert result["status"] == "needs_clarification" and not calls
    packet = packet_from(model)
    assert packet["task_spec"] == spec
    assert isinstance(packet["context"], dict)
    assert packet["context"] == {k: v for k, v in context.items() if k not in {"task_spec", "original_task"}}
    assert packet["task_spec"]["confirmed_replies"] != packet["context"]["replies"]
    assert json.dumps(packet, ensure_ascii=False).count(original) == 1
    assert runner.get_snapshot().profile == profile
    assert runner.get_task_spec() == spec
    for mapping, summary, full in zip(packet["columns"], packet["profile"]["columns"], profile["columns"]):
        assert mapping == {"id": f"c{full['position']}", "name": full["name"]}
        assert summary == {"id": mapping["id"], **{k: v for k, v in full.items() if k not in {"position", "name"}}}
    assert {k: v for k, v in packet["profile"].items() if k != "columns"} == {k: v for k, v in profile.items() if k != "columns"}
    old_packet = {**packet, "profile": profile, "context": json.dumps(context, ensure_ascii=False)}
    assert len(json.dumps(packet, ensure_ascii=False)) < len(json.dumps(old_packet, ensure_ascii=False)) / 2


def test_equal_replies_are_sent_once_and_context_is_not_mutated(runner_factory):
    model = ClarificationModel()
    runner, _, _, _ = runner_factory(model)
    spec = {"goal": "compute", "confirmed_replies": ["保留全部类别"]}
    runner.get_task_spec = lambda: spec
    context = {"task_spec": spec, "original_task": "未冻结的原始要求", "replies": ["保留全部类别"]}
    original_context = copy.deepcopy(context)

    runner.delegate("Compute", PLAN, "same-replies", context=context)

    packet = packet_from(model)
    assert packet["context"] == {"original_task": "未冻结的原始要求"}
    assert packet["task_spec"]["confirmed_replies"] == ["保留全部类别"]
    assert context == original_context


@pytest.mark.parametrize("context", ["普通上下文保持原样", "{not JSON", '["ordinary", "JSON list"]', '"JSON scalar"'])
def test_plain_or_non_object_context_stays_compatible(runner_factory, context):
    model = ClarificationModel()
    runner, _, _, _ = runner_factory(model)
    runner.delegate("Compute", PLAN, "legacy-context", context=context)
    assert packet_from(model)["context"] == context


@pytest.mark.parametrize("modeling", [None, {
    "task_type": "classification", "target": "c2", "features": ["c1"],
    "split": "stratified", "seed": 42, "test_size": 0.2,
    "duplicate_policy": "keep", "prediction_delivery": "predictions",
    "metadata_delivery": "metadata", "metrics_delivery": "metrics",
}])
def test_modeling_policy_is_injected_only_for_modeling_contract(runner_factory, modeling):
    model = ClarificationModel()
    runner, _, calls, _ = runner_factory(model)
    spec = {"goal": "Compute", "modeling": modeling}
    runner.get_task_spec = lambda: spec

    runner.delegate("Compute", PLAN, "policy")

    prompt = next(m.content for m in model.received[0] if m.type == "system")
    assert prompt.startswith(ANALYSIS_PROMPT + (ANALYSIS_POLICY if modeling else ""))
    assert (ANALYSIS_POLICY in prompt) is bool(modeling)
    assert "scikit-learn" in prompt
    assert "emit_table" in prompt and "不能联网或安装依赖" in prompt
    assert packet_from(model)["artifact_contracts"] == artifact_contracts(spec)
    assert not calls


def test_compact_execution_receipts_keep_source_and_host_evidence(runner_factory):
    model = ReceiptAfterRepairModel(responses=[code_call()])
    runner, assets, calls, events = runner_factory(model)

    result = runner.delegate("Compute complete result", PLAN, "receipt")

    assert result["status"] == "completed" and len(calls) == 1
    aid = result["artifact_ids"][0]
    metadata = assets[aid]["metadata"]
    receipt = result["artifacts"][0]
    assert {key: receipt[key] for key in ("id", "kind", "deliverable_id", "label", "source")} == {
        key: metadata[key] for key in ("id", "kind", "deliverable_id", "label", "source")
    }
    assert receipt["rows"] == [{"group": "A", "value": "2"}, {"group": "B", "value": "4"}]
    assert not {"files", "execution_id", "image_id"}.intersection(receipt)
    assert metadata["files"]["table.csv"]["sha256"]
    assert metadata["execution_id"] == "execution-1"
    assert metadata["image_id"].startswith("sha256:")
    assert assets[aid]["files"]["table.csv"] == b"group,value\nA,2\nB,4\n"
    assert runner.runs[0]["result"]["artifacts"] == result["artifacts"]
    execution = next(json.loads(m.content) for m in model.received[-1]
                     if isinstance(m, ToolMessage) and m.name == "execute_python")
    assert execution["artifacts"] == result["artifacts"]
    event = next(e for e in events if e["type"] == "execution_end")
    assert event["record"]["code"] == calls[0]
    assert event["record"]["image_id"] == metadata["image_id"]


def test_later_delegation_keeps_real_failure_and_compact_provisional_receipts(runner_factory, monkeypatch):
    monkeypatch.setattr("labweaver.runtime.verification.verify_artifacts", lambda *args: {
        "passed": False, "results": [{"check": "fixture", "status": "failed", "evidence": {"actual": 2}}]
    })
    code = "print('complete failed script kept for repair')"
    runner, assets, _, _ = runner_factory(ReceiptAfterRepairModel(responses=[code_call(code)]))
    assert runner.delegate("Compute", PLAN, "failed")["status"] == "error"
    model = ClarificationModel()
    runner.model = model

    result = runner.delegate("Repair authorized outputs", PLAN, "repair")

    assert result["status"] == "needs_clarification"
    packet = packet_from(model)
    assert packet["previous_failed_attempt"]["code"] == code
    assert packet["previous_failed_attempt"]["verification"]["passed"] is False
    receipt = packet["provisional_artifacts"][0]
    assert receipt["id"] in assets
    assert receipt["source"] == runner.get_snapshot().source
    assert not {"files", "execution_id", "image_id"}.intersection(receipt)
    runner.new_task()
    assert runner._repair_context({"table"}) is None


def test_offline_demo_still_reads_compact_column_mapping(runner_factory, tmp_path):
    model = ClarificationModel()
    runner, _, _, _ = runner_factory(model)
    # The demo's deterministic department/score path consumes id/name mappings.
    source = tmp_path / "departments.csv"
    source.write_text("department,score\nA,2\nB,4\n", encoding="utf-8")
    snapshot = load_csv_snapshot(source)
    runner.get_snapshot = lambda: snapshot
    runner.delegate("Compute", PLAN, "offline-shape")
    code = demo_code(packet_from(model))
    assert "df['c2']" in code and "groupby('c1'" in code
