"""Incremental public answers, tool evidence, and resumable graph streaming."""

import json
import socket
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from pydantic import PrivateAttr

from labweaver.agent import create_session
from labweaver.offline import OfflineIntakeModel, OfflineVisualizationModel
from labweaver.runtime.streaming import ConsoleStream, _answer_prefix
from test_agent import ScriptedModel, paired


@pytest.mark.parametrize("text", ['中文\n换行与"引号"和\\路径 😀', 'nested {"answer":"not a key"}', 'a\tb'])
@pytest.mark.parametrize("ascii", [False, True])
def test_decode_every_partial_json_boundary(text, ascii):
    raw = json.dumps({"task_kind": "summary", "nested": {"answer": "ignore"}, "answer": text}, ensure_ascii=ascii)
    previous = ""
    for i in range(len(raw) + 1):
        actual = _answer_prefix(raw[:i])
        assert actual.startswith(previous) and text.startswith(actual)
        previous = actual
    assert previous == text


class StreamingModel(ScriptedModel):
    _events: list = PrivateAttr(default_factory=list)
    _delivered: threading.Event = PrivateAttr(default_factory=threading.Event)
    invalid: bool = False

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        if self._cursor == 0:
            self._cursor += 1
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": "profile_csv", "args": "{}", "id": "p1", "index": 0}]))
            return
        text = "逐步生成的中文回答。\n已读取实际数据。"
        raw = json.dumps({"task_kind": "summary", "retrieval_reason": "不需要检索", "answer": text}, ensure_ascii=True)
        if self.invalid:
            raw = raw[:-1]
        for index in range(0, len(raw), 7):
            yield ChatGenerationChunk(message=AIMessageChunk(content=raw[index:index + 7]))
            if index + 7 >= len(raw):
                assert self._delivered.wait(3), "Tokens were buffered until model completion"


def test_real_graph_streams_before_model_completion_without_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("No network allowed")
    monkeypatch.setattr(socket.socket, "connect", deny)
    model = StreamingModel()
    events = model._events
    session = create_session("examples/data/survey.csv", model)
    def emit(event):
        events.append(event)
        if event["type"] == "answer_delta":
            model._delivered.set()
    report = session.invoke("概览", on_event=emit)
    assert report["status"] == "completed", report.get("error")
    deltas = [e["text"] for e in events if e["type"] == "answer_delta"]
    assert len(deltas) > 2
    assert "".join(deltas) == report["answer_text"]
    assert [e["type"] for e in events[:2]] == ["tool_start", "tool_end"]
    assert events[-1]["type"] == "validated" and events[-1]["status"] == "completed"
    paired(report)


def test_streamed_invalid_json_is_explicitly_rejected(capsys):
    model = StreamingModel(invalid=True)
    console = ConsoleStream()
    def emit(event):
        model._events.append(event)
        if event["type"] == "answer_delta":
            model._delivered.set()
        console(event)
    report = create_session("examples/data/survey.csv", model).invoke("概览", on_event=emit)
    assert report["status"] == "error" and report["error"]["code"] == "invalid_final_answer"
    assert not console.validated_answer
    assert "不作为已完成结果" in capsys.readouterr().out


def test_streaming_interrupt_resume_followup_keeps_counts_and_exports(tmp_path):
    session = create_session("tests/fixtures/medals.csv", OfflineIntakeModel())
    events = []
    pending = session.invoke("统计出获得最多奖牌的前5个国家", on_event=events.append)
    assert pending["status"] == "awaiting_input", pending.get("error")
    report = session.resume("全部年份、按 Total 累计、保留原始 NOC", on_event=events.append)
    assert report["status"] == "completed", report.get("error")
    assert report["tool_counts"]["ask_user"] == report["tool_counts"]["profile_csv"] == 1
    paired(report)
    saved = session.save(tmp_path)
    assert Path(saved["result_csv_paths"][0]).is_file()
    followup = session.invoke("改为 2024 年", on_event=events.append)
    assert followup["status"] == "completed"
    assert followup["tool_counts"]["profile_csv"] == 0


def test_streaming_chart_does_not_expose_child_json():
    session = create_session("examples/data/survey.csv", OfflineIntakeModel(), visualization_model=OfflineVisualizationModel())
    events = []
    report = session.invoke("按 department 计算 satisfaction 均值，画柱状图", on_event=events.append)
    assert report["status"] == "completed", report.get("error")
    assert report["charts"]
    assert "".join(e["text"] for e in events if e["type"] == "answer_delta") == report["answer_text"]
    assert any(e["type"] == "tool_start" and e["name"] == "delegate_visualization" for e in events)


def test_terminal_previews_only_ten_rows_and_does_not_repeat_streamed_answer(capsys):
    from labweaver.app import _show_report
    from test_result_pages import table
    report = {"status": "completed", "answer_text": "回答已流式显示", "final_answer": "huge table", "analysis_results": [table(280)]}
    _show_report(report, Path("record.json"), streamed_answer=report["answer_text"])
    output = capsys.readouterr().out
    assert "回答已流式显示" not in output
    assert "280 项" in output and "预览前 10 项" in output
    assert output.count("category-") == 10
