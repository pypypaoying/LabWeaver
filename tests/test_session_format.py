"""CSV format clarification resumes the actual Agent graph with one tool ID."""

import hashlib
import json
import socket

import pytest

from labweaver.agent import create_session
from labweaver.offline import OfflineIntakeModel


@pytest.fixture(autouse=True)
def no_network_or_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def deny(*args, **kwargs):
        raise AssertionError("Format clarification must run without a network connection.")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)


def _candidate_reply(report, key, value):
    choices = report["profile"]["error"]["candidates"]
    return str(next(i for i, choice in enumerate(choices, 1) if choice[key] == value))


def _assert_one_real_profile(report):
    assert report["tool_counts"]["profile_csv"] == 1
    assert report["tool_counts"]["ask_user"] == 0
    assert report["tool_attempts"] == 1
    ledger = report["execution_ledger"]
    assert len(ledger) == 1
    assert ledger[0]["name"] == "profile_csv"
    calls = [item for item in report["trace"] if item["kind"] == "tool_call"]
    results = [item for item in report["trace"] if item["kind"] == "tool_result"]
    assert len(calls) == len(results) == 1
    assert calls[0]["id"] == results[0]["tool_call_id"] == ledger[0]["tool_call_id"]
    assert json.loads(results[0]["content"]) == ledger[0]["result"]


def test_ambiguous_delimiter_resumes_matching_tool_without_extra_budget(tmp_path):
    path = tmp_path / "ambiguous.csv"
    raw = b"a,b;c\n1,2;3\n4,5;6\n"
    path.write_bytes(raw)
    model = OfflineIntakeModel()
    session = create_session(path, model)
    pending = session.invoke("概览这个 CSV")
    assert pending["status"] == "awaiting_input"
    assert pending["profile"]["error"]["code"] == "ambiguous_delimiter"
    assert pending["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert pending["model_calls"] == 1
    assert pending["execution_ledger"] == []
    assert pending["tool_counts"]["profile_csv"] == 1
    assert "候选序号" in pending["question"]
    call_id = pending["trace"][0]["id"]

    completed = session.resume(_candidate_reply(pending, "delimiter", ";"))
    assert completed["status"] == "completed"
    assert completed["task_id"] == pending["task_id"]
    assert completed["session_id"] == pending["session_id"]
    assert completed["model_calls"] == 2
    assert completed["parsing"]["delimiter"] == ";"
    assert completed["profile"]["sample_rows"] == [["1,2", "3"], ["4,5", "6"]]
    assert completed["source"] == pending["source"]
    assert completed["trace"][0]["id"] == call_id
    _assert_one_real_profile(completed)
    assert path.read_bytes() == raw


def test_ambiguous_encoding_choice_preserves_raw_bytes_and_reports_actual_codec(tmp_path):
    path = tmp_path / "legacy.csv"
    raw = b"name\n\xff\n"
    path.write_bytes(raw)
    session = create_session(path, OfflineIntakeModel())
    pending = session.invoke("给出数据概览")
    assert pending["status"] == "awaiting_input"
    assert pending["profile"]["error"]["code"] == "ambiguous_encoding"
    completed = session.resume(_candidate_reply(pending, "encoding", "cp1252"))
    assert completed["status"] == "completed"
    assert completed["parsing"]["encoding"] == "cp1252"
    assert completed["profile"]["sample_rows"] == [["ÿ"]]
    assert completed["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    _assert_one_real_profile(completed)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("raw,encoding,delimiter", [(b"a,b;c\n1,2;3\n", "utf-8", ";"),
                                                  (b"name\n\xff\n", "cp1252", ",")])
def test_advanced_explicit_format_skips_choice_dialogue(tmp_path, raw, encoding, delimiter):
    path = tmp_path / "explicit.csv"
    path.write_bytes(raw)
    session = create_session(path, OfflineIntakeModel(), encoding=encoding, delimiter=delimiter)
    completed = session.invoke("给出数据概览")
    assert completed["status"] == "completed"
    assert completed["question"] == ""
    assert completed["replies"] == []
    assert completed["parsing"]["encoding"] == encoding
    assert completed["parsing"]["delimiter"] == delimiter
    assert completed["parsing"]["encoding_method"] == "explicit"
    assert completed["parsing"]["delimiter_method"] == "explicit"
    _assert_one_real_profile(completed)
    assert path.read_bytes() == raw


def test_encoding_then_delimiter_choice_keeps_one_profile_call_and_same_task(tmp_path):
    path = tmp_path / "two-choices.csv"
    raw = b"a,b;c\n\xff,2;3\n"
    path.write_bytes(raw)
    session = create_session(path, OfflineIntakeModel())
    first = session.invoke("概览数据")
    assert first["status"] == "awaiting_input"
    assert first["profile"]["error"]["code"] == "ambiguous_encoding"
    second = session.resume(_candidate_reply(first, "encoding", "cp1252"))
    assert second["status"] == "awaiting_input"
    assert second["profile"]["error"]["code"] == "ambiguous_delimiter"
    assert second["model_calls"] == 1
    assert second["tool_attempts"] == 1
    completed = session.resume(_candidate_reply(second, "delimiter", ";"))
    assert completed["status"] == "completed"
    assert completed["task_id"] == first["task_id"] == second["task_id"]
    assert completed["profile"]["sample_rows"] == [["ÿ,2", "3"]]
    assert completed["model_calls"] == 2
    assert len(completed["replies"]) == 2
    _assert_one_real_profile(completed)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("reply", ["0", "-1", "999", "1.0", "invalid"])
def test_invalid_displayed_choice_is_controlled_failure(tmp_path, reply):
    path = tmp_path / "ambiguous.csv"
    raw = b"a,b;c\n1,2;3\n"
    path.write_bytes(raw)
    session = create_session(path, OfflineIntakeModel())
    pending = session.invoke("概览数据")
    assert pending["status"] == "awaiting_input"
    failed = session.resume(reply)
    assert failed["status"] == "error"
    assert failed["error"]["code"] == "profile_failed"
    assert failed["profile"]["error"]["code"] == "invalid_format_choice"
    assert failed["source"] == pending["source"]
    assert failed["model_calls"] == 1
    assert failed["analysis_results"] == []
    assert failed["tool_counts"]["profile_csv"] == 1
    assert len(failed["execution_ledger"]) == 1
    assert path.read_bytes() == raw


@pytest.mark.parametrize("changed", [b"new,data\nX,2\n", b"a,b;c\n9,8;7\n", b"\xef\xbb\xbfname\n\xff\n"])
def test_source_changed_while_waiting_cannot_silently_replace_snapshot(tmp_path, changed):
    path = tmp_path / "ambiguous.csv"
    raw = b"a,b;c\n1,2;3\n"
    path.write_bytes(raw)
    session = create_session(path, OfflineIntakeModel())
    pending = session.invoke("概览数据")
    assert pending["status"] == "awaiting_input"
    path.write_bytes(changed)
    failed = session.resume("1")
    assert failed["status"] == "error"
    assert failed["profile"]["error"]["code"] == "source_changed"
    assert failed["source"] == pending["source"]
    assert failed["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert failed["profile_completed"] is False
    assert failed["analysis_results"] == []
    assert path.read_bytes() == changed


def test_cancelled_format_dialogue_runs_no_profile_or_analysis(tmp_path):
    path = tmp_path / "ambiguous.csv"
    raw = b"a,b;c\n1,2;3\n"
    path.write_bytes(raw)
    session = create_session(path, OfflineIntakeModel())
    pending = session.invoke("概览数据")
    cancelled = session.cancel()
    assert pending["status"] == "awaiting_input"
    assert cancelled["status"] == "cancelled"
    assert cancelled["source"] == pending["source"]
    assert cancelled["execution_ledger"] == []
    assert cancelled["analysis_results"] == []
    with pytest.raises(ValueError):
        session.resume("1")
    assert path.read_bytes() == raw
