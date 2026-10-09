"""Packaged UI: only three inputs, resumable replies, follow-ups and cancellation."""

from unittest.mock import Mock
import pytest
from labweaver import app
from labweaver.run_config import load_intake_config


def record(status, tid="t1", question=""):
    return {
        "report_version": 2,
        "session_id": "s1",
        "task_id": tid,
        "task": "示例",
        "status": status,
        "question": question,
        "source": None,
        "artifacts": [],
    }


def test_entry_answers_and_follows_up(tmp_path, monkeypatch):
    import labweaver.runtime.intake as runtime

    path = tmp_path / "data.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    session = Mock()
    session.artifact_assets = {}
    session.invoke.side_effect = [
        record("awaiting_input", question="范围？"),
        record("completed", "t2"),
    ]
    session.resume.return_value = record("completed")
    monkeypatch.setattr(runtime, "create_runtime_session", lambda config: session)
    monkeypatch.setattr(
        app, "OVERRIDES", {"mode": "offline", "output_dir": str(tmp_path / "runs")}
    )
    replies = iter([str(path), "", "统计数据", "全部", "改为今年", ""])
    prompts = []

    def input_value(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr("builtins.input", input_value)
    assert app.main(interactive=True) == 0
    session.resume.assert_called_once()
    assert session.resume.call_args.args == ("全部",)
    assert [c.args[0] for c in session.invoke.call_args_list] == [
        "统计数据",
        "改为今年",
    ]
    assert len(prompts) == 6 and not any("分隔符" in p or "编码" in p for p in prompts)


@pytest.mark.parametrize("reply", ["退出", "quit"])
def test_pending_exit_cancels(tmp_path, monkeypatch, reply):
    import labweaver.runtime.intake as runtime

    session = Mock()
    session.artifact_assets = {}
    session.invoke.return_value = record("awaiting_input", question="范围？")
    session.cancel.return_value = record("cancelled")
    monkeypatch.setattr(runtime, "create_runtime_session", lambda config: session)
    monkeypatch.setattr("builtins.input", lambda prompt: reply)
    config = load_intake_config(
        overrides={"mode": "offline", "output_dir": str(tmp_path), "task": "统计数据"}
    )
    assert app._conversation(config) == 0
    session.cancel.assert_called_once()
    session.resume.assert_not_called()
