"""Exercise the packaged VS Code conversation without live credentials."""

import csv
import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _runner():
    return importlib.import_module("labweaver.app")


def _completed(source, task_id, rows):
    return {
        "session_id": "session-test", "task_id": task_id, "status": "completed",
        "task": "国家奖牌前五", "source": source, "execution_ledger": [],
        "final_answer": "结果由实际分析工具生成。",
        "analysis_results": [{
            "status": "completed", "source": source, "columns": ["NOC", "medals"],
            "rows": rows, "spec": {"group_by": [2], "metrics": [{"op": "sum", "column": 6, "alias": "medals"}]},
        }],
    }


def test_interactive_entry_accepts_answers_and_followups_and_saves_real_tables(tmp_path, monkeypatch, capsys):
    import labweaver.runtime.intake as runtime

    data = tmp_path / "medals.csv"
    data.write_text("NOC,Total,Year\nA,5,2024\nB,3,2024\n", encoding="utf-8")
    source = {"name": data.name, "sha256": "a" * 64}
    pending = {
        "session_id": "session-test", "task_id": "task-one", "status": "awaiting_input",
        "task": "统计前五", "source": source, "question": "统计全部年份还是 2024 年？",
        "execution_ledger": [], "analysis_results": [],
    }
    all_years = _completed(source, "task-one", [{"NOC": "A", "medals": 12}, {"NOC": "B", "medals": 7}])
    year = _completed(source, "task-two", [{"NOC": "A", "medals": 5}, {"NOC": "B", "medals": 3}])
    session = Mock()
    session.chart_assets = {}
    session.invoke.side_effect = [pending, year]
    session.resume.return_value = all_years
    session_factory = Mock(return_value=session)
    monkeypatch.setattr(runtime, "create_runtime_session", session_factory)
    runner = _runner()
    monkeypatch.setattr(runner, "OVERRIDES", {"mode": "offline", "output_dir": str(tmp_path / "runs")})
    prompts = []
    replies = iter([str(data), "", "统计前五", "全部年份、按 Total 累计、保留原始 NOC", "改为 2024 年", ""])

    def selected_input(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr("builtins.input", selected_input)
    assert runner.main(interactive=True) == 0
    output = capsys.readouterr().out
    assert "2024 年" in output
    assert len(prompts) == 6
    assert all("编码" not in prompt and "分隔符" not in prompt for prompt in prompts)
    session.resume.assert_called_once_with("全部年份、按 Total 累计、保留原始 NOC")
    assert [call.args[0] for call in session.invoke.call_args_list] == ["统计前五", "改为 2024 年"]
    config = session_factory.call_args.args[0]
    assert config.csv_path == data and config.material_paths == ()
    assert config.encoding == config.delimiter == "auto"
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "runs").rglob("*.json")]
    assert sorted(report["status"] for report in reports) == ["awaiting_input", "completed", "completed"]
    assert len(list((tmp_path / "runs").rglob("*.md"))) == 2
    completed = [report for report in reports if report["status"] == "completed"]
    for report in completed:
        result_path = Path(report["result_csv_paths"][0])
        with result_path.open(encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
        expected = report["analysis_results"][0]["rows"]
        assert actual == [{key: str(value) for key, value in row.items()} for row in expected]


def test_pending_reply_exit_cancels_without_resuming(tmp_path, monkeypatch):
    import labweaver.runtime.intake as runtime
    from labweaver.run_config import load_intake_config

    config = load_intake_config(ROOT / "labweaver.toml", overrides={
        "mode": "offline", "task": "统计前五", "output_dir": str(tmp_path / "runs"),
    })
    session = Mock()
    session.chart_assets = {}
    session.invoke.return_value = {
        "status": "awaiting_input", "session_id": "s1", "task_id": "t1", "question": "哪些年份？",
    }
    session.cancel.return_value = {"status": "cancelled", "session_id": "s1", "task_id": "t1"}
    monkeypatch.setattr(runtime, "create_runtime_session", lambda selected: session)
    monkeypatch.setattr("builtins.input", lambda prompt: "退出")
    assert _runner()._conversation(config) == 0
    session.cancel.assert_called_once()
    session.resume.assert_not_called()
    assert len(list((tmp_path / "runs").rglob("*.json"))) == 2
    assert not list((tmp_path / "runs").rglob("*.md"))


def test_external_task_file_and_quoted_file_path_are_user_selected(tmp_path):
    runner = _runner()
    data = tmp_path / "real data.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    task = tmp_path / "task.txt"
    task.write_text("按全部年份统计奖牌前五", encoding="utf-8-sig")
    assert runner._selected_file('"' + str(data) + '"') == data
    assert runner._task_text("@" + str(task)) == "按全部年份统计奖牌前五"
    assert runner._task_text("") == runner.DEFAULT_TASK


@pytest.mark.parametrize("reply", ["@missing.txt"])
def test_missing_external_task_file_is_clear(reply):
    from labweaver.config import ConfigurationError

    with pytest.raises(ConfigurationError):
        _runner()._task_text(reply)
