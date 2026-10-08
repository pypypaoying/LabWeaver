import hashlib
import json

import pytest

from labweaver.cli import main
from labweaver.config import ConfigurationError, load_config
from labweaver.runtime.records import save_run


@pytest.fixture(autouse=True)
def isolated_model_environment(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def test_config_requires_explicit_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError, match="LLM_MODEL_ID"):
        load_config()


def test_config_reads_selected_file_without_mutating_environment(tmp_path, monkeypatch):
    env = tmp_path / "settings.env"
    env.write_text(
        "LLM_API_KEY=local-test-placeholder\n"
        "LLM_BASE_URL=https://example.invalid/v1\n"
        "LLM_MODEL_ID=explicit-model\nLLM_TIMEOUT=20\n",
        encoding="utf-8",
    )
    config = load_config(env)
    assert config.model_id == "explicit-model"
    assert config.timeout == 20
    assert "local-test-placeholder" not in repr(config)
    import os

    assert "LLM_API_KEY" not in os.environ
    monkeypatch.setenv("LLM_MODEL_ID", "environment-model")
    assert load_config(env).model_id == "environment-model"


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "121", "invalid"])
def test_invalid_timeout_does_not_expose_key(tmp_path, timeout):
    env = tmp_path / "settings.env"
    env.write_text(
        "LLM_API_KEY=local-test-placeholder\nLLM_BASE_URL=https://example.invalid/v1\n"
        f"LLM_MODEL_ID=explicit-model\nLLM_TIMEOUT={timeout}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError) as caught:
        load_config(env)
    assert "local-test-placeholder" not in str(caught.value)


def test_run_artifacts_are_strict_json_and_do_not_replace(tmp_path):
    first = save_run({"status": "completed", "text": "中文"}, tmp_path)
    second = save_run({"status": "completed"}, tmp_path)
    assert first != second
    document = json.loads(first.read_text(encoding="utf-8"))
    assert document["text"] == "中文"
    assert document["run_id"]
    with pytest.raises(ValueError):
        save_run({"value": float("nan")}, tmp_path)


def test_intake_cli_runs_offline_and_keeps_source(tmp_path, capsys):
    csv = tmp_path / "table.csv"
    csv.write_text("group,value\nA,2\nB,4\n", encoding="utf-8")
    original = hashlib.sha256(csv.read_bytes()).hexdigest()
    runs = tmp_path / "runs"
    assert main(["intake", "--csv", str(csv), "--task", "检查数据并确认分析方向", "--offline",
                 "--output-dir", str(runs)]) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["status"] == "completed"
    assert report["profile"]["row_count"] == 2
    assert report["profile"]["column_count"] == 2
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == original
    assert len(list(runs.rglob("*.json"))) == 1


def test_intake_cli_reports_csv_read_failure(tmp_path, capsys):
    assert main(["intake", "--csv", str(tmp_path / "missing.csv"), "--task", "检查数据", "--offline",
                 "--output-dir", str(tmp_path / "runs")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"


def test_offline_intake_cli_has_no_credentials(tmp_path, capsys):
    csv = tmp_path / "measurements.csv"
    csv.write_text("condition,measurement\nA,2\nB,4\n", encoding="utf-8")
    task = tmp_path / "task.txt"
    task.write_text("比较不同条件的测量结果", encoding="utf-8")
    assert main([
        "intake", "--csv", str(csv), "--task-file", str(task), "--offline",
        "--output-dir", str(tmp_path / "runs"),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "completed"
    assert report["mode"] == "offline"
    assert report["profile_completed"] is True
    assert "condition" in report["final_answer"]


def test_live_mode_configuration_error_is_clear(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["intake", "--csv", "table.csv", "--task", "Analyze", "--live"]) == 2
    output = capsys.readouterr()
    assert "Missing model settings" in output.err
    assert not list(tmp_path.glob("runs/*.json"))


def test_encoding_and_delimiter_cli(tmp_path, capsys):
    csv = tmp_path / "chinese.csv"
    csv.write_bytes("组别;数值\n甲;2\n乙;4\n".encode("gb18030"))
    assert main([
        "intake", "--csv", str(csv), "--task", "比较组别", "--offline",
        "--encoding", "gb18030", "--delimiter", ";",
        "--output-dir", str(tmp_path / "runs"),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["profile"]["columns"][0]["name"] == "组别"
    assert report["profile"]["row_count"] == 2


def test_removed_profile_command_is_rejected(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["profile"])
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_version_stays_available(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() == "LabWeaver 0.1.0"


def test_cli_can_run_intake_with_only_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv").write_text("组别,得分\n甲,2\n乙,4\n", encoding="utf-8")
    (tmp_path / "labweaver.toml").write_text(
        '[intake]\ncsv = "data.csv"\ntask = "先确认问卷分析方向"\nmode = "offline"\n',
        encoding="utf-8",
    )
    assert main(["intake"]) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["status"] == "completed"
    assert report["task"] == "先确认问卷分析方向"
    assert report["profile"]["columns"][0]["name"] == "组别"
    assert "Run report saved:" in output.err
    assert len(list((tmp_path / "runs").rglob("*.json"))) == 1


def test_explicit_missing_intake_config_is_clear(tmp_path, capsys):
    assert main(["intake", "--config", str(tmp_path / "absent.toml"), "--offline"]) == 2
    assert "configuration file does not exist" in capsys.readouterr().err


def test_cli_missing_task_file_is_clear(tmp_path, capsys):
    csv = tmp_path / "table.csv"
    csv.write_text("value\n1\n", encoding="utf-8")
    assert main(["intake", "--csv", str(csv), "--task-file", str(tmp_path / "absent.txt"),
                 "--offline", "--output-dir", str(tmp_path / "runs")]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert not list(tmp_path.glob("runs/*.json"))


def test_cli_overrides_config_and_paths_follow_cwd(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    (config_dir / "labweaver.toml").write_text(
        '[intake]\ncsv = "missing.csv"\ntask_file = "missing-task.txt"\nmode = "live"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "selected.csv").write_text("condition,value\nA,2\n", encoding="utf-8")
    assert main(["intake", "--config", "project/labweaver.toml", "--csv", "selected.csv",
                 "--task", "覆盖文件任务", "--offline", "--output-dir", "results"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task"] == "覆盖文件任务"
    assert report["source"]["name"] == "selected.csv"
    assert len(list((tmp_path / "results").rglob("*.json"))) == 1
    assert not (config_dir / "results").exists()


def test_cli_success_requires_completed(tmp_path, monkeypatch, capsys):
    import labweaver.runtime.intake as intake_runtime

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(intake_runtime, "execute_intake", lambda config: (
        {"status": "completed"}, tmp_path / "record.json",
    ))
    assert main(["intake", "--offline"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


@pytest.mark.parametrize("arguments", [
    ["--task", "one", "--task-file", "two.txt"],
    ["--offline", "--live"],
])
def test_cli_conflicting_task_or_mode_is_rejected(arguments):
    with pytest.raises(SystemExit) as caught:
        main(["intake", *arguments])
    assert caught.value.code == 2


def test_cli_negative_sample_count_is_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["intake", "--offline", "--sample-rows", "-1"]) == 2
    assert "nonnegative integer" in capsys.readouterr().err

