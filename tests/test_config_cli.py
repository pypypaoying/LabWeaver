import hashlib
import json
from pathlib import Path

import pytest

from labweaver.cli import main
from labweaver.config import ConfigurationError, load_config
from labweaver.runtime.records import save_run


@pytest.fixture(autouse=True)
def isolated_model_environment(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


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


def test_profile_cli_runs_without_api_and_keeps_source(tmp_path, capsys):
    csv = tmp_path / "table.csv"
    csv.write_text("group,value\nA,2\nB,4\n", encoding="utf-8")
    original = hashlib.sha256(csv.read_bytes()).hexdigest()
    runs = tmp_path / "runs"
    assert main(["profile", "--csv", str(csv), "--output-dir", str(runs)]) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["row_count"] == 2
    assert report["column_count"] == 2
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == original
    assert len(list(runs.glob("*.json"))) == 1


def test_profile_cli_reports_read_failure(tmp_path, capsys):
    assert main(["profile", "--csv", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path / "runs")]) == 1
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
    assert report["status"] == "awaiting_confirmation"
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
        "profile", "--csv", str(csv), "--encoding", "gb18030", "--delimiter", ";",
        "--output-dir", str(tmp_path / "runs"),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["columns"][0]["name"] == "组别"
    assert report["row_count"] == 2

