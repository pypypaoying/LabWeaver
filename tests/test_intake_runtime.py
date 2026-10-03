"""Shared entrypoint checks for credential boundaries and early task errors."""

from dataclasses import replace
import json
import socket

import pytest

import labweaver.config as model_configuration
import labweaver.runtime.intake as intake_runtime
from labweaver.config import ConfigurationError
from labweaver.offline import OfflineIntakeModel
from labweaver.run_config import IntakeConfig


@pytest.fixture(autouse=True)
def isolated_no_network_environment(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def deny_network(*args, **kwargs):
        raise AssertionError("Shared runtime tests must not access the network.")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)


def _intake_config(tmp_path):
    source = tmp_path / "measurements.csv"
    source.write_text("condition,value\nA,2\nB,4\n", encoding="utf-8")
    return IntakeConfig(
        csv_path=source,
        task="先理解测量记录并确认分析方向",
        task_file=None,
        mode="live",
        env_file=None,
        encoding="utf-8-sig",
        delimiter=",",
        sample_rows=5,
        output_dir=tmp_path / "runs",
    )


def _process_model_settings(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "process-test-placeholder")
    monkeypatch.setenv("LLM_BASE_URL", "https://process.example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_ID", "process-test-model")


def test_none_env_file_ignores_unrelated_cwd_dotenv_and_accepts_process_settings(tmp_path, monkeypatch):
    config = _intake_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=unrelated-file-test-secret\n"
        "LLM_BASE_URL=https://unrelated.example.invalid/v1\n"
        "LLM_MODEL_ID=unrelated-model\n",
        encoding="utf-8",
    )

    def forbidden_dotenv_read(*args, **kwargs):
        raise AssertionError("An env_file=None intake must not read cwd/.env.")

    monkeypatch.setattr(model_configuration, "dotenv_values", forbidden_dotenv_read)
    received_settings = []

    def offline_model_for_live_settings(settings):
        received_settings.append(settings)
        assert settings.api_key == "process-test-placeholder"
        assert settings.base_url == "https://process.example.invalid/v1"
        assert settings.model_id == "process-test-model"
        return OfflineIntakeModel()

    monkeypatch.setattr(intake_runtime, "create_model", offline_model_for_live_settings)
    with pytest.raises(ConfigurationError, match="Missing model settings"):
        intake_runtime.execute_intake(config)
    assert received_settings == []
    assert not config.output_dir.exists()

    _process_model_settings(monkeypatch)
    report, saved = intake_runtime.execute_intake(config)
    assert len(received_settings) == 1
    assert report["status"] == "awaiting_confirmation"
    assert report["mode"] == "live"  # Live configuration path; only model creation is replaced.
    assert report["model_calls"] == 2
    assert len(report["execution_ledger"]) == 1
    assert report["profile"]["row_count"] == 2
    assert saved.parent == config.output_dir
    recorded = saved.read_text(encoding="utf-8")
    assert json.loads(recorded)["status"] == "awaiting_confirmation"
    assert "unrelated-file-test-secret" not in recorded
    assert "process-test-placeholder" not in recorded


def test_unexpected_model_initialization_error_keeps_only_safe_exception_type(tmp_path, monkeypatch):
    config = _intake_config(tmp_path)
    _process_model_settings(monkeypatch)

    def failed_model_creation(settings):
        raise ValueError(
            "API key=process-test-placeholder; credential body; https://private.example.invalid"
        )

    monkeypatch.setattr(intake_runtime, "create_model", failed_model_creation)
    with pytest.raises(ConfigurationError) as caught:
        intake_runtime.execute_intake(config)
    message = str(caught.value)
    assert "ValueError" in message
    assert "Model initialization failed" in message
    assert "process-test-placeholder" not in message
    assert "credential body" not in message
    assert "private.example.invalid" not in message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert not config.output_dir.exists()


def test_empty_task_file_is_rejected_before_loading_or_constructing_model(tmp_path, monkeypatch):
    task_file = tmp_path / "task.txt"
    task_file.write_text(" \n\t ", encoding="utf-8-sig")
    config = replace(_intake_config(tmp_path), task=None, task_file=task_file)
    invoked = []

    def forbidden_model_work(*args, **kwargs):
        invoked.append(True)
        raise AssertionError("An empty task must fail before any model configuration or construction.")

    monkeypatch.setattr(intake_runtime, "load_config", forbidden_model_work)
    monkeypatch.setattr(intake_runtime, "create_model", forbidden_model_work)
    with pytest.raises(ConfigurationError, match="task must not be empty"):
        intake_runtime.execute_intake(config)
    assert invoked == []
    assert not config.output_dir.exists()
