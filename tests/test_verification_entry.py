"""Check the one-click verification entry's safety and process-state cleanup."""

import importlib.util
import json
import os
from pathlib import Path
import socket

import pytest


def _load_runner():
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("labweaver_verification_under_test", project_root / "verify_d3.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_context_denies_connections_and_restores_cwd_and_environment_on_error(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "unusual-existing-value")
    monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
    before_environment = dict(os.environ)
    before_connections = (socket.create_connection, socket.socket.connect, socket.socket.connect_ex)
    with pytest.raises(ValueError, match="controlled failure"):
        with runner._offline_environment(runner.PROJECT_ROOT) as attempts:
            assert Path.cwd() == runner.PROJECT_ROOT
            assert all(os.environ[name] == "false" for name in
                       ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"))
            with pytest.raises(RuntimeError, match="network connection"):
                socket.create_connection(("example.invalid", 443))
            assert attempts == [True]
            raise ValueError("controlled failure")
    assert Path.cwd() == tmp_path
    assert dict(os.environ) == before_environment
    assert (socket.create_connection, socket.socket.connect, socket.socket.connect_ex) == before_connections


def test_main_suppresses_arbitrary_failure_text_and_restores_state(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    monkeypatch.chdir(tmp_path)
    before_cwd = Path.cwd()
    before_environment = dict(os.environ)

    def fail_to_read_resources(relative):
        raise ValueError("private-test-api-key-should-never-be-printed")

    monkeypatch.setattr(runner, "_project_path", fail_to_read_resources)
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert "ValueError" in captured.err
    assert "private-test-api-key" not in captured.out + captured.err
    assert "Passed: 0/12" in captured.out
    assert Path.cwd() == before_cwd
    assert dict(os.environ) == before_environment


def test_main_verifies_actual_statistics_resume_and_artifacts_without_credentials(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    import labweaver.config as configuration
    import labweaver.runtime.records as records

    def deny_credentials(*args, **kwargs):
        raise AssertionError("Offline acceptance must not load model credentials or create a live model.")

    monkeypatch.setattr(configuration, "load_config", deny_credentials)
    monkeypatch.setattr(configuration, "create_model", deny_credentials)
    actual_save = records.save_run

    def temporary_artifacts(report, output_dir, **kwargs):
        return actual_save(report, tmp_path / "artifacts", **kwargs)

    monkeypatch.setattr(records, "save_run", temporary_artifacts)
    monkeypatch.chdir(tmp_path)
    previous = (Path.cwd(), dict(os.environ), socket.create_connection, socket.socket.connect)
    assert runner.main() == 0
    output = capsys.readouterr()
    assert "Passed: 12/12" in output.out and output.err == ""
    assert previous == (Path.cwd(), dict(os.environ), socket.create_connection, socket.socket.connect)
    saved = [json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "artifacts").rglob("*.json")]
    assert sum(report["status"] == "awaiting_input" for report in saved) == 1
    tables = [report["analysis_results"][0] for report in saved if report.get("analysis_results")]
    assert sorted(table["rows"][0]["total_medals"] for table in tables) == [12, 22]
    assert len(list((tmp_path / "artifacts").rglob("*result-1.csv"))) == 2
