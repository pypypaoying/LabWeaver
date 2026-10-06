"""Check the one-click verification entry's safety and process-state cleanup."""

import importlib.util
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
