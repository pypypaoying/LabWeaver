"""Exercise Run Python File's config-driven entry point without credentials."""

import hashlib
import importlib.util
import json
from pathlib import Path
import socket


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_direct_runner():
    spec = importlib.util.spec_from_file_location(
        "labweaver_direct_runner_under_test", PROJECT_ROOT / "run_labweaver.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_runner_resolves_project_config_from_another_cwd_offline(
    tmp_path, monkeypatch, capsys,
):
    # Even a machine with tracing/model variables must not contact an API here.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)

    runner = _load_direct_runner()
    assert runner.CONFIG_PATH == PROJECT_ROOT / "labweaver.toml"
    output_dir = tmp_path / "saved-runs"
    monkeypatch.setattr(runner, "OVERRIDES", {
        "mode": "offline", "output_dir": str(output_dir),
    })
    unrelated_directory = tmp_path / "unrelated-cwd"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)

    import labweaver.config as config_module
    import labweaver.runtime.intake as intake_module

    credential_reads = []
    network_attempts = []

    def deny_model_config(*args, **kwargs):
        credential_reads.append(True)
        raise AssertionError("Offline intake attempted to read model credentials.")

    def deny_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError("The direct offline runner attempted a network connection.")

    # Cover both the defining module and runtime's already-imported reference.
    monkeypatch.setattr(config_module, "load_config", deny_model_config)
    monkeypatch.setattr(intake_module, "load_config", deny_model_config)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)

    csv_path = PROJECT_ROOT / "examples" / "data" / "survey.csv"
    source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert runner.main() == 0
    capsys.readouterr()

    records = list(output_dir.glob("*.json"))
    assert len(records) == 1
    report = json.loads(records[0].read_text(encoding="utf-8"))
    assert report["mode"] == "offline"
    assert report["status"] == "awaiting_confirmation"
    assert report["profile_completed"] is True
    assert report["profile"]["row_count"] == 8
    assert report["profile"]["column_count"] == 5
    assert report["model_calls"] == 2
    assert report["tool_attempts"] == 1
    assert len(report["execution_ledger"]) == 1
    assert report["source"] == {"name": "survey.csv", "sha256": source_hash}
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == source_hash
    assert report["run_id"] and report["recorded_at"]
    assert credential_reads == []
    assert network_attempts == []
    json.dumps(report, ensure_ascii=False, allow_nan=False)
