"""Exercise Run Python File's config-driven entry point without credentials."""

import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_direct_runner():
    spec = importlib.util.spec_from_file_location(
        "labweaver_direct_runner_under_test", PROJECT_ROOT / "run_labweaver.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("material_override", [None, []], ids=["D3-default-materials", "D2-no-materials"])
def test_direct_runner_resolves_project_config_from_another_cwd_offline(
    tmp_path, monkeypatch, capsys, material_override,
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
    monkeypatch.setattr(runner, "MATERIAL_PATHS", material_override)
    from labweaver.run_config import load_intake_config

    selected = load_intake_config(runner.CONFIG_PATH, overrides={"mode": "offline"})
    materials = selected.material_paths if material_override is None else ()
    material_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in materials}
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

    records = list(output_dir.rglob("*.json"))
    assert len(records) == 1
    report = json.loads(records[0].read_text(encoding="utf-8"))
    assert report["mode"] == "offline"
    assert report["status"] == "completed"
    assert report["profile_completed"] is True
    assert report["profile"]["row_count"] == 8
    assert report["profile"]["column_count"] == 5
    assert report["model_calls"] == (3 if materials else 2)
    assert report["tool_attempts"] == (2 if materials else 1)
    assert sum(item["name"] == "profile_csv" for item in report["execution_ledger"]) == 1
    if materials:
        assert report["materials_completed"] is True
        assert len(report["retrieval_queries"]) == 1
        assert sum(item["name"] == "search_materials" for item in report["execution_ledger"]) == 1
        assert report["citations"]
        assert report["brief_path"] == str(records[0].with_suffix(".md"))
        brief = records[0].with_suffix(".md").read_text(encoding="utf-8")
        assert all(f"[{citation}]" in brief for citation in report["citations"])
    else:
        assert len(report["execution_ledger"]) == 1
        assert report.get("retrieval_queries", []) == []
        assert report["brief_path"] == str(records[0].with_suffix(".md"))
        assert records[0].with_suffix(".md").exists()
    assert report["source"] == {"name": "survey.csv", "sha256": source_hash}
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == source_hash
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in material_hashes.items())
    assert report["run_id"] and report["recorded_at"]
    assert credential_reads == []
    assert network_attempts == []
    json.dumps(report, ensure_ascii=False, allow_nan=False)
