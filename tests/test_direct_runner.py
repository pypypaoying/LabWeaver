"""Exercise Run Python File's config-driven entry point without credentials."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_direct_runner():
    return importlib.import_module("labweaver.app")


@pytest.mark.parametrize(
    "material_override", [None, []], ids=["with-materials", "without-materials"]
)
def test_direct_runner_uses_working_directory_config_offline(
    tmp_path,
    monkeypatch,
    capsys,
    material_override,
):
    # Even a machine with tracing/model variables must not contact an API here.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)

    runner = _load_direct_runner()
    assert runner.CONFIG_PATH is None
    output_dir = tmp_path / "saved-runs"
    monkeypatch.setattr(
        runner,
        "OVERRIDES",
        {
            "mode": "offline",
            "output_dir": str(output_dir),
        },
    )
    monkeypatch.setattr(runner, "MATERIAL_PATHS", material_override)
    from labweaver.run_config import load_intake_config

    selected = load_intake_config(
        PROJECT_ROOT / "labweaver.toml", overrides={"mode": "offline"}
    )
    materials = selected.material_paths if material_override is None else ()
    material_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in materials
    }
    unrelated_directory = tmp_path / "unrelated-cwd"
    unrelated_directory.mkdir()
    (unrelated_directory / "labweaver.toml").write_text(
        '[intake]\ncsv = "'
        + (PROJECT_ROOT / "examples/data/survey.csv").as_posix()
        + '"\n'
        'task = "基于资料概述字段定义与数据规模"\n'
        "materials = ["
        + ", ".join('"' + path.as_posix() + '"' for path in selected.material_paths)
        + "]\n",
        encoding="utf-8",
    )
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
        raise AssertionError(
            "The direct offline runner attempted a network connection."
        )

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
    assert report["profile_result"]["status"] == "completed"
    assert report["profile_result"]["row_count"] == 8
    assert report["profile_result"]["column_count"] == 5
    assert report["model_calls"] == (3 if materials else 2)
    assert report["tool_attempts"] == (2 if materials else 1)
    assert (
        sum(item["name"] == "profile_csv" for item in report["execution_ledger"]) == 1
    )
    if materials:
        assert report["retrieval_status"] == "used"
        assert len(report["retrieval_queries"]) == 1
        assert (
            sum(
                item["name"] == "search_materials"
                for item in report["execution_ledger"]
            )
            == 1
        )
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
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in material_hashes.items()
    )
    assert report["record_path"] and report["report_version"] == 2
    assert credential_reads == []
    assert network_attempts == []
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_packaged_script_run_python_file_uses_cwd_configuration(tmp_path):
    source = tmp_path / "selected.csv"
    source.write_text("category,value\nA,2\nB,4\n", encoding="utf-8")
    (tmp_path / "labweaver.toml").write_text(
        '[intake]\ncsv = "selected.csv"\ntask = "总结行列和缺失情况"\nmode = "offline"\n',
        encoding="utf-8",
    )
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
    )
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src/labweaver/app.py")],
        cwd=tmp_path,
        input="\n\n\n\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "runs").rglob("*.json")
    ]
    assert len(reports) == 1
    assert reports[0]["source"]["name"] == "selected.csv"
    assert reports[0]["profile_result"]["row_count"] == 2
    assert reports[0]["analysis_status"] == "not_needed"


def test_f5_launches_packaged_conversation_in_project_directory():
    configuration = json.loads(
        (PROJECT_ROOT / ".vscode/launch.json").read_text(encoding="utf-8")
    )
    assert len(configuration["configurations"]) == 1
    launch = configuration["configurations"][0]
    assert launch["module"] == "labweaver.app"
    assert launch["cwd"] == "${workspaceFolder}"
    assert "program" not in launch
