"""Host-side container policy, bounded failures and immutable snapshots."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from labweaver.runtime.execution import (
    DockerExecutor,
    ExecutionConfig,
    ExecutionError,
    snapshot_payload,
)
from labweaver.tools.csv_profile import load_csv_snapshot


def test_container_command_contains_only_staged_input_and_resource_policy():
    executor = DockerExecutor()
    command = executor.command(
        "docker", "sha256:" + "a" * 64, "labweaver-test", "/selected-stage"
    )
    for option, value in [
        ("--network", "none"),
        ("--user", "65532:65532"),
        ("--cap-drop", "ALL"),
        ("--memory", "1g"),
        ("--cpus", "2"),
        ("--security-opt", "no-new-privileges"),
    ]:
        assert command[command.index(option) + 1] == value
    assert "--read-only" in command and "--init" in command
    assert "readonly" in command[command.index("--mount") + 1]
    assert "size=128m" in command[command.index("--tmpfs") + 1]
    assert sum(value == "--mount" for value in command) == 1
    assert not any("socket" in value or "API_KEY" in value for value in command)
    assert all(
        name + "=" in command
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
        )
    )


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 121},
        {"cpus": 4},
        {"memory": "8g"},
        {"workspace_mib": 500},
        {"image": "bad image"},
    ],
)
def test_execution_config_cannot_bypass_policy(options):
    with pytest.raises(ValueError):
        ExecutionConfig(**options)


def test_snapshot_preserves_duplicate_names_and_raw_cells(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text('id,value,value\n001,NA,NULL\n002,0," "\n', encoding="utf-8")
    snapshot = load_csv_snapshot(path)
    payload = snapshot_payload(snapshot)
    assert [c["id"] for c in payload["columns"]] == ["c1", "c2", "c3"]
    assert [c["name"] for c in payload["columns"]] == ["id", "value", "value"]
    assert payload["rows"][0] == ("001", "NA", "NULL")
    assert payload["rows"][1] == ("002", "0", " ")
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "code",
    [None, "", " " * 4, "x" * (128 * 1024 + 1)],
    ids=["wrong_type", "empty", "whitespace", "over_limit"],
)
def test_invalid_code_never_reaches_docker(code, monkeypatch):
    executor = DockerExecutor()
    monkeypatch.setattr(
        executor, "preflight", lambda: pytest.fail("invalid script reached Docker")
    )
    snapshot = SimpleNamespace(source={"name": "fixture", "sha256": "a" * 64})
    record, assets = executor.execute(code, snapshot, [])
    assert record["error"]["code"] == "invalid_code" and not assets


def test_missing_docker_is_clear_and_no_host_fallback(monkeypatch):
    import labweaver.runtime.execution as module

    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(ExecutionError, match="host Python fallback is disabled"):
        module.docker_executable()


def test_daemon_and_image_preflight_errors(monkeypatch):
    import labweaver.runtime.execution as module

    monkeypatch.setattr(module, "docker_executable", lambda: "docker")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"", stderr=b""),
    )
    with pytest.raises(ExecutionError) as exc:
        DockerExecutor().preflight()
    assert exc.value.code == "docker_not_ready"


def test_remote_docker_context_refused_before_sending_input(monkeypatch):
    import labweaver.runtime.execution as module

    monkeypatch.setattr(module, "docker_executable", lambda: "docker")
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.example:2375")
    with pytest.raises(ExecutionError) as exc:
        DockerExecutor().preflight()
    assert exc.value.code == "remote_docker_forbidden"


def test_docker_context_takes_precedence_over_host_environment(monkeypatch):
    import labweaver.runtime.execution as module

    monkeypatch.setattr(module, "docker_executable", lambda: "docker")
    monkeypatch.setenv("DOCKER_HOST", "unix:///local.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"ssh://remote.example", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ExecutionError) as exc:
        DockerExecutor().preflight()
    assert exc.value.code == "remote_docker_forbidden"
    assert len(calls) == 1 and "remote" in calls[0]


def test_verified_endpoint_is_pinned_and_context_environment_removed(monkeypatch):
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.example:2375")
    executor = DockerExecutor()
    executor._endpoint = "unix:///verified.sock"
    assert executor.command("docker", "sha256:" + "a" * 64, "task", "/stage")[:4] == [
        "docker",
        "--host",
        "unix:///verified.sock",
        "run",
    ]
    assert "DOCKER_CONTEXT" not in executor._client_env()
    assert "DOCKER_HOST" not in executor._client_env()
