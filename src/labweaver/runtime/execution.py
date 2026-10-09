"""Fresh Docker containers only. Never executes model-generated code on the host."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from labweaver.runtime.artifacts import strict_json, validate_payload


@dataclass(frozen=True)
class ExecutionConfig:
    image: str = "labweaver-python:0.2.0"
    timeout: float = 60
    memory: str = "1g"
    cpus: float = 2
    workspace_mib: int = 128

    def __post_init__(self):
        if (
            not isinstance(self.image, str)
            or not self.image
            or any(c.isspace() for c in self.image)
        ):
            raise ValueError("Invalid execution image")
        if (
            not 0 < self.timeout <= 120
            or not 0 < self.cpus <= 2
            or self.memory != "1g"
            or self.workspace_mib != 128
        ):
            raise ValueError(
                "Execution limits may not exceed the supported sandbox policy"
            )


class ExecutionError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def docker_executable():
    found = shutil.which("docker")
    if found:
        return found
    for base in [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Docker/Docker",
        Path.home() / "AppData/Local/Programs/DockerDesktop",
    ]:
        candidate = base / "resources/bin/docker.exe"
        if candidate.is_file():
            return str(candidate)
    raise ExecutionError(
        "docker_unavailable",
        "Docker is required. Install/start Docker Desktop Linux containers and build sandbox/Dockerfile; host Python fallback is disabled.",
    )


def snapshot_payload(snapshot):
    return {
        "source": snapshot.source,
        "columns": [
            {"id": f"c{i}", "position": i, "name": name}
            for i, name in enumerate(snapshot.headers, 1)
        ],
        "rows": snapshot.rows,
    }


class DockerExecutor:
    def __init__(self, config=None):
        self.config = config or ExecutionConfig()
        self._lock = threading.RLock()
        self._container = None
        self._cancelled = threading.Event()
        self.image_id = None
        self._endpoint = None

    def _docker_cli(self, docker):
        return [docker] + (["--host", self._endpoint] if self._endpoint else [])

    def _client_env(self):
        # Pin the inspected daemon even if the default context changes mid-run.
        env = os.environ.copy()
        if self._endpoint:
            env.pop("DOCKER_CONTEXT", None)
            env.pop("DOCKER_HOST", None)
        return env

    def preflight(self):
        docker = docker_executable()
        try:
            selected_context = os.environ.get("DOCKER_CONTEXT")
            endpoint = None if selected_context else os.environ.get("DOCKER_HOST")
            if not endpoint:
                context = subprocess.run(
                    [
                        docker,
                        "context",
                        "inspect",
                        *([selected_context] if selected_context else []),
                        "--format",
                        "{{.Endpoints.docker.Host}}",
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if context.returncode:
                    raise ExecutionError(
                        "docker_not_ready", "Cannot inspect the local Docker context"
                    )
                endpoint = context.stdout.decode().strip()
            if not endpoint.startswith(("unix://", "npipe://")):
                raise ExecutionError(
                    "remote_docker_forbidden",
                    "D5 requires local Docker; remote TCP/SSH contexts are not accepted",
                )
            self._endpoint = endpoint
            info = subprocess.run(
                self._docker_cli(docker) + ["info", "--format", "{{.OSType}}"],
                capture_output=True,
                timeout=15,
                env=self._client_env(),
            )
            if info.returncode or info.stdout.strip() != b"linux":
                raise ExecutionError(
                    "docker_not_ready",
                    "Start Docker Desktop with Linux containers; the daemon is unavailable or incompatible.",
                )
            image = subprocess.run(
                self._docker_cli(docker)
                + ["image", "inspect", self.config.image, "--format", "{{.Id}}"],
                capture_output=True,
                timeout=15,
                env=self._client_env(),
            )
            if image.returncode:
                raise ExecutionError(
                    "sandbox_image_missing",
                    "Build the locked sandbox image: docker build -t labweaver-python:0.2.0 sandbox",
                )
            image_id = image.stdout.decode().strip()
            if not image_id.startswith("sha256:") or len(image_id) != 71:
                raise ExecutionError(
                    "invalid_image_id",
                    "Docker returned an invalid immutable image identifier",
                )
            self.image_id = image_id
            return docker, image_id
        except (OSError, subprocess.TimeoutExpired):
            raise ExecutionError(
                "docker_not_ready", "Docker preflight failed or timed out"
            ) from None

    def reset(self):
        if self._container:
            raise ValueError("An execution is still active")
        self._cancelled.clear()

    def cancel(self):
        self._cancelled.set()
        with self._lock:
            name = self._container
        if name:
            self._remove(name)

    def _remove(self, name):
        try:
            docker = docker_executable()
            removed = subprocess.run(
                self._docker_cli(docker) + ["rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                env=self._client_env(),
            )
            if removed.returncode == 0:
                return True
            probe = subprocess.run(
                self._docker_cli(docker) + ["inspect", name],
                capture_output=True,
                timeout=10,
                env=self._client_env(),
            )
            return b"No such" in probe.stderr
        except (OSError, subprocess.TimeoutExpired, ExecutionError):
            return False

    def command(self, docker, image_id, name, input_dir):
        return self._docker_cli(docker) + [
            "run",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--pids-limit",
            "64",
            "--cpus",
            str(self.config.cpus),
            "--memory",
            self.config.memory,
            "--memory-swap",
            self.config.memory,
            "--init",
            "--tmpfs",
            f"/work:rw,nosuid,nodev,size={self.config.workspace_mib}m,uid=65532,gid=65532,mode=700",
            "--mount",
            f"type=bind,source={input_dir},target=/input,readonly",
            "--entrypoint",
            "python",
            # Docker client config can automatically inject credential-bearing
            # proxy variables. Explicit empty overrides prevent that transfer.
            *[
                item
                for name in (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "FTP_PROXY",
                    "ALL_PROXY",
                    "NO_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "ftp_proxy",
                    "all_proxy",
                    "no_proxy",
                )
                for item in ("--env", name + "=")
            ],
            image_id,
            "/opt/labweaver/runner.py",
        ]

    def execute(
        self,
        code,
        snapshot,
        deliverables,
        *,
        remaining_bytes=50 * 1024 * 1024,
        remaining_figures=2,
    ):
        start = time.monotonic()
        invalid_code = not isinstance(code, str)
        code = code if isinstance(code, str) else ""
        record = {
            "id": "execution-" + uuid.uuid4().hex,
            "status": "error",
            "code": code,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "source": snapshot.source.copy(),
            "image": self.config.image,
            "image_id": None,
            "exit_code": None,
            "artifact_ids": [],
            "limits": {
                "seconds": self.config.timeout,
                "cpus": self.config.cpus,
                "memory": self.config.memory,
                "workspace_mib": self.config.workspace_mib,
            },
        }
        assets, name = {}, "labweaver-" + uuid.uuid4().hex
        try:
            if invalid_code or not code.strip() or len(code.encode()) > 128 * 1024:
                raise ExecutionError("invalid_code", "Script must contain 1..128 KiB")
            if self._cancelled.is_set():
                raise ExecutionError("cancelled", "Execution cancelled")
            docker, image_id = self.preflight()
            record["image_id"] = image_id
            with tempfile.TemporaryDirectory(prefix="labweaver-input-") as folder:
                input_dir = Path(folder)
                input_dir.chmod(0o755)
                (input_dir / "dataset.json").write_text(
                    json.dumps(
                        snapshot_payload(snapshot), ensure_ascii=False, allow_nan=False
                    ),
                    encoding="utf-8",
                )
                (input_dir / "analysis.py").write_text(code, encoding="utf-8")
                # Both mounted files contain selected data/code only, never environment variables.
                with self._lock:
                    self._container = name
                process = subprocess.Popen(
                    self.command(docker, image_id, name, input_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._client_env(),
                )
                buffers = {"stdout": bytearray(), "stderr": bytearray()}
                overflow = threading.Event()

                def drain(pipe, key, cap):
                    while chunk := pipe.read(64 * 1024):
                        if len(buffers[key]) + len(chunk) <= cap:
                            buffers[key].extend(chunk)
                        else:
                            overflow.set()
                    pipe.close()

                readers = [
                    threading.Thread(
                        target=drain,
                        args=(process.stdout, "stdout", 72 * 1024 * 1024),
                        daemon=True,
                    ),
                    threading.Thread(
                        target=drain,
                        args=(process.stderr, "stderr", 32 * 1024),
                        daemon=True,
                    ),
                ]
                for reader in readers:
                    reader.start()
                reason = None
                deadline = time.monotonic() + self.config.timeout
                try:
                    while process.poll() is None:
                        if (
                            self._cancelled.is_set()
                            or overflow.is_set()
                            or time.monotonic() >= deadline
                        ):
                            reason = (
                                "cancelled"
                                if self._cancelled.is_set()
                                else "output_limit"
                                if overflow.is_set()
                                else "execution_timeout"
                            )
                            self._remove(name)
                            process.kill()
                            break
                        time.sleep(0.02)
                    process.wait(timeout=10)
                    for reader in readers:
                        reader.join(timeout=10)
                    state = subprocess.run(
                        self._docker_cli(docker)
                        + ["inspect", name, "--format", "{{json .State}}"],
                        capture_output=True,
                        timeout=10,
                        env=self._client_env(),
                    )
                    details = strict_json(state.stdout) if state.returncode == 0 else {}
                    record["container_state"] = {
                        k: details.get(k) for k in ("ExitCode", "OOMKilled", "Status")
                    }
                    if reason or details.get("OOMKilled"):
                        raise ExecutionError(
                            reason or "memory_limit",
                            "Container stopped by its resource limit or cancellation",
                        )
                    if process.returncode != 0:
                        record["stderr"] = {
                            "text": buffers["stderr"].decode("utf-8", errors="replace"),
                            "truncated": overflow.is_set(),
                        }
                        record["exit_code"] = process.returncode
                        raise ExecutionError(
                            "container_failed",
                            "Container did not produce a successful protocol response",
                        )
                    payload = strict_json(buffers["stdout"])
                    if payload.get("protocol") != 1:
                        raise ExecutionError(
                            "invalid_protocol", "Invalid sandbox protocol version"
                        )
                    record.update(
                        {k: payload[k] for k in ("exit_code", "stdout", "stderr")}
                    )
                    if payload["exit_code"] != 0:
                        raise ExecutionError(
                            "python_failed",
                            "Python failed; inspect bounded stderr before revising the full script",
                        )
                    if payload.get("protocol_error"):
                        raise ExecutionError(
                            "artifact_rejected", payload["protocol_error"]
                        )
                    assets = validate_payload(
                        payload["artifacts"],
                        deliverables=deliverables,
                        source=snapshot.source,
                        execution_id=record["id"],
                        image_id=image_id,
                        remaining_bytes=remaining_bytes,
                        remaining_figures=remaining_figures,
                    )
                    record.update(status="completed", artifact_ids=list(assets))
                finally:
                    if process.poll() is None:
                        self._remove(name)
                        process.kill()
                        process.wait(timeout=10)
                    removed = self._remove(name)
                    record["container_removed"] = removed
                    if not removed and not self._cancelled.is_set():
                        raise ExecutionError(
                            "container_cleanup_failed",
                            "Container cleanup could not be confirmed",
                        )
        except ExecutionError as exc:
            record["error"] = {"code": exc.code, "message": str(exc)}
            record["status"] = "error"
            assets = {}
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            subprocess.TimeoutExpired,
        ) as exc:
            record["error"] = {
                "code": "execution_protocol_failed",
                "message": type(exc).__name__,
            }
            record["status"] = "error"
            assets = {}
        finally:
            with self._lock:
                self._container = None
            record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        return record, assets
