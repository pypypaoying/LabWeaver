"""Host-only application settings. No model credentials cross the HTTP boundary."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import secrets
from labweaver.config import ModelConfig


def default_data_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "LabWeaver"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "labweaver"


@dataclass(frozen=True)
class WebSettings:
    data_dir: Path
    token: str
    project_dir: Path
    mode: str = "live"
    env_file: Path | None = None
    image: str = "labweaver-python:0.3.0"
    execution_timeout: int = 60
    analysis_max_tokens: int = 12288
    skills_enabled: bool = False
    model_timeout: float = 180.0
    task_timeout: float = 900.0
    frontend_dir: Path | None = None
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    allowed_origins: tuple[str, ...] = ()
    cancel_grace: float = 5.0
    model_config: ModelConfig | None = field(default=None, repr=False)
    runtime_dir: Path | None = field(default=None, repr=False)

    @classmethod
    def from_project(cls, *, data_dir=None, mode=None, token=None, config=None):
        from labweaver.run_config import load_intake_config

        project = Path.cwd().resolve()
        intake = load_intake_config(config, overrides={"mode": mode} if mode else {})
        return cls(
            data_dir=Path(data_dir or default_data_dir()).resolve(),
            token=token or secrets.token_urlsafe(32), project_dir=project,
            mode=intake.mode, env_file=intake.env_file,
            image=intake.execution_image, execution_timeout=intake.execution_timeout,
            analysis_max_tokens=intake.analysis_max_tokens,
            skills_enabled=intake.skills_enabled,
            frontend_dir=project / "frontend/dist",
        )
