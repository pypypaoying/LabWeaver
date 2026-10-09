"""Layer public, local, and command-line intake settings without model secrets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from labweaver.config import ConfigurationError


@dataclass(frozen=True)
class IntakeConfig:
    csv_path: Path
    task: str | None
    task_file: Path | None
    mode: str
    env_file: Path | None
    encoding: str
    delimiter: str
    sample_rows: int
    output_dir: Path
    material_paths: tuple[Path, ...] = ()
    execution_image: str = "labweaver-python:0.2.0"
    execution_timeout: int = 60
    analysis_max_tokens: int = 4096


_DEFAULTS = {
    "csv": "examples/data/survey.csv",
    "task_file": "examples/tasks/survey.txt",
    "mode": "live",
    "encoding": "auto",
    "delimiter": "auto",
    "sample_rows": 5,
    "output_dir": "runs",
    "materials": [],
    "execution_image": "labweaver-python:0.2.0",
    "execution_timeout": 60,
    "analysis_max_tokens": 4096,
}
_KEYS = frozenset(
    {
        "csv",
        "task",
        "task_file",
        "mode",
        "env_file",
        "encoding",
        "delimiter",
        "sample_rows",
        "output_dir",
        "materials",
        "execution_image",
        "execution_timeout",
        "analysis_max_tokens",
    }
)
_PATH_KEYS = frozenset({"csv", "task_file", "env_file", "output_dir"})


def _normalize_delimiter(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigurationError("Setting 'delimiter' must be a string.")
    if value == "auto":
        return value
    normalized = "\t" if value in {"tab", "\\t"} else value
    if len(normalized) != 1 or normalized in {"\n", "\r", "\x00", '"'}:
        raise ConfigurationError(
            "Setting 'delimiter' must be 'auto', one separator character, or 'tab'."
        )
    return normalized


def _resolve_path(value: Any, base: Path, key: str) -> Path:
    if not isinstance(value, (str, Path)) or (
        isinstance(value, str) and not value.strip()
    ):
        raise ConfigurationError(
            f"Setting '{key}' must be a nonempty file or directory path."
        )
    try:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()
    except (OSError, ValueError, RuntimeError):
        raise ConfigurationError(f"Setting '{key}' contains an invalid path.") from None


def _read_layer(path: Path, *, required: bool) -> dict:
    if not path.exists():
        if required:
            raise ConfigurationError(
                "The selected intake configuration file does not exist."
            )
        return {}
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError):
        raise ConfigurationError(
            "The intake configuration file could not be read as UTF-8."
        ) from None
    except tomllib.TOMLDecodeError:
        # TOML exceptions can include source text, which may contain secrets.
        raise ConfigurationError(
            "The intake configuration file contains invalid TOML."
        ) from None
    if set(document) - {"intake"}:
        raise ConfigurationError(
            "Only the '[intake]' section is supported in intake configuration files."
        )
    layer = document.get("intake", {})
    if not isinstance(layer, dict):
        raise ConfigurationError(
            "The intake configuration must contain an '[intake]' table."
        )
    return layer


def _apply_layer(settings: dict, layer: dict, *, base: Path) -> None:
    unknown = set(layer) - _KEYS
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ConfigurationError(f"Unknown intake setting(s): {names}.")
    if "task" in layer and "task_file" in layer:
        raise ConfigurationError(
            "Use either 'task' or 'task_file' within one configuration layer."
        )
    normalized = {}
    for key, value in layer.items():
        if key == "materials":
            if not isinstance(value, (list, tuple)):
                raise ConfigurationError(
                    "Setting 'materials' must be an array of file paths."
                )
            normalized[key] = tuple(_resolve_path(item, base, key) for item in value)
        elif key in _PATH_KEYS:
            normalized[key] = _resolve_path(value, base, key)
        elif key in {"sample_rows", "execution_timeout", "analysis_max_tokens"}:
            if type(value) is not int or value < 0:
                raise ConfigurationError(
                    "Setting 'sample_rows' must be a nonnegative integer."
                )
            normalized[key] = value
            if (
                key == "execution_timeout"
                and not 1 <= value <= 120
                or key == "analysis_max_tokens"
                and not 256 <= value <= 16384
            ):
                raise ConfigurationError(f"Setting '{key}' is out of range.")
        elif key == "delimiter":
            normalized[key] = _normalize_delimiter(value)
        elif key == "mode":
            if not isinstance(value, str) or value not in {"live", "offline"}:
                raise ConfigurationError("Setting 'mode' must be 'live' or 'offline'.")
            normalized[key] = value
        else:
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"Setting '{key}' must be a nonempty string.")
            normalized[key] = value
    if "task" in normalized:
        settings["task_file"] = None
    elif "task_file" in normalized:
        settings["task"] = None
    settings.update(normalized)


def load_intake_config(
    config_path: Path | None = None,
    *,
    overrides: dict | None = None,
) -> IntakeConfig:
    """Load defaults < selected public TOML < sibling local TOML < CLI overrides.

    Paths from defaults and TOML resolve relative to the selected configuration
    directory. Override paths resolve relative to the current working directory.
    Passing no config path permits a missing default ``labweaver.toml``; an
    explicitly selected file must exist. No credentials or task files are read.
    """
    cwd = Path.cwd()
    if config_path is None:
        public_path = cwd / "labweaver.toml"
    else:
        public_path = _resolve_path(config_path, cwd, "config")
    directory = public_path.parent
    settings: dict[str, Any] = {"task": None, "env_file": None}
    _apply_layer(settings, _DEFAULTS, base=directory)
    dotenv = directory / ".env"
    if dotenv.is_file():
        settings["env_file"] = dotenv.resolve()
    _apply_layer(
        settings,
        _read_layer(public_path, required=config_path is not None),
        base=directory,
    )
    local_path = directory / "labweaver.local.toml"
    if local_path != public_path:
        _apply_layer(settings, _read_layer(local_path, required=False), base=directory)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ConfigurationError(
                "Intake overrides must be a dictionary of settings."
            )
        _apply_layer(settings, overrides, base=cwd)
    return IntakeConfig(
        csv_path=settings["csv"],
        task=settings.get("task"),
        task_file=settings.get("task_file"),
        mode=settings["mode"],
        env_file=settings.get("env_file"),
        encoding=settings["encoding"],
        delimiter=settings["delimiter"],
        sample_rows=settings["sample_rows"],
        output_dir=settings["output_dir"],
        material_paths=settings["materials"],
        execution_image=settings["execution_image"],
        execution_timeout=settings["execution_timeout"],
        analysis_max_tokens=settings["analysis_max_tokens"],
    )
