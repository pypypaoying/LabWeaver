"""Shared model construction, task resolution, sessions, and artifact saving."""

import json
from pathlib import Path

from labweaver.agent import create_session, run_intake
from labweaver.config import ConfigurationError, create_model, load_config
from labweaver.run_config import IntakeConfig
from labweaver.runtime.records import save_run


def resolve_task(config: IntakeConfig) -> str:
    """Read an explicitly configured UTF-8 task before creating any model."""
    task = config.task
    if task is None and config.task_file is not None:
        with config.task_file.open("rb") as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ConfigurationError("The task file exceeds 64 KiB.")
        task = raw.decode("utf-8-sig", errors="strict")
    if not task or not task.strip():
        raise ConfigurationError("The selected task must not be empty.")
    return task.strip()


def _selected_model(config: IntakeConfig):
    if config.mode == "offline":
        from labweaver.offline import OfflineIntakeModel

        return OfflineIntakeModel()
    try:
        return create_model(load_config(config.env_file, use_default_env=False))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"Model initialization failed ({type(exc).__name__}); check model settings."
        ) from None


def create_runtime_session(config: IntakeConfig):
    """Bind a model and source once for the current interactive process."""
    return create_session(
        config.csv_path, _selected_model(config),
        material_paths=config.material_paths, encoding=config.encoding,
        delimiter=config.delimiter, sample_rows=config.sample_rows,
    )


def save_session_report(report: dict, config: IntakeConfig) -> tuple[dict, Path]:
    """Save a pending/error event or a completed brief and result tables."""
    report["mode"] = config.mode
    saved = save_run(report, config.output_dir, with_brief=report.get("status") == "completed")
    document = json.loads(saved.read_text(encoding="utf-8"))
    for key in ("brief_path", "result_csv_paths"):
        if key in document:
            report[key] = document[key]
    return report, saved


def execute_intake(config: IntakeConfig) -> tuple[dict, Path]:
    """Run one task without an input loop; pending questions remain recorded."""
    task = resolve_task(config)
    report = run_intake(
        task, config.csv_path, _selected_model(config),
        encoding=config.encoding, delimiter=config.delimiter,
        sample_rows=config.sample_rows, material_paths=config.material_paths,
    )
    return save_session_report(report, config)
