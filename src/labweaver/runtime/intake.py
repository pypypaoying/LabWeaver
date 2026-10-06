"""Shared intake execution for the CLI and the VS Code entry point."""

from pathlib import Path

from labweaver.agent import run_intake
from labweaver.config import ConfigurationError, create_model, load_config
from labweaver.run_config import IntakeConfig
from labweaver.runtime.records import save_run


def execute_intake(config: IntakeConfig) -> tuple[dict, Path]:
    """Resolve the task, run the selected model, and save its real evidence."""
    task = config.task
    if task is None and config.task_file is not None:
        task = config.task_file.read_text(encoding="utf-8-sig")
    if not task or not task.strip():
        raise ConfigurationError("The selected task must not be empty.")

    if config.mode == "offline":
        from labweaver.offline import OfflineIntakeModel

        model = OfflineIntakeModel()
    else:
        try:
            model = create_model(load_config(config.env_file, use_default_env=False))
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"Model initialization failed ({type(exc).__name__}); check model settings."
            ) from None

    report = run_intake(
        task,
        config.csv_path,
        model,
        encoding=config.encoding,
        delimiter=config.delimiter,
        sample_rows=config.sample_rows,
        material_paths=config.material_paths,
    )
    report["mode"] = config.mode
    with_brief = bool(config.material_paths) and (
        report.get("status") == "awaiting_confirmation"
        and report.get("materials_completed") is True
    )
    saved = save_run(report, config.output_dir, with_brief=with_brief)
    if with_brief:
        report["brief_path"] = str(saved.with_suffix(".md").resolve())
    return report, saved
