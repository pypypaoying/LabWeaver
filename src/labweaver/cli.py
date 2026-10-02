"""Command-line entry points for direct profiling and evidence-based intake."""

import argparse
import json
from pathlib import Path
import sys

from labweaver.config import ConfigurationError, create_model, load_config
from labweaver.runtime.records import save_run
from labweaver.tools.csv_profile import profile_csv


def _delimiter(value: str) -> str:
    result = "\t" if value in {"\\t", "tab"} else value
    if len(result) != 1 or result in {"\n", "\r", '"'}:
        raise argparse.ArgumentTypeError("Use one delimiter character, or 'tab'.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LabWeaver data-project intake agent")
    parser.add_argument("--version", action="version", version="LabWeaver 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "intake"):
        command = commands.add_parser(name)
        command.add_argument("--csv", required=True, type=Path)
        command.add_argument("--encoding", default="utf-8-sig")
        command.add_argument("--delimiter", default=",", type=_delimiter)
        command.add_argument("--sample-rows", default=5, type=int)
        command.add_argument("--output-dir", default="runs", type=Path)
        if name == "intake":
            task = command.add_mutually_exclusive_group(required=True)
            task.add_argument("--task")
            task.add_argument("--task-file", type=Path)
            mode = command.add_mutually_exclusive_group(required=True)
            mode.add_argument("--offline", action="store_true")
            mode.add_argument("--live", action="store_true")
            command.add_argument("--env-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Python on Windows may otherwise encode redirected JSON using a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    options = {"encoding": args.encoding, "delimiter": args.delimiter, "sample_rows": args.sample_rows}
    try:
        if args.command == "profile":
            report = {"stage": "profile", **profile_csv(args.csv, **options)}
        else:
            from labweaver.agent import run_intake

            task = args.task if args.task is not None else args.task_file.read_text(encoding="utf-8-sig")
            if not task.strip():
                raise ConfigurationError("The task description must not be empty.")
            if args.offline:
                from labweaver.offline import OfflineIntakeModel

                model = OfflineIntakeModel()
            else:
                model = create_model(load_config(args.env_file))
            report = run_intake(task, args.csv, model, **options)
            report["mode"] = "offline" if args.offline else "live"
        saved = save_run(report, args.output_dir)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("Could not read the task file or write the local run report.", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Run report saved: {saved}", file=sys.stderr)
    return 0 if report.get("status") in {"completed", "awaiting_confirmation"} else 1

