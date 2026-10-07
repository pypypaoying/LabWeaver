"""Configuration-driven command-line entry point for evidence-based intake."""

import argparse
import json
from pathlib import Path
import sys

from labweaver.config import ConfigurationError
from labweaver.run_config import _normalize_delimiter, load_intake_config


def _delimiter(value: str) -> str:
    try:
        return _normalize_delimiter(value)
    except ConfigurationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LabWeaver read-only CSV task agent")
    parser.add_argument("--version", action="version", version="LabWeaver 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("intake")
    command.add_argument("--config", type=Path)
    command.add_argument("--csv", type=Path)
    command.add_argument("--encoding")
    command.add_argument("--delimiter", type=_delimiter)
    command.add_argument("--sample-rows", type=int)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--material", action="append", type=Path, dest="materials")
    task = command.add_mutually_exclusive_group()
    task.add_argument("--task")
    task.add_argument("--task-file", type=Path)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_const", const="offline", dest="mode")
    mode.add_argument("--live", action="store_const", const="live", dest="mode")
    command.add_argument("--env-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Python on Windows may otherwise encode redirected JSON using a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    overrides = {key: value for key, value in vars(args).items()
                 if key not in {"command", "config"} and value is not None}
    try:
        from labweaver.runtime.intake import execute_intake

        config = load_intake_config(args.config, overrides=overrides)
        report, saved = execute_intake(config)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("Could not read the task file or write the local run report.", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Run report saved: {saved}", file=sys.stderr)
    if report.get("brief_path"):
        print(f"Task brief saved: {report['brief_path']}", file=sys.stderr)
    if report.get("result_csv_paths"):
        for path in report["result_csv_paths"]:
            print(f"Result CSV saved: {path}", file=sys.stderr)
    if report.get("status") == "awaiting_input":
        print("A reply is required. Use run_labweaver.py in VS Code for a continuous conversation.", file=sys.stderr)
        return 3
    return 0 if report.get("status") == "completed" else 1

