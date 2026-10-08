"""Run a continuous CSV task conversation with VS Code's Run Python File."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


# None loads labweaver.toml from the project working directory at invocation.
CONFIG_PATH = None
# Advanced, optional overrides. Normal use supplies paths/task in the terminal.
OVERRIDES = {}
# None inherits TOML only for noninteractive runs; an interactive empty reply uses no materials.
MATERIAL_PATHS = None
DEFAULT_TASK = "请总结这份 CSV 的规模、字段、缺失情况与数值摘要。"


def _clean_path(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value


def _selected_file(value: str, suffix: str | None = None) -> Path:
    from labweaver.config import ConfigurationError

    value = _clean_path(value)
    if not value:
        raise ConfigurationError("文件路径不能为空。")
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise ConfigurationError("文件路径无法解析或访问。") from None
    if not path.is_file():
        raise ConfigurationError("所选路径不是现有文件。")
    if suffix is not None and path.suffix.lower() != "." + suffix:
        raise ConfigurationError(f"请选择现有 {suffix.upper()} 文件。")
    return path


def _task_text(value: str, default: str = DEFAULT_TASK) -> str:
    from labweaver.config import ConfigurationError

    value = value.strip()
    if not value:
        return default
    if not value.startswith("@"):
        return value
    path = _selected_file(value[1:], "txt")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ConfigurationError("任务 TXT 超过 64 KiB，请缩短任务说明。")
    task = raw.decode("utf-8-sig", errors="strict").strip()
    if not task:
        raise ConfigurationError("任务 TXT 不能为空。")
    return task


def _interactive_config(config):
    """Collect only CSV, optional materials, and the user's own task."""
    from labweaver.config import ConfigurationError
    from labweaver.runtime.intake import resolve_task

    print("LabWeaver：只读 CSV 数据任务助手。输入退出可结束当前会话。")
    selected = input(f"CSV 完整路径（回车使用 {config.csv_path}）：").strip()
    csv_path = _selected_file(selected or str(config.csv_path))
    material_text = input("可选 TXT/MD/PDF 资料路径（多份用分号分隔；回车不提供资料）：").strip()
    materials = []
    if material_text:
        for value in material_text.split(";"):
            path = _selected_file(value)
            if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
                raise ConfigurationError("资料仅支持 TXT、Markdown 和文字型 PDF。")
            materials.append(path)
    try:
        default_task = resolve_task(config)
    except (OSError, UnicodeError):
        default_task = DEFAULT_TASK
    task = _task_text(input("任务说明（回车使用示例任务；也可输入 @任务TXT路径）："), default_task)
    return replace(config, csv_path=csv_path, material_paths=tuple(materials), task=task, task_file=None)


def _show_report(report: dict, saved: Path) -> None:
    print(f"\n运行状态：{report['status']}")
    print(f"模型调用：{report.get('model_calls', 0)} 次；实际工具执行：{len(report.get('execution_ledger', []))} 次")
    print(f"可视化模型调用：{report.get('visualization_model_calls', 0)} 次；"
          f"绘图状态：{report.get('visualization_status', 'not_needed')}")
    visualization_elapsed = sum(run.get("elapsed_seconds", 0) for run in report.get("visualization_runs", []))
    if report.get("visualization_runs"):
        print(f"可视化耗时：{visualization_elapsed:.2f} 秒")
    for run in report.get("visualization_runs", []):
        if run.get("error"):
            error = run["error"]
            print(f"绘图失败：{error['code']} — {error['message']}")
    profile = report.get("profile_result") or report.get("profile")
    if profile and profile.get("status") == "completed":
        print(f"数据概览：{profile['row_count']} 行 × {profile['column_count']} 列")
        parsing = profile.get("parsing") or report.get("parsing")
        if parsing:
            print(f"实际解析设置：{parsing}")
    print(f"资料检索：{report.get('retrieval_status', 'not_used')}")
    if report.get("error"):
        error = report["error"]
        print(f"失败原因：{error['code']} — {error['message']}")
    if report.get("final_answer"):
        print("\nAgent 回答：\n" + report["final_answer"])
    if report.get("status") == "awaiting_input":
        question = report.get("question", "")
        if isinstance(question, dict):
            question = question.get("question") or question.get("message") or str(question)
        print("\nAgent 需要确认：\n" + str(question))
    print(f"\n完整运行记录：{saved}")
    if report.get("brief_path"):
        print(f"任务简报：{report['brief_path']}")
    for path in report.get("result_csv_paths", []):
        print(f"结果 CSV：{path}")
    for paths in report.get("chart_paths", []):
        for kind, path in paths.items():
            print(f"{'绘图数据 CSV' if kind == 'data_csv' else '图表 ' + kind.upper()}：{path}")


def _conversation(config) -> int:
    """Resume interrupts and accept follow-ups on the same in-memory graph."""
    from labweaver.runtime.intake import create_runtime_session, resolve_task, save_session_report

    session = create_runtime_session(config)
    exit_words = {"退出", "取消", ":quit", "quit", "exit", "q"}
    try:
        report = session.invoke(resolve_task(config))
        while True:
            report, saved = save_session_report(report, config, session=session)
            _show_report(report, saved)
            if report.get("status") == "cancelled":
                return 0
            if report.get("status") == "error":
                print("本次执行未完成；检查失败原因后重新运行以创建新会话。")
                return 1
            if report.get("status") == "awaiting_input":
                reply = input("\n你的回答（输入退出结束）：").strip()
                if reply.lower() in exit_words:
                    report, saved = save_session_report(session.cancel(), config, session=session)
                    _show_report(report, saved)
                    return 0
                if not reply:
                    print("请回答当前问题，或输入退出。")
                    continue
                report = session.resume(reply)
            else:
                follow_up = input("\n继续追问（回车或输入退出结束）：").strip()
                if not follow_up or follow_up.lower() in exit_words:
                    return 0 if report.get("status") == "completed" else 1
                report = session.invoke(_task_text(follow_up))
    except (EOFError, KeyboardInterrupt):
        cancelled, saved = save_session_report(session.cancel(), config, session=session)
        _show_report(cancelled, saved)
        print("当前会话已结束。")
        return 0


def main(*, interactive: bool = False) -> int:
    """Load shared settings; script execution enables the conversation loop."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if sys.version_info < (3, 11):
        print("请在 VS Code 中选择项目 .venv 的 Python 3.11+ 解释器。", file=sys.stderr)
        return 2
    try:
        from labweaver.config import ConfigurationError
        from labweaver.run_config import load_intake_config
        from labweaver.runtime.intake import execute_intake
    except ModuleNotFoundError:
        executable = Path.cwd() / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        print(f"缺少项目依赖，请先安装依赖并选择项目解释器：{executable}", file=sys.stderr)
        return 2
    try:
        overrides = dict(OVERRIDES)
        if MATERIAL_PATHS is not None:
            overrides["materials"] = MATERIAL_PATHS
        config = load_intake_config(CONFIG_PATH, overrides=overrides)
        if interactive:
            config = _interactive_config(config)
        print(f"LabWeaver | 模式：{config.mode} | CSV：{config.csv_path.name}", flush=True)
        print(f"可用任务资料：{len(config.material_paths)} 份", flush=True)
        print("正在运行 Agent，等待工具结果和模型回答……", flush=True)
        if interactive:
            return _conversation(config)
        report, saved = execute_intake(config)
        _show_report(report, saved)
        if report.get("status") == "awaiting_input":
            print("需要连续回答时，使用 VS Code Run Python File 或 F5。")
            return 3
        return 0 if report.get("status") == "completed" else 1
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError, ValueError):
        print("文件读取或运行产物写入失败，请检查路径、编码和权限。", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("输入已结束；请在 VS Code 的集成终端运行此文件。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(interactive=True))
