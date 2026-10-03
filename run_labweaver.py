"""Run LabWeaver with VS Code's Run Python File button or F5."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "labweaver.toml"
# 通常修改 labweaver.toml 即可；临时覆盖示例：{"mode": "offline"}。
OVERRIDES = {}


def main() -> int:
    """Load shared settings, run intake, and display the saved result."""
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
        executable = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        print(f"缺少项目依赖，请先安装依赖并选择项目解释器：{executable}", file=sys.stderr)
        return 2

    try:
        config = load_intake_config(CONFIG_PATH, overrides=OVERRIDES)
        print(f"LabWeaver | 模式：{config.mode} | CSV：{config.csv_path.name}", flush=True)
        print("正在运行 Agent，等待工具结果和模型回答……", flush=True)
        report, saved = execute_intake(config)
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("任务文件读取或运行记录写入失败，请检查路径、编码和权限。", file=sys.stderr)
        return 2

    print(f"运行状态：{report['status']}")
    print(f"模型调用：{report['model_calls']} 次；工具尝试：{report['tool_attempts']} 次")
    print(f"实际工具执行：{len(report['execution_ledger'])} 次")
    if report.get("profile"):
        profile = report["profile"]
        print(f"数据概览：{profile['row_count']} 行 × {profile['column_count']} 列")
    if report.get("error"):
        error = report["error"]
        print(f"失败原因：{error['code']} — {error['message']}")
        if report.get("diagnostics"):
            print(f"诊断：{report['diagnostics']}")
    if report.get("final_answer"):
        print("\nAgent 回答：\n" + report["final_answer"])
    print(f"\n完整运行记录：{saved}")
    return 0 if report["status"] == "awaiting_confirmation" else 1


if __name__ == "__main__":
    raise SystemExit(main())