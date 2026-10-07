"""Run two live checks with user-selected CSV/PDF and task in VS Code."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "labweaver.toml"
DEFAULT_TASK = (
    "请总结这份 CSV 的行列规模、字段、逐列缺失情况和工具已计算的数值摘要。"
    "区分实际观测与推测，说明字段含义的不确定之处，列出待确认问题。"
    "若提供资料，请主动检索资料，用实际返回的片段解释相关字段、单位或方法要求，"
    "并标明出处；资料不支持时明确说明。不要声称已执行清洗、建模或绘图。"
)


def _selected_file(value: str, suffix: str) -> Path:
    from labweaver.config import ConfigurationError

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1]
    if not cleaned:
        raise ConfigurationError(f"请选择一个 {suffix.upper()} 文件。")
    try:
        path = Path(cleaned).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise ConfigurationError("文件路径无法解析或访问。") from None
    if path.suffix.lower() != f".{suffix}" or not path.is_file():
        raise ConfigurationError(f"所选 {suffix.upper()} 路径不是对应格式的现有文件。")
    return path


def _task_text(value: str) -> str:
    from labweaver.config import ConfigurationError

    value = value.strip()
    if not value:
        return DEFAULT_TASK
    if not value.startswith("@"):
        return value
    path = _selected_file(value[1:], "txt")
    try:
        with path.open("rb") as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ConfigurationError("任务 TXT 超过 64 KiB，请缩短任务说明。")
        task = raw.decode("utf-8-sig", errors="strict").strip()
    except (OSError, UnicodeError):
        raise ConfigurationError("任务 TXT 需要可读取的 UTF-8 文本。") from None
    if not task:
        raise ConfigurationError("任务 TXT 不能为空。")
    return task


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _actual_profile(report: dict) -> dict | None:
    for entry in report.get("execution_ledger", []):
        result = entry.get("result")
        if (entry.get("name") == "profile_csv" and isinstance(result, dict)
                and result.get("status") == "completed"):
            return result
    return None


def _run_pair(config, csv_path: Path, pdf_path: Path, task: str, output_dir: Path) -> dict:
    """Use the same task/CSV settings for both cases; retain failures as evidence."""
    from labweaver.runtime.intake import execute_intake

    before = {"csv": _digest(csv_path), "pdf": _digest(pdf_path)}
    cases = []
    for name, materials in (("csv-only", ()), ("csv-with-pdf", (pdf_path,))):
        selected = replace(
            config, csv_path=csv_path, task=task, task_file=None,
            material_paths=materials, output_dir=output_dir / name,
        )
        print(f"\n正在运行：{name} | 模式：{selected.mode}", flush=True)
        report, saved = execute_intake(selected)
        cases.append({"name": name, "report": report, "saved": saved})
        print(f"状态：{report['status']}；实际工具执行：{len(report['execution_ledger'])}")
        if report.get("error"):
            error = report["error"]
            print(f"失败原因：{error['code']} — {error['message']}")
        print("\nAgent 回答：\n" + report.get("final_answer", ""))
        print(f"JSON：{saved}")
        if report.get("brief_path"):
            print(f"Markdown：{report['brief_path']}")

    first, second = (case["report"] for case in cases)
    first_profile, second_profile = _actual_profile(first), _actual_profile(second)
    after = {"csv": _digest(csv_path), "pdf": _digest(pdf_path)}
    checks = {
        "csv_only_success": first.get("status") == "awaiting_confirmation",
        "csv_with_pdf_success": second.get("status") == "awaiting_confirmation",
        "pdf_has_retrieved_citations": bool(second.get("citations"))
            and second.get("materials_completed") is True,
        "csv_statistics_identical": first_profile is not None and first_profile == second_profile,
        "inputs_unchanged": before == after,
    }
    summary = {
        "mode": config.mode,
        "task": task,
        "sources": {"csv": csv_path.name, "pdf": pdf_path.name},
        "hashes_before": before,
        "hashes_after": after,
        "checks": checks,
        "passed": all(checks.values()),
        "runs": [{"case": case["name"], "status": case["report"]["status"],
                  "json_path": str(case["saved"].resolve()),
                  "brief_path": case["report"].get("brief_path")}
                 for case in cases],
    }
    target = output_dir / "comparison.json"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("\n两项检验汇总：")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
    if not checks["pdf_has_retrieved_citations"]:
        print("本次没有完成带 PDF 出处的总结，请查看检索命中、资料相关性或失败原因。")
    print(f"比较记录：{target}")
    return summary


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if sys.version_info < (3, 11):
        print("请在 VS Code 中选择项目 .venv 的 Python 3.11+ 解释器。", file=sys.stderr)
        return 2
    try:
        from labweaver.config import ConfigurationError
        from labweaver.run_config import load_intake_config
    except ModuleNotFoundError:
        print("缺少项目依赖，请选择项目 .venv 解释器。", file=sys.stderr)
        return 2
    try:
        print("LabWeaver 真实检验：同一 CSV 和任务，分别运行无资料与单 PDF 资料。")
        print("使用现有模型配置在线调用；运行记录写入项目 runs/。")
        csv_path = _selected_file(input("CSV 完整路径："), "csv")
        pdf_path = _selected_file(input("文字型 PDF 完整路径："), "pdf")
        task = _task_text(input("任务说明（回车使用通用总结；多行任务可输入 @任务TXT完整路径）："))
        encoding = input("CSV 编码（回车为 utf-8-sig，可填 gb18030）：").strip() or "utf-8-sig"
        delimiter = input("CSV 分隔符（回车为逗号，可填 tab 或其他单字符）：") or ","
        config = load_intake_config(CONFIG_PATH, overrides={
            "mode": "live", "encoding": encoding, "delimiter": delimiter,
        })
        output_dir = PROJECT_ROOT / "runs" / f"real-checks-{uuid4().hex}"
        summary = _run_pair(config, csv_path, pdf_path, task, output_dir)
        return 0 if summary["passed"] else 1
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("输入文件读取或运行产物写入失败，请检查路径、编码和权限。", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("输入已结束；请在 VS Code 的集成终端运行此文件。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
