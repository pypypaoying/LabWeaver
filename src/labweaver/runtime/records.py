"""Write JSON-safe run artifacts without replacing existing reports."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def _location_label(location: dict) -> str:
    if type(location.get("page")) is int and location["page"] > 0:
        return f"第 {location['page']} 页"
    start, end = location.get("line_start"), location.get("line_end")
    if type(start) is int and type(end) is int and 0 < start <= end:
        return f"第 {start}–{end} 行"
    raise ValueError("A cited chunk needs a valid page or line location.")


def _markdown_brief(report: dict) -> str:
    if report.get("status") != "awaiting_confirmation" or report.get("materials_completed") is not True:
        raise ValueError("A task brief requires a successful materials intake.")
    chunks = {chunk["id"]: chunk for chunk in report.get("retrieved_chunks", [])}
    citations = report.get("citations", [])
    if any(citation not in chunks for citation in citations):
        raise ValueError("A task brief cannot cite a chunk absent from retrieval evidence.")
    sources = report.get("materials", [])
    source_lookup = {material["id"]: material for material in sources}
    for citation in citations:
        chunk = chunks[citation]
        material = source_lookup.get(chunk["source_id"])
        if material is None or any(material[key] != chunk[key] for key in ("name", "sha256")):
            raise ValueError("A cited chunk must match the registered material source.")
    source = report.get("source", {})
    lines = [
        "# LabWeaver 任务简报", "",
        f"运行状态：{report['status']}", "",
        f"任务：{report.get('task', '')}", "",
        f"数据文件：`{source.get('name', '')}`", "",
        f"数据 SHA-256：`{source.get('sha256', '')}`", "",
        "## 任务方案", "", str(report.get("final_answer", "")).strip(), "",
        "## 资料与来源", "",
    ]
    for material in sources:
        lines.append(
            f"- {material['id']}：`{material['name']}`（{material['format']}）；"
            f"SHA-256：`{material['sha256']}`"
        )
    lines.extend(["", "## 引用片段", ""])
    if not citations:
        lines.append("本次没有可引用的检索片段；资料依据不足的事项需要补充确认。")
    for citation in dict.fromkeys(citations):
        chunk = chunks[citation]
        lines.extend([
            f"### [{citation}] {chunk['name']}", "",
            f"来源位置：{_location_label(chunk['location'])}", "",
            f"SHA-256：`{chunk['sha256']}`", "",
        ])
        lines.extend("> " + line for line in chunk["text"].splitlines())
        lines.append("")
    lines.extend([
        "## 当前边界", "",
        "等待用户确认分析目标；尚未执行清洗、建模或绘图。",
        "引用记录只证明片段实际检索到且出处可追溯，不自动证明回答的全部论断。", "",
    ])
    return "\n".join(lines)


def save_markdown_brief(report: dict, target_path: str | Path) -> Path:
    """Write a source-linked brief exclusively, cleaning up a failed new write."""
    payload = _markdown_brief(report)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(payload + "\n")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


def save_run(
    report: dict, output_dir: str | Path = "runs", *, with_brief: bool = False
) -> Path:
    """Save JSON and optionally its sibling brief without overwriting artifacts.

    The JSON only advertises a brief after the brief write succeeds. A failed
    write removes files newly created by this call, never preexisting files.
    Run IDs and timestamps belong to this write and are not taken from input.
    """
    document = {
        **report,
        "run_id": uuid4().hex,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    folder = Path(output_dir)
    target = folder / f"{document['run_id']}.json"
    brief_target = target.with_suffix(".md")
    document.pop("brief_path", None)
    if with_brief:
        document["brief_path"] = str(brief_target.resolve())
        _markdown_brief(document)  # Validate source references before creating files.
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    folder.mkdir(parents=True, exist_ok=True)
    brief_created = False
    json_created = False
    try:
        if with_brief:
            save_markdown_brief(document, brief_target)
            brief_created = True
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json_created = True
            stream.write(payload + "\n")
    except BaseException:
        if json_created:
            target.unlink(missing_ok=True)
        if brief_created:
            brief_target.unlink(missing_ok=True)
        raise
    return target

