"""Write strict JSON, a source-linked brief, and deterministic result CSVs."""

from datetime import datetime, timezone
import csv
import io
import json
import hashlib
from pathlib import Path
import re
from uuid import uuid4


def _location_label(location: dict) -> str:
    if type(location.get("page")) is int and location["page"] > 0:
        return f"第 {location['page']} 页"
    start, end = location.get("line_start"), location.get("line_end")
    if type(start) is int and type(end) is int and 0 < start <= end:
        return f"第 {start}–{end} 行"
    raise ValueError("A cited chunk needs a valid page or line location.")


def _result_tables(report: dict) -> list[dict]:
    """Validate program result tables before serializing artifacts."""
    if report.get("status") not in {"completed", "error"}:
        return []
    tables = []
    for result in report.get("analysis_results", []):
        if result.get("status") != "completed":
            continue
        columns, rows = result.get("columns"), result.get("rows")
        if not isinstance(columns, list) or not columns or any(
            not isinstance(column, str) for column in columns
        ) or len(set(columns)) != len(columns):
            raise ValueError("An analysis table requires distinct string column names.")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or set(row) != set(columns) for row in rows
        ):
            raise ValueError("Analysis rows must match their declared table columns.")
        source = report.get("source")
        if source and result.get("source") != source:
            raise ValueError("An analysis table must match the bound CSV source.")
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        tables.append(result)
    return tables


def _markdown_cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _markdown_brief(report: dict) -> str:
    if report.get("status") != "completed":
        raise ValueError("A task brief requires a completed task.")
    chunks = {chunk["id"]: chunk for chunk in report.get("citation_chunks", report.get("retrieved_chunks", []))}
    citations = report.get("citations", [])
    if any(citation not in chunks for citation in citations):
        raise ValueError("A task brief cannot cite a chunk absent from retrieval evidence.")
    sources = report.get("materials", [])
    source_lookup = {material["id"]: material for material in sources}
    for citation in citations:
        chunk = chunks[citation]
        material = source_lookup.get(chunk["source_id"])
        if material is None or any(material.get(key) != chunk.get(key) for key in ("name", "sha256")):
            raise ValueError("A cited chunk must match the registered material source.")
    source = report.get("source") or {}
    lines = [
        "# LabWeaver 任务结果", "",
        f"运行状态：{report['status']}", "",
        f"任务：{report.get('task', '')}", "",
        f"数据文件：\x60{source.get('name', '')}\x60", "",
        f"数据 SHA-256：\x60{source.get('sha256', '')}\x60", "",
        "## Agent 回答", "", str(report.get("answer_text", report.get("final_answer", ""))).strip(), "",
    ]
    for number, result in enumerate(_result_tables(report), 1):
        columns = result["columns"]
        lines.extend([
            f"## 实际统计结果 {number}", "",
            "| " + " | ".join(_markdown_cell(column) for column in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ])
        for row in result["rows"]:
            lines.append("| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |")
        lines.extend(["", "统计请求：", "", "\x60\x60\x60json", json.dumps(result.get("spec", {}), ensure_ascii=False, indent=2, allow_nan=False), "\x60\x60\x60", ""])
        if result.get("truncated"):
            lines.extend([f"本表按请求的 top_k={result['spec']['top_k']} 选取；分组总数见 JSON。", ""])
    if report.get("charts"):
        lines.extend(["## 数据可视化", ""])
        for chart in report["charts"]:
            paths = chart.get("paths") or {}
            if paths.get("png"):
                name = Path(paths["png"]).name
                lines.extend([f"![统计图表]({name})", "", f"图表：{chart['spec'].get('chart_type', '')}；数据 ID：{chart['data_id']}。", ""])
                lines.append("文件：" + "、".join(f"[{extension}]({Path(path).name})" for extension, path in paths.items()))
                lines.append("")
        lines.extend([f"绘图决策：{report.get('visualization_reason', '')}", ""])
    lines.extend(["## 资料与来源", ""])
    for material in sources:
        lines.append(f"- {material['id']}：\x60{material['name']}\x60（{material.get('format', '')}）；SHA-256：\x60{material['sha256']}\x60")
    lines.extend(["", f"检索状态：{report.get('retrieval_status', 'not_used')}", "", "## 引用片段", ""])
    if not citations:
        lines.append("本次没有可引用的检索片段；CSV 统计以实际工具结果为准。")
    for citation in dict.fromkeys(citations):
        chunk = chunks[citation]
        lines.extend([
            f"### [{citation}] {chunk['name']}", "",
            f"来源位置：{_location_label(chunk['location'])}", "",
            f"SHA-256：\x60{chunk['sha256']}\x60", "",
        ])
        lines.extend("> " + line for line in chunk["text"].splitlines())
        lines.append("")
    lines.extend([
        "## 当前边界", "",
        "执行只读概览、筛选、聚合、排名、分布计算与受控绘图；源文件保持不变。",
        "引用记录证明片段实际检索到且出处可追溯，不自动证明回答的全部论断。",
        "当前进程的内存检查点支持继续对话；JSON 记录不提供退出后恢复。", "",
    ])
    return "\n".join(lines)


def _write_exclusive(target: Path, payload: str) -> None:
    created = False
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(payload)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def save_markdown_brief(report: dict, target_path: str | Path) -> Path:
    """Write a completed task brief without replacing an existing artifact."""
    payload = _markdown_brief(report)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(target, payload + "\n")
    return target


def _artifact_folder(report: dict, output_dir: str | Path) -> Path:
    folder = Path(output_dir)
    for key in ("session_id", "task_id"):
        value = report.get(key)
        if value is not None:
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is None:
                raise ValueError(f"A report {key} must be a safe artifact identifier.")
            folder /= value
    return folder


def save_run(report: dict, output_dir: str | Path = "runs", *, with_brief: bool = False,
             chart_assets: dict | None = None) -> Path:
    """Persist one task event and validated results using exclusive writes.

    Events are grouped by session and task. A failed write removes only files
    created by this call; existing artifacts survive collisions and failures.
    """
    document = {**report, "run_id": uuid4().hex, "recorded_at": datetime.now(timezone.utc).isoformat()}
    folder = _artifact_folder(document, output_dir)
    target = folder / f"{document['run_id']}.json"
    brief_target = target.with_suffix(".md")
    document.pop("brief_path", None)
    document.pop("result_csv_paths", None)
    document.pop("chart_paths", None)
    chart_payloads = []
    chart_paths = []
    assets = chart_assets or {}
    document["charts"] = [dict(chart) for chart in report.get("charts", [])]
    chart_ids = [chart.get("chart_id") for chart in document["charts"]]
    if len(chart_ids) != len(set(chart_ids)):
        raise ValueError("Chart IDs must be unique.")
    for chart in document["charts"]:
        cid = chart.get("chart_id")
        if not isinstance(cid, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", cid) is None:
            raise ValueError("A chart requires a safe registered identifier.")
        asset = assets.get(cid)
        metadata = {key: value for key, value in chart.items() if key != "paths"}
        if not isinstance(asset, dict) or asset.get("metadata") != metadata:
            raise ValueError("A chart requires matching registered host assets.")
        if metadata.get("source") != report.get("source") or any(metadata.get(key) != report.get(key) for key in ("session_id", "task_id")):
            raise ValueError("A chart must match its session, task and source.")
        png, svg, data_csv = asset.get("png"), asset.get("svg"), asset.get("data_csv")
        if not isinstance(png, bytes) or not png.startswith(b'\x89PNG\r\n\x1a\n') or not isinstance(svg, bytes) or b'<svg' not in svg or not isinstance(data_csv, str):
            raise ValueError("Chart assets must contain real PNG, SVG and data CSV payloads.")
        chart["paths"] = {}
        for extension, payload in (("png", png), ("svg", svg), ("data_csv", data_csv.encode("utf-8"))):
            if metadata.get("hashes", {}).get(extension) != hashlib.sha256(payload).hexdigest():
                raise ValueError("Chart asset hash mismatch.")
            suffix = "csv" if extension == "data_csv" else extension
            path = target.with_name(f"{target.stem}-chart-{cid}.{suffix}")
            chart["paths"][extension] = str(path.resolve())
            chart_payloads.append((path, payload))
        chart_paths.append(dict(chart["paths"]))
    if chart_paths:
        document["chart_paths"] = chart_paths
    result_payloads = []
    for number, result in enumerate(_result_tables(document), 1):
        csv_target = target.with_name(f"{target.stem}-result-{number}.csv")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=result["columns"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
        result_payloads.append((csv_target, stream.getvalue()))
    if result_payloads:
        document["result_csv_paths"] = [str(path.resolve()) for path, _ in result_payloads]
    brief_payload = None
    if with_brief:
        document["brief_path"] = str(brief_target.resolve())
        brief_payload = _markdown_brief(document) + "\n"
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    folder.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for chart_target, chart_payload in chart_payloads:
            _write_binary_exclusive(chart_target, chart_payload)
            created.append(chart_target)
        for csv_target, csv_payload in result_payloads:
            _write_exclusive(csv_target, csv_payload)
            created.append(csv_target)
        if brief_payload is not None:
            _write_exclusive(brief_target, brief_payload)
            created.append(brief_target)
        _write_exclusive(target, payload)
        created.append(target)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return target


def _write_binary_exclusive(target: Path, payload: bytes) -> None:
    created = False
    try:
        with target.open("xb") as stream:
            created = True
            stream.write(payload)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
