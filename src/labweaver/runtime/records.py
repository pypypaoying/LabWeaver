"""Exclusive export with rollback, host hashes and source-bound artifacts."""

import copy
import hashlib
import json
from pathlib import Path
import re
import uuid
from labweaver.runtime.artifacts import (
    csv_table,
    validate_png,
    validate_svg,
    strict_json,
)


def _markdown_brief(report):
    from labweaver.agent import _table_markdown

    lines = [
        "# LabWeaver 任务报告",
        "",
        f"状态：{report['status']}",
        "",
        f"任务：{report.get('task', '')}",
        "",
        f"输入 SHA-256：{(report.get('source') or {}).get('sha256', '')}",
        "",
        report.get("final_answer", ""),
        "",
        "## 交付项",
        "",
    ]
    for delivery in report.get("task_plan", []):
        marker = (
            "完成"
            if delivery["id"] in report.get("fulfilled_deliverable_ids", [])
            else "未完成"
        )
        lines.append(f"- {delivery['id']}：{delivery['description']}（{marker}）")
    for artifact in report.get("artifacts", []):
        lines.extend(
            [
                "",
                f"## {artifact['label']}",
                "",
                f"执行：{artifact['execution_id']}；交付项：{artifact['deliverable_id']}",
                "",
            ]
        )
        paths = artifact.get("paths", {})
        if artifact["kind"] == "figure" and paths.get("figure.png"):
            lines.extend([f"![分析图表]({Path(paths['figure.png']).name})", ""])
        elif artifact["kind"] == "metric":
            lines.extend(
                [
                    "```json",
                    json.dumps(
                        artifact.get("value"),
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                    ),
                    "```",
                    "",
                ]
            )
        elif artifact.get("preview"):
            lines.extend(
                [
                    _table_markdown(artifact["preview"]),
                    "",
                    f"预览最多 20 行，完整结果 {artifact['row_count']} 行见 CSV。",
                    "",
                ]
            )
        lines.append(
            "文件："
            + "、".join(f"[{name}]({Path(path).name})" for name, path in paths.items())
        )
    lines.extend(
        [
            "",
            "## 执行与边界",
            "",
            "每次脚本从同一字符串快照重新计算，代码和错误记录随报告保存。源 CSV 不修改。",
            "执行成功与产物校验只证明程序运行并取得成果，不自动证明全部业务口径或结论。",
            "内存检查点仅支持本次进程中的继续对话；replay_analysis 重放成功脚本，不恢复会话。",
            "",
        ]
    )
    for chunk in report.get("citation_chunks", []):
        if chunk["id"] in report.get("citations", []):
            lines.extend(
                [
                    f"### [{chunk['id']}] {chunk['name']} {chunk['location']}",
                    "",
                    chunk["text"],
                    "",
                ]
            )
    return "\n".join(lines)


def save_run(report, output_dir="runs", *, artifact_assets=None, with_brief=True):
    """Write only registered current-task assets. Roll back all created files on failure."""
    assets = artifact_assets or {}
    output = copy.deepcopy(report)
    if output.get("report_version") != 2:
        raise ValueError("Only D5 report version 2 may be exported by this runtime")
    folder = Path(output_dir)
    for key in ("session_id", "task_id"):
        value = output.get(key)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is None
        ):
            raise ValueError("Unsafe session/task identifier")
        folder /= value
    metas = output.get("artifacts", [])
    if set(assets) != {m["id"] for m in metas}:
        raise ValueError("Export requires matching registered artifact bytes")
    total = 0
    for meta in metas:
        asset = assets[meta["id"]]
        if (
            asset["metadata"] != meta
            or meta["source"] != output.get("source")
            or meta["session_id"] != output["session_id"]
            or meta["task_id"] != output["task_id"]
        ):
            raise ValueError("Artifact provenance mismatch")
        if set(asset["files"]) != set(meta["files"]):
            raise ValueError("Artifact file manifest mismatch")
        for name, raw in asset["files"].items():
            if not re.fullmatch(
                r"(?:table|data)\.csv|metric\.json|figure\.(?:png|svg)", name
            ):
                raise ValueError("Unsafe artifact filename")
            if (
                hashlib.sha256(raw).hexdigest() != meta["files"][name]["sha256"]
                or len(raw) != meta["files"][name]["size"]
            ):
                raise ValueError("Artifact bytes/hash mismatch")
            total += len(raw)
            if name.endswith(".png"):
                validate_png(raw)
            elif name.endswith(".svg"):
                validate_svg(raw)
            elif name.endswith(".json"):
                strict_json(raw)
            else:
                columns, rows = csv_table(raw)
                if columns != meta["columns"] or len(rows) != meta["row_count"]:
                    raise ValueError("Table bytes/metadata mismatch")
    if total > 50 * 1024 * 1024:
        raise ValueError("Artifacts exceed 50 MiB")
    folder.mkdir(parents=True, exist_ok=True)
    export_id, created = uuid.uuid4().hex, []

    def write(target, payload):
        with target.open("xb") as stream:
            created.append(target)
            stream.write(payload)

    try:
        all_paths, csv_paths, chart_paths = [], [], []
        for number, meta in enumerate(metas, 1):
            asset, paths = assets[meta["id"]], {}
            for name, raw in asset["files"].items():
                path = folder / f"{export_id}-artifact-{number}-{name}"
                write(path, raw)
                paths[name] = str(path.resolve())
                all_paths.append(str(path.resolve()))
                if name == "table.csv":
                    csv_paths.append(str(path.resolve()))
            meta["paths"] = paths
            if meta["kind"] == "figure":
                chart_paths.append(paths)
            if meta["kind"] == "table":
                columns, rows = csv_table(asset["files"]["table.csv"])
                meta["preview"] = {"columns": columns, "rows": rows[:20]}
        for number, execution in enumerate(output.get("code_executions", []), 1):
            if (
                hashlib.sha256(execution["code"].encode()).hexdigest()
                != execution["code_sha256"]
            ):
                raise ValueError("Analysis code hash mismatch")
            path = folder / f"{export_id}-attempt-{number}.py"
            write(path, execution["code"].encode("utf-8"))
            execution["code_path"] = str(path.resolve())
            all_paths.append(str(path.resolve()))
        output.update(
            artifact_paths=all_paths,
            result_csv_paths=csv_paths,
            chart_paths=chart_paths,
        )
        if with_brief and output["status"] in {"completed", "error"}:
            path = folder / f"{export_id}.md"
            output["brief_path"] = str(path.resolve())
            write(path, _markdown_brief(output).encode("utf-8"))
        path = folder / f"{export_id}.json"
        output["record_path"] = str(path.resolve())
        write(
            path,
            json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False).encode(
                "utf-8"
            ),
        )
        return path
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
