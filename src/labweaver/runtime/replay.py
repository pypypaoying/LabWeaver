"""Replay successful code in the recorded immutable Docker image, without a model."""

import hashlib
from pathlib import Path
import uuid
from labweaver.runtime.artifacts import strict_json
from labweaver.runtime.execution import DockerExecutor, ExecutionConfig
from labweaver.tools.csv_profile import load_csv_snapshot


def replay_analysis(record_path, csv_path):
    path = Path(record_path)
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Replay record exceeds 100 MiB")
    report = strict_json(path.read_bytes())
    if report.get("report_version") != 2:
        raise ValueError("Replay requires a D5 report version 2")
    parsing = report["parsing"]
    snapshot = load_csv_snapshot(
        csv_path, encoding=parsing["encoding"], delimiter=parsing["delimiter"]
    )
    if snapshot.source["sha256"] != report["source"]["sha256"]:
        raise ValueError("Replay refused: input SHA-256 differs")
    successful = [
        r
        for r in report["code_executions"]
        if r["status"] == "completed" and r["artifact_ids"]
    ]
    if not successful:
        raise ValueError("No successful artifact-producing script to replay")
    assets, executions = {}, []
    session_id, task_id = uuid.uuid4().hex, uuid.uuid4().hex
    for old in successful:
        if hashlib.sha256(old["code"].encode()).hexdigest() != old["code_sha256"]:
            raise ValueError("Replay refused: code SHA-256 differs")
        image_id = old["image_id"]
        if (
            not isinstance(image_id, str)
            or len(image_id) != 71
            or not image_id.startswith("sha256:")
        ):
            raise ValueError("Replay requires an immutable image ID")
        executor = DockerExecutor(ExecutionConfig(image=image_id))
        record, new_assets = executor.execute(
            old["code"],
            snapshot,
            report["task_plan"],
            remaining_bytes=50 * 1024 * 1024
            - sum(len(b) for a in assets.values() for b in a["files"].values()),
            remaining_figures=2
            - sum(a["metadata"]["kind"] == "figure" for a in assets.values()),
        )
        executions.append(record)
        for asset in new_assets.values():
            asset["metadata"].update(session_id=session_id, task_id=task_id)
        assets.update(new_assets)
        if record["status"] != "completed":
            break
    fulfilled = {a["metadata"]["deliverable_id"] for a in assets.values()}
    missing = [d["id"] for d in report["task_plan"] if d["id"] not in fulfilled]
    result = {
        "report_version": 2,
        "session_id": session_id,
        "task_id": task_id,
        "task": report["task"],
        "source": snapshot.source,
        "parsing": snapshot.profile["parsing"],
        "task_plan": report["task_plan"],
        "code_executions": executions,
        "artifacts": [a["metadata"] for a in assets.values()],
        "fulfilled_deliverable_ids": sorted(fulfilled),
        "missing_deliverables": missing,
        "status": "completed"
        if not missing and all(r["status"] == "completed" for r in executions)
        else "error",
        "model_calls": 0,
        "analysis_model_calls": 0,
        "replay_of": str(path.resolve()),
        "answer_text": "已在相同镜像内重放成功脚本；请核对结果表与指标。",
    }
    from labweaver.runtime.records import save_run

    saved = save_run(result, path.parent / "replay", artifact_assets=assets)
    return strict_json(saved.read_bytes())
