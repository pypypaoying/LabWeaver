"""Verify D3 offline by clicking Run Python File in VS Code.

The checks use real local materials and the real Deep Agents tool loop. They
verify execution and source evidence, rather than a live model's understanding.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_CHECKS = 12


@contextmanager
def _offline_environment(project_root: Path):
    """Deny outgoing connections and restore process state, even on failure."""
    import os
    import socket

    previous_cwd = Path.cwd()
    tracing_names = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
    previous_environment = {name: os.environ.get(name) for name in tracing_names}
    previous_connections = (
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex,
    )
    attempts = []

    def deny_connection(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("Offline verification attempted a network connection.")

    try:
        for name in tracing_names:
            os.environ[name] = "false"
        socket.create_connection = deny_connection
        socket.socket.connect = deny_connection
        socket.socket.connect_ex = deny_connection
        # The tokenizer's writable cache is local to this project, regardless
        # of the directory from which VS Code launched this file.
        os.chdir(project_root)
        yield attempts
    finally:
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex = previous_connections
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        os.chdir(previous_cwd)


def _project_path(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Verification resources must belong to the project.")
    return path


def _hashes(paths) -> dict:
    import hashlib

    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _check(condition: bool, label: str) -> None:
    # Explicit checks still execute if Python was launched with -O.
    if not condition:
        raise AssertionError(label)


def _check_tool_evidence(report: dict, model) -> None:
    import json

    ledger = report["execution_ledger"]
    calls = [item for item in report["trace"] if item["kind"] == "tool_call"]
    results = [item for item in report["trace"] if item["kind"] == "tool_result"]
    names = ["profile_csv", "search_materials"]
    _check([entry["name"] for entry in ledger] == names, "Actual tool execution order")
    _check([entry["name"] for entry in calls] == names, "Model tool request order")
    _check(len(calls) == len(results) == len(ledger) == 2, "Paired tool evidence count")
    _check(len({call["id"] for call in calls}) == 2, "Unique tool call IDs")
    for call in calls:
        matching_results = [item for item in results if item["tool_call_id"] == call["id"]]
        matching_executions = [item for item in ledger if item["tool_call_id"] == call["id"]]
        _check(len(matching_results) == len(matching_executions) == 1, "Tool ID pairing")
        result, execution = matching_results[0], matching_executions[0]
        _check(call["name"] == result["name"] == execution["name"], "Paired tool names")
        _check(result["status"] == "success", "Real ToolMessage succeeded")
        _check(json.loads(result["content"]) == execution["result"], "ToolMessage equals actual result")
        if call["name"] == "search_materials":
            _check(call["args"]["query"] == execution["result"]["query"], "Actual retrieval query")
    _check(set(model.seen_tool_results) == {entry["tool_call_id"] for entry in ledger},
           "Offline model consumed real paired ToolMessages")
    _check(model.bound_tool_sets[0] == ["profile_csv"], "Profile tool exposed first")
    _check("search_materials" in model.bound_tool_sets[-1], "Retrieval exposed after profile")


def _check_run(report: dict, model, index, csv_path: Path, dimensions: tuple[int, int],
               original_hashes: dict, *, no_hits: bool) -> None:
    import json
    import re

    _check(report["status"] == "awaiting_confirmation", "Intake succeeded and stopped for confirmation")
    _check(report["profile_completed"] is True and report["materials_completed"] is True,
           "Completed real profile and material retrieval")
    _check(report["model_calls"] == 3 and report["tool_attempts"] == 2, "Bounded offline call counts")
    _check(report["source"] == {"name": csv_path.name, "sha256": original_hashes[csv_path]},
           "Observed CSV identity and hash")
    profile = report["profile"]
    _check((profile["row_count"], profile["column_count"]) == dimensions, "Expected CSV dimensions")
    _check(report["materials"] == index.sources, "Registered real material metadata")
    _check_tool_evidence(report, model)
    retrieval = report["execution_ledger"][1]["result"]
    _check(retrieval["status"] == "completed", "Retrieval completed")
    _check(report["retrieval_queries"] == [retrieval["query"]], "Recorded actual retrieval query")
    _check(report["retrieved_chunks"] == retrieval["matches"], "Recorded actual retrieved chunks")
    chunk_lookup = {chunk["id"]: chunk for chunk in retrieval["matches"]}
    for chunk in chunk_lookup.values():
        _check(index.chunk_by_id(chunk["id"]) == chunk, "Retrieved text matches local index")
    _check(set(report["citations"]) <= set(chunk_lookup), "Only retrieved chunk IDs cited")
    answer_ids = list(dict.fromkeys(re.findall(r"\[(D\d+-C\d+)\]", report["final_answer"])))
    _check(answer_ids == report["citations"], "Answer citation IDs match citation records")
    if no_hits:
        _check(not report["retrieved_chunks"] and not report["citations"], "No-answer query has no fake evidence")
        _check("资料不足" in report["final_answer"], "No-answer case states missing material evidence")
    else:
        _check(bool(report["retrieved_chunks"]) and bool(report["citations"]), "Answer uses actual citations")
    _check(_hashes(original_hashes) == original_hashes, "Input files unchanged")
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def _check_saved(report: dict, saved: Path) -> None:
    import json

    brief = saved.with_suffix(".md")
    _check(saved.is_file() and brief.is_file() and saved.stem == brief.stem, "Saved JSON/Markdown sibling pair")
    recorded = json.loads(saved.read_text(encoding="utf-8"))
    _check(recorded["brief_path"] == str(brief.resolve()), "JSON points to actual Markdown brief")
    _check(bool(recorded["run_id"]) and bool(recorded["recorded_at"]), "Saved run identity and timestamp")
    _check(all(recorded[key] == value for key, value in report.items()), "Saved actual Agent report")
    text = brief.read_text(encoding="utf-8")
    chunks = {chunk["id"]: chunk for chunk in report["retrieved_chunks"]}
    for citation in report["citations"]:
        chunk = chunks[citation]
        _check(f"[{citation}]" in text and chunk["name"] in text and chunk["sha256"] in text,
               "Markdown includes real source identity")
        location = chunk["location"]
        position = (f"第 {location['page']} 页" if "page" in location else
                    f"第 {location['line_start']}–{location['line_end']} 行")
        _check(position in text, "Markdown includes real page or line location")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if sys.version_info < (3, 11):
        print("[FAIL] 请在 VS Code 中选择项目 .venv 的 Python 3.11+ 解释器。")
        return 1

    # Project/dependency imports are intentionally below the version check and
    # inside the offline context. No model configuration or credentials load.
    passed = 0
    stage = "初始化离线验证"
    output_dir = None

    def pass_check(label: str) -> None:
        nonlocal passed
        passed += 1
        print(f"[PASS {passed:02d}] {label}", flush=True)

    try:
        import json
        from datetime import datetime, timezone
        from uuid import uuid4

        with _offline_environment(PROJECT_ROOT) as network_attempts:
            from labweaver.agent import run_intake
            from labweaver.materials import build_material_index
            from labweaver.offline import OfflineIntakeModel
            from labweaver.runtime.records import save_run

            output_dir = PROJECT_ROOT / "runs" / (
                "d3-verification-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-" + uuid4().hex[:8]
            )
            query_path = _project_path("examples/retrieval_queries.json")
            labeled_queries = json.loads(query_path.read_text(encoding="utf-8"))
            _check(len(labeled_queries) == 7 and len({item["id"] for item in labeled_queries}) == 7,
                   "Seven distinct labeled retrieval cases")
            indices = {}

            def selected_index(relative_paths):
                paths = tuple(_project_path(relative) for relative in relative_paths)
                if paths not in indices:
                    indices[paths] = build_material_index(list(paths))
                return paths, indices[paths]

            for case in labeled_queries:
                stage = "标签检索 " + case["id"]
                paths, index = selected_index(case["materials"])
                before = _hashes(paths)
                result = index.search(case["query"])
                _check(result["status"] == "completed" and result["query"] == case["query"],
                       "Labeled retrieval completed")
                for chunk in result["matches"]:
                    _check(index.chunk_by_id(chunk["id"]) == chunk, "Literal local retrieval evidence")
                expected = case["expected"]
                if expected.get("no_hits"):
                    _check(result["matches"] == [], "Expected zero retrieval matches")
                else:
                    _check(any(
                        chunk["name"] == expected["source_name"]
                        and expected["text_contains"] in chunk["text"]
                        and ("page" not in expected or chunk["location"].get("page") == expected["page"])
                        for chunk in result["matches"]
                    ), "Expected source passage/page retrieved")
                _check(_hashes(paths) == before, "Retrieval did not modify sources")
                pass_check(stage)

            survey_materials = labeled_queries[0]["materials"]
            experiment_materials = next(case["materials"] for case in labeled_queries
                                        if case["id"] == "experiment_comparison")
            alternative_materials = next(case["materials"] for case in labeled_queries
                                         if case["id"] == "survey_alternative_goal")
            survey_task = _project_path("examples/tasks/survey_rag.txt").read_text(encoding="utf-8-sig")
            experiment_task = _project_path("examples/tasks/experiments_rag.txt").read_text(encoding="utf-8-sig")
            cases = [
                ("survey", "examples/data/survey.csv", survey_task, survey_materials, (8, 5), False),
                ("experiments", "examples/data/experiments.csv", experiment_task, experiment_materials, (6, 6), False),
                ("alternative", "examples/data/survey.csv", survey_task, alternative_materials, (8, 5), False),
                ("no-answer", "examples/data/survey.csv", "quasar_redshift", survey_materials, (8, 5), True),
            ]
            reports = {}
            for name, csv_relative, task, material_relatives, dimensions, no_hits in cases:
                stage = "真实离线 Agent " + name
                paths, index = selected_index(material_relatives)
                csv_path = _project_path(csv_relative)
                before = _hashes((csv_path, *paths))
                model = OfflineIntakeModel()
                report = run_intake(task, csv_path, model, material_paths=list(paths))
                report["mode"] = "offline"
                _check_run(report, model, index, csv_path, dimensions, before, no_hits=no_hits)
                saved = save_run(report, output_dir, with_brief=True)
                _check_saved(report, saved)
                _check(_hashes(before) == before, "Saving did not modify input sources")
                reports[name] = report
                pass_check(stage + "：真实工具配对、引用及 JSON/MD 保存")

            stage = "相同 CSV 更换资料"
            standard, alternative = reports["survey"], reports["alternative"]
            _check(standard["source"] == alternative["source"] and standard["profile"] == alternative["profile"],
                   "Changing documents leaves observed CSV statistics unchanged")

            def references(report):
                cited = set(report["citations"])
                return {(chunk["name"], chunk["sha256"], chunk["text"])
                        for chunk in report["retrieved_chunks"] if chunk["id"] in cited}

            _check(references(standard) != references(alternative), "Changing documents changes actual cited evidence")
            _check(any(chunk["name"] == "requirements_alternative.md"
                       for chunk in alternative["retrieved_chunks"] if chunk["id"] in alternative["citations"]),
                   "Alternative document actually cited")
            _check(network_attempts == [], "All verification completed without a connection attempt")
            pass_check(stage + "：统计不变，实际引用来源改变")
            _check(passed == EXPECTED_CHECKS, "All intended checks completed")
    except Exception as exc:
        # Never display arbitrary parser/model exception text or credentials.
        print(f"[FAIL] {stage}（{type(exc).__name__}）", file=sys.stderr)
        print(f"Passed: {passed}/{EXPECTED_CHECKS}")
        if output_dir is not None and output_dir.exists():
            print(f"已保存的运行记录目录：{output_dir}")
        return 1

    print(f"Passed: {passed}/{EXPECTED_CHECKS}")
    print(f"运行记录目录：{output_dir}")
    print("离线检查验证执行流程和资料出处；真实模型的理解质量需要另行验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
