"""One-click offline acceptance for retrieval and continuous CSV statistics.

The historical filename is retained for compatibility. These checks validate
real tool results and artifacts, using a deterministic model and no API.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_CHECKS = 12


@contextmanager
def _offline_environment(project_root: Path):
    """Deny network connections and restore process state, even on failure."""
    import os
    import socket

    previous_cwd = Path.cwd()
    names = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
    previous_environment = {name: os.environ.get(name) for name in names}
    previous_connections = (socket.create_connection, socket.socket.connect, socket.socket.connect_ex)
    attempts = []

    def deny_connection(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("Offline verification attempted a network connection.")

    try:
        for name in names:
            os.environ[name] = "false"
        socket.create_connection = deny_connection
        socket.socket.connect = deny_connection
        socket.socket.connect_ex = deny_connection
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
    if not condition:
        raise AssertionError(label)


def _check_tool_evidence(report: dict, model) -> None:
    import json

    ledger = report["execution_ledger"]
    calls = [item for item in report["trace"] if item["kind"] == "tool_call"]
    results = [item for item in report["trace"] if item["kind"] == "tool_result"]
    _check(len(calls) == len(results) == len(ledger), "Completed tool evidence count")
    _check(len({call["id"] for call in calls}) == len(calls), "Unique tool call IDs")
    for call in calls:
        paired_result = [item for item in results if item["tool_call_id"] == call["id"]]
        paired_execution = [item for item in ledger if item["tool_call_id"] == call["id"]]
        _check(len(paired_result) == len(paired_execution) == 1, "Tool ID pairing")
        result, execution = paired_result[0], paired_execution[0]
        _check(call["name"] == result["name"] == execution["name"], "Paired tool names")
        _check(result["status"] == "success", "ToolMessage succeeded")
        _check(json.loads(result["content"]) == execution["result"], "ToolMessage equals execution result")
    _check({entry["tool_call_id"] for entry in ledger} <= set(model.seen_tool_results),
           "Offline model consumed actual ToolMessages")
    _check(all(set(tools) <= {"profile_csv", "analyze_csv", "search_materials", "ask_user"}
               for tools in model.bound_tool_sets), "Only allowed tools exposed")


def _check_saved(report: dict, saved: Path) -> None:
    import csv
    import json

    recorded = json.loads(saved.read_text(encoding="utf-8"))
    brief = saved.with_suffix(".md")
    _check(brief.is_file() and recorded["brief_path"] == str(brief.resolve()), "Actual Markdown brief")
    _check(all(recorded[key] == value for key, value in report.items()), "Recorded real report")
    chunks = {chunk["id"]: chunk for chunk in report.get("citation_chunks", report.get("retrieved_chunks", []))}
    text = brief.read_text(encoding="utf-8")
    for citation in report.get("citations", []):
        chunk = chunks[citation]
        _check(citation in text and chunk["name"] in text and chunk["sha256"] in text, "Saved true citation")
        location = chunk["location"]
        label = f"第 {location['page']} 页" if "page" in location else f"第 {location['line_start']}–{location['line_end']} 行"
        _check(label in text, "Saved source location")
    tables = [result for result in report.get("analysis_results", []) if result.get("status") == "completed"]
    _check(len(recorded.get("result_csv_paths", [])) == len(tables), "Every actual table exported")
    for table, filename in zip(tables, recorded.get("result_csv_paths", [])):
        with Path(filename).open(encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
        expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in table["rows"]]
        _check(actual == expected, "Result CSV exactly matches tool table")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if sys.version_info < (3, 11):
        print("[FAIL] 请在 VS Code 中选择项目 .venv 的 Python 3.11+ 解释器。")
        return 1
    passed, stage, output_dir = 0, "初始化离线验证", None

    def pass_check(label: str):
        nonlocal passed
        passed += 1
        print(f"[PASS {passed:02d}] {label}", flush=True)

    try:
        import json
        from uuid import uuid4

        with _offline_environment(PROJECT_ROOT) as network_attempts:
            from labweaver.agent import create_session, run_intake
            from labweaver.materials import build_material_index
            from labweaver.offline import OfflineIntakeModel
            from labweaver.runtime.records import save_run

            output_dir = PROJECT_ROOT / "runs" / ("session-verification-" + uuid4().hex)
            labeled = json.loads(_project_path("examples/retrieval_queries.json").read_text(encoding="utf-8"))
            _check(len(labeled) == 7, "Seven labeled retrieval queries")
            all_inputs = {}
            indices = {}

            def selected_index(relatives):
                paths = tuple(_project_path(relative) for relative in relatives)
                all_inputs.update(_hashes(paths))
                if paths not in indices:
                    indices[paths] = build_material_index(list(paths))
                return paths, indices[paths]

            stage = "七个标注检索查询（含无答案）"
            for case in labeled:
                paths, index = selected_index(case["materials"])
                result = index.search(case["query"])
                _check(result["status"] == "completed", "Labeled retrieval completed")
                _check(all(index.chunk_by_id(chunk["id"]) == chunk for chunk in result["matches"]), "Literal chunks")
                expected = case["expected"]
                if expected.get("no_hits"):
                    _check(result["matches"] == [], "No fabricated no-answer result")
                else:
                    _check(any(chunk["name"] == expected["source_name"]
                               and expected["text_contains"] in chunk["text"]
                               and ("page" not in expected or chunk["location"].get("page") == expected["page"])
                               for chunk in result["matches"]), "Expected passage and location")
            pass_check(stage)

            survey_materials = labeled[0]["materials"]
            experiment_materials = next(case["materials"] for case in labeled if case["id"] == "experiment_comparison")
            alternative_materials = next(case["materials"] for case in labeled if case["id"] == "survey_alternative_goal")
            survey_task = _project_path("examples/tasks/survey_rag.txt").read_text(encoding="utf-8-sig")
            experiment_task = _project_path("examples/tasks/experiments_rag.txt").read_text(encoding="utf-8-sig")
            cases = [
                ("问卷", "examples/data/survey.csv", survey_task, survey_materials, (8, 5), False),
                ("实验", "examples/data/experiments.csv", experiment_task, experiment_materials, (6, 6), False),
                ("替代资料", "examples/data/survey.csv", survey_task, alternative_materials, (8, 5), False),
                ("资料无答案", "examples/data/survey.csv", "资料 quasar_redshift", survey_materials, (8, 5), True),
            ]
            reports, saved_reports = {}, []
            for name, relative, task, material_relatives, dimensions, no_hits in cases:
                stage = "真实离线 Agent：" + name
                paths, index = selected_index(material_relatives)
                source = _project_path(relative)
                all_inputs.update(_hashes([source]))
                model = OfflineIntakeModel()
                report = run_intake(task, source, model, material_paths=paths)
                _check(report["status"] == "completed", "Task completed")
                profile = report["profile"]
                _check((profile["row_count"], profile["column_count"]) == dimensions, "Actual dimensions")
                _check([entry["name"] for entry in report["execution_ledger"]] == ["profile_csv", "search_materials"], "Real retrieval order")
                _check(report["retrieval_status"] == "used", "Actual retrieval recorded")
                _check_tool_evidence(report, model)
                _check(report["materials"] == index.sources, "Actual source identity")
                _check(all(index.chunk_by_id(chunk["id"]) == chunk for chunk in report["retrieved_chunks"]), "True retrieved chunks")
                if no_hits:
                    _check(report["retrieved_chunks"] == [] and report["citations"] == [], "No fake references")
                    _check("资料不足" in report["final_answer"], "Missing reference disclosure")
                else:
                    _check(bool(report["citations"]), "Actual references used")
                if name == "替代资料":
                    _check(report["profile"] == reports["问卷"]["profile"], "CSV unaffected by materials")
                    _check(any(chunk["name"] == "requirements_alternative.md" for chunk in report["retrieved_chunks"]), "Alternative requirement retrieved")
                report["mode"] = "offline"
                saved = save_run(report, output_dir, with_brief=True)
                _check_saved(report, saved)
                reports[name] = report
                saved_reports.append((report, saved))
                pass_check(stage + "：真实出处与简报")

            stage = "资料可用但任务不需要时零检索"
            survey_csv = _project_path("examples/data/survey.csv")
            paths, _ = selected_index(survey_materials)
            model = OfflineIntakeModel()
            skipped = run_intake("总结 CSV 行列规模和缺失情况", survey_csv, model, material_paths=paths)
            _check(skipped["status"] == "completed" and skipped["retrieval_status"] == "not_used", "Optional retrieval")
            _check(skipped["tool_counts"]["search_materials"] == 0 and not skipped["materials"], "No lazy index construction")
            _check([entry["name"] for entry in skipped["execution_ledger"]] == ["profile_csv"], "No actual search")
            _check_tool_evidence(skipped, model)
            pass_check(stage)

            stage = "奖牌任务真正暂停等待口径"
            medal_csv = _project_path("examples/data/medals.csv")
            all_inputs.update(_hashes([medal_csv]))
            model = OfflineIntakeModel()
            session = create_session(medal_csv, model)
            pending = session.invoke("统计获得最多奖牌的前 5 个国家")
            _check(pending["status"] == "awaiting_input" and pending["question"], "Real interrupt")
            _check(pending["tool_counts"]["ask_user"] == 1 and not pending["analysis_results"], "No pretend computation")
            _check(pending["tool_counts"]["profile_csv"] == 1, "Actual profile once")
            pending_path = save_run(pending, output_dir)
            _check(not pending_path.with_suffix(".md").exists(), "Pending does not get a successful brief")
            pass_check(stage)

            stage = "回答后真正计算、恢复不重复扣预算"
            complete = session.resume("全部年份、按 Total 累计、保留原始 NOC")
            _check(complete["status"] == "completed", "Resumed computation completed")
            _check(complete["session_id"] == pending["session_id"] and complete["task_id"] == pending["task_id"], "Same suspended task")
            _check(complete["tool_counts"]["ask_user"] == complete["tool_counts"]["profile_csv"] == 1, "Replay deduplication")
            _check(complete["model_calls"] > pending["model_calls"], "Model budget retained on resume")
            _check([entry["name"] for entry in complete["execution_ledger"]] == ["profile_csv", "ask_user", "analyze_csv"], "Actually analyzed after reply")
            table = complete["analysis_results"][0]
            _check(table["columns"] == ["column_2", "total_medals"], "Actual positional grouping")
            _check(table["rows"] == [{"column_2": name, "total_medals": total}
                                     for name, total in [("Alpha", 22), ("Beta", 17), ("Gamma", 11), ("Delta", 7), ("Epsilon", 5)]], "Hand-computed all-years ranking")
            _check(table["spec"]["filters"] == [] and table["spec"]["group_by"] == [2], "All-years original labels")
            _check_tool_evidence(complete, model)
            saved = save_run(complete, output_dir, with_brief=True)
            saved_reports.append((complete, saved))
            pass_check(stage)

            stage = "继续追问 2024 年并重新计算"
            followup = session.invoke("改为 2024 年")
            _check(followup["status"] == "completed", "Follow-up completed")
            _check(followup["session_id"] == complete["session_id"] and followup["task_id"] != complete["task_id"], "Same session, new task")
            table = followup["analysis_results"][0]
            _check(table["rows"] == [{"column_2": name, "total_medals": total}
                                     for name, total in [("Alpha", 12), ("Beta", 9), ("Gamma", 6), ("Delta", 4), ("Epsilon", 3)]], "Hand-computed 2024 ranking")
            _check(table["spec"]["filters"] == [{"column": 7, "op": "eq", "value": 2024}], "Actual year filter")
            _check(followup["tool_counts"]["ask_user"] == 0 and followup["tool_counts"]["analyze_csv"] == 1, "New task budget, no repeated question")
            _check_tool_evidence(followup, model)
            saved = save_run(followup, output_dir, with_brief=True)
            saved_reports.append((followup, saved))
            pass_check(stage)

            stage = "JSON、Markdown、结果 CSV 与实际工具表一致"
            for report, saved in saved_reports:
                _check_saved(report, saved)
            pass_check(stage)
            stage = "全部源数据与资料哈希不变"
            _check(_hashes(all_inputs) == all_inputs, "All sources remain read-only")
            pass_check(stage)
            stage = "全部真实工具流程零网络请求"
            _check(network_attempts == [], "No network attempts")
            pass_check(stage)
            _check(passed == EXPECTED_CHECKS, "All intended checks completed")
    except Exception as exc:
        print(f"[FAIL] {stage}（{type(exc).__name__}）", file=sys.stderr)
        print(f"Passed: {passed}/{EXPECTED_CHECKS}")
        if output_dir is not None and output_dir.exists():
            print(f"已保存的运行记录目录：{output_dir}")
        return 1
    print(f"Passed: {passed}/{EXPECTED_CHECKS}")
    print(f"运行记录目录：{output_dir}")
    print("离线检查验证实际成果与执行流程；真实模型的理解质量需要另行验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
