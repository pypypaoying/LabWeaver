"""End-to-end chart delegation, conversation, source integrity and export."""

import copy
import csv
import hashlib
import json
import socket
from pathlib import Path

import pytest

from labweaver.agent import create_session
from labweaver.offline import OfflineIntakeModel, OfflineVisualizationModel
from labweaver.runtime import records
from langchain_core.messages import ToolMessage


def test_missing_required_chart_tool_gets_one_real_budgeted_correction(tmp_path):
    from langchain_core.messages import AIMessage
    from test_agent_rag import RagScriptedModel, _call, _answer
    source = tmp_path / "data.csv"
    source.write_text("group,score\nA,2\nB,4\n", encoding="utf-8")
    spec = {"filters": [], "group_by": [1], "metrics": [{"op": "sum", "column": 2, "alias": "total"}],
            "order_by": [], "top_k": 100}
    # Register IDs are intentionally discovered from real ToolMessages.
    class SkippingModel(RagScriptedModel):
        def _generate(self, messages, **kwargs):
            if self._cursor == 4:
                data = next(json.loads(message.content) for message in reversed(messages)
                            if isinstance(message, ToolMessage) and message.name == "analyze_csv")
                self.responses[4] = _call("delegate_visualization", "delegate-1",
                                          {"data_id": data["data_id"], "instruction": "绘制柱状图"})
            return super()._generate(messages, **kwargs)
    model = SkippingModel(responses=[_call(), AIMessage(content="Tools unavailable."),
                                    _call("analyze_csv", "analysis-1", {"spec": spec}),
                                    AIMessage(content="Cannot plot."), AIMessage(content="placeholder"),
                                    _answer("已完成统计与绘图。")])
    report = create_session(source, model, visualization_model=OfflineVisualizationModel()).invoke("按 group 汇总 score 并绘制柱状图")
    assert report["status"] == "completed", report
    assert report["model_calls"] == 6
    assert report["tool_counts"]["analyze_csv"] == report["tool_counts"]["delegate_visualization"] == 1
    paired(report)


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    def deny(*args, **kwargs):
        raise AssertionError("Offline chart acceptance contacted the network")
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)


def paired(report):
    calls = {item["id"]: item for item in report["trace"] if item["kind"] == "tool_call"}
    returned = {item["tool_call_id"]: item for item in report["trace"] if item["kind"] == "tool_result"}
    ledger = {item["tool_call_id"]: item for item in report["execution_ledger"]}
    assert set(calls) == set(returned) == set(ledger)
    for cid, call in calls.items():
        assert call["name"] == returned[cid]["name"] == ledger[cid]["name"]
        assert json.loads(returned[cid]["content"]) == ledger[cid]["result"]


@pytest.mark.parametrize("content,task,chart_type,expected", [
    ("group,score\nA,0\nB,1\nA,1\nB,2\nB,3\nA,\n", "按 group 计算 score 均值并绘制柱状图", "bar",
     [{"column_1": "B", "value": 2}, {"column_1": "A", "value": 0.5}]),
    ("month,revenue\n2024-03,2\n2024-01,3\n2024-02,4\n2024-02,1\n", "按 month 汇总 revenue 并绘制折线图", "line",
     [{"column_1": "2024-01", "value": 3}, {"column_1": "2024-02", "value": 5}, {"column_1": "2024-03", "value": 2}]),
    ("group,score\nA,0\nB,1\nA,1\nB,2\nB,3\nA,\n", "绘制 score 的直方图，使用3箱", "histogram",
     [{"bin_left": 0, "bin_right": 1, "count": 1}, {"bin_left": 1, "bin_right": 2, "count": 2}, {"bin_left": 2, "bin_right": 3, "count": 2}]),
])
def test_three_real_charts_and_exclusive_artifacts(tmp_path, content, task, chart_type, expected):
    source = tmp_path / "selected.csv"
    source.write_text(content, encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    child = OfflineVisualizationModel()
    session = create_session(source, OfflineIntakeModel(), visualization_model=child)
    report = session.invoke(task)
    assert report["status"] == "completed", report
    assert report["visualization_status"] == "completed"
    assert report["tool_counts"]["delegate_visualization"] == 1
    assert report["visualization_model_calls"] == 3
    assert report["charts"][0]["spec"]["chart_type"] == chart_type
    assert report["charts"][0]["rows"] == expected
    assert report["visualization_runs"][0]["tool_exposure"][0] == ["get_plot_data"]
    assert all(set(names) <= {"get_plot_data", "render_chart"} for names in child.bound_tool_sets)
    if chart_type == "histogram":
        assert report["tool_counts"]["analyze_csv"] == 0
        assert report["distribution_results"][0]["valid_count"] == 5
        assert report["distribution_results"][0]["missing_count"] == 1
    paired(report)
    paired(report["visualization_runs"][0])
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()
    saved = session.save(tmp_path / "runs")
    files = saved["chart_paths"][0]
    assert Path(files["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert b"<svg" in Path(files["svg"]).read_bytes()
    with Path(files["data_csv"]).open(encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream))
    assert data == [{key: str(value) for key, value in row.items()} for row in expected]
    assert Path(files["png"]).name in Path(saved["brief_path"]).read_text(encoding="utf-8")
    json.dumps(report, allow_nan=False)
    assert "iVBOR" not in json.dumps(report)
    again = session.save(tmp_path / "runs")
    assert again["run_id"] != saved["run_id"]
    assert Path(files["png"]).exists()


def test_simple_summary_has_no_child_calls(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("category,value\nA,2\nB,3\n", encoding="utf-8")
    report = create_session(source, OfflineIntakeModel(), visualization_model=OfflineVisualizationModel()).invoke("只总结规模与字段，不要图表")
    assert report["status"] == "completed"
    assert report["visualization_status"] == "not_needed"
    assert report["visualization_model_calls"] == 0
    assert report["visualization_runs"] == report["charts"] == []


def test_retrieval_and_chart_both_use_real_evidence(tmp_path):
    source, material = tmp_path / "data.csv", tmp_path / "variables.md"
    source.write_text("group,score\nA,2\nB,4\nA,6\n", encoding="utf-8")
    material.write_text("score 表示评分。按 group 分组比较 score 均值。", encoding="utf-8")
    before = material.read_bytes()
    report = create_session(source, OfflineIntakeModel(), material_paths=[material], visualization_model=OfflineVisualizationModel()).invoke("根据资料比较 group 的 score 均值并绘制柱状图")
    assert report["status"] == "completed", report
    assert report["tool_counts"]["search_materials"] == 1
    assert report["citations"] and report["charts"]
    assert material.read_bytes() == before


def test_clarification_resume_and_style_followup_keep_same_graph(tmp_path):
    source = Path("tests/fixtures/medals.csv")
    session = create_session(source, OfflineIntakeModel(), visualization_model=OfflineVisualizationModel())
    graph = session.graph
    pending = session.invoke("统计奖牌最多的前5个国家并绘制柱状图")
    assert pending["status"] == "awaiting_input"
    assert pending["visualization_model_calls"] == 0
    completed = session.resume("全部年份、按 Total 累计、保留原始 NOC")
    assert completed["status"] == "completed", completed
    assert completed["tool_counts"]["ask_user"] == completed["tool_counts"]["delegate_visualization"] == 1
    assert completed["visualization_model_calls"] == 3
    original_data_id = completed["charts"][0]["data_id"]
    changed = session.invoke("改为横向柱状图")
    assert session.graph is graph and changed["status"] == "completed", changed
    assert changed["tool_counts"]["profile_csv"] == changed["tool_counts"]["analyze_csv"] == 0
    assert changed["charts"][0]["data_id"] == original_data_id
    assert changed["charts"][0]["spec"]["orientation"] == "horizontal"
    assert changed["charts"][0]["task_id"] == changed["task_id"] != completed["task_id"]
    narrowed = session.invoke("改为2024年并绘制柱状图")
    assert narrowed["status"] == "completed", narrowed
    assert narrowed["tool_counts"]["analyze_csv"] == 1
    assert narrowed["charts"][0]["data_id"] != original_data_id
    assert narrowed["analysis_results"][0]["rows"][0]["total_medals"] == 12


def test_chart_storage_failure_rolls_back_new_files(tmp_path, monkeypatch):
    source = tmp_path / "data.csv"
    source.write_text("category,value\nA,2\nB,3\n", encoding="utf-8")
    session = create_session(source, OfflineIntakeModel(), visualization_model=OfflineVisualizationModel())
    report = session.invoke("按 category 汇总 value 并绘制柱状图")
    assert report["status"] == "completed", report
    original = records._write_binary_exclusive
    def fail_svg(path, payload):
        if path.suffix == ".svg":
            raise OSError("synthetic storage failure")
        return original(path, payload)
    monkeypatch.setattr(records, "_write_binary_exclusive", fail_svg)
    with pytest.raises(OSError):
        session.save(tmp_path / "runs")
    assert not list((tmp_path / "runs").rglob("*.*"))


def test_corrupted_or_missing_chart_assets_rejected_before_writes(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("category,value\nA,2\nB,3\n", encoding="utf-8")
    session = create_session(source, OfflineIntakeModel(), visualization_model=OfflineVisualizationModel())
    report = session.invoke("按 category 汇总 value 并绘制柱状图")
    assert report["status"] == "completed", report
    with pytest.raises(ValueError, match="registered"):
        records.save_run(report, tmp_path / "absent", chart_assets={})
    assets = copy.deepcopy(session.chart_assets)
    cid = report["charts"][0]["chart_id"]
    assets[cid]["png"] += b"corrupted"
    with pytest.raises(ValueError, match="hash"):
        records.save_run(report, tmp_path / "invalid", chart_assets=assets)
    assert not (tmp_path / "absent").exists() and not (tmp_path / "invalid").exists()


def test_second_chart_registration_failure_discards_entire_delegation(tmp_path):
    class TwoChartModel(OfflineVisualizationModel):
        def _generate(self, messages, **kwargs):
            rendered = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == "render_chart"]
            if len(rendered) == 1:
                request = json.loads(next(m.content for m in messages if m.type == "human"))
                return self._call("render_chart", {"data_id": request["data_id"], "spec": {"chart_type": "bar", "title": "Second chart"}})
            return super()._generate(messages, **kwargs)
    source = tmp_path / "data.csv"
    source.write_text("category,value\nA,2\nB,3\n", encoding="utf-8")
    session = create_session(source, OfflineIntakeModel(), visualization_model=TwoChartModel())
    register, calls = session.visualizer.register_asset, []
    def fail_second(asset):
        calls.append(asset["metadata"]["chart_id"])
        if len(calls) == 2:
            raise ValueError("synthetic second-registration failure")
        register(asset)
    session.visualizer.register_asset = fail_second
    report = session.invoke("按 category 汇总 value 并绘制柱状图")
    assert len(calls) == 2
    assert report["status"] == "error" and report["visualization_status"] == "error"
    assert report["charts"] == [] and session.chart_assets == {}
    assert session._pending_chart_assets == {}
    saved = session.save(tmp_path / "runs")
    assert saved["analysis_results"] and saved["result_csv_paths"]
    assert not list((tmp_path / "runs").rglob("*.png"))


def test_same_model_call_id_can_be_reused_in_a_new_task(tmp_path):
    from test_agent import ScriptedModel, answer, call
    source = tmp_path / "data.csv"
    source.write_text("category,value\nA,2\nB,3\n", encoding="utf-8")
    spec = {"filters": [], "group_by": [1], "metrics": [{"op": "sum", "column": 2, "alias": "total"}], "order_by": [{"field": "total", "direction": "desc"}], "top_k": 100}
    model = ScriptedModel(responses=[call(), call("analyze_csv", "same-analysis-id", {"spec": spec}), answer("calculation"),
                                     call("analyze_csv", "same-analysis-id", {"spec": spec}), answer("calculation")])
    session = create_session(source, model)
    first = session.invoke("按 category 汇总 value")
    second = session.invoke("再按同一口径汇总 value")
    assert first["status"] == second["status"] == "completed", second
    assert first["task_id"] != second["task_id"]
    assert second["tool_counts"]["analyze_csv"] == 1 and second["model_calls"] == 2
    assert first["analysis_results"][0]["data_id"] != second["analysis_results"][0]["data_id"]
    paired(second)
