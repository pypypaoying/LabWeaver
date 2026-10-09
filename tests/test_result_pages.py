"""Full calculation/export, with bounded and authorized model-only pages."""

import csv
import hashlib
import json

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from labweaver.agent import create_session
from labweaver.tools.result_pages import MAX_PAGE_BYTES, result_page
from test_agent import ScriptedModel, answer, call, paired
from test_csv_analysis import result, snapshot, spec


def table(count=1500):
    data = snapshot(["category", "value"], [[f"category-{i:05d}", str(i)] for i in range(count)])
    output = result(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "total"}]))
    output.update(data_id="data-test", kind="aggregate")
    return output


def test_pages_reassemble_the_exact_full_table_and_do_not_truncate_calculation():
    data = table()
    rows, offset = [], 0
    while offset is not None:
        view = result_page(data, offset, 317)
        assert len(json.dumps(view, ensure_ascii=False).encode("utf-8")) <= MAX_PAGE_BYTES
        assert view["returned_row_count"] == 1500 and not view["truncated"]
        rows.extend(view["rows"])
        offset = view["page"]["next_offset"]
    assert rows == data["rows"]
    view["rows"][0]["total"] = -99
    assert data["rows"][-1]["total"] == 0


def test_default_byte_page_retains_280_small_groups_but_bounds_wide_tables():
    small = table(280)
    assert result_page(small)["rows"] == small["rows"]
    assert not result_page(small)["preview_only"]
    wide = table(280)
    for row in wide["rows"]:
        row["column_1"] *= 100
    view = result_page(wide)
    assert view["preview_only"] and 0 < len(view["rows"]) < 280
    assert view["page"]["next_offset"] == len(view["rows"])
    assert len(json.dumps(view, ensure_ascii=False).encode("utf-8")) <= MAX_PAGE_BYTES


def test_oversized_single_cell_is_explicit_without_a_nonadvancing_cursor():
    data = table(1)
    data["rows"][0]["column_1"] = "中" * 65536
    view = result_page(data)
    assert view["preview_only"] and view["view_error"]
    assert view["available_row_count"] == 1 and not view["rows"]
    assert view["page"]["next_offset"] is None


@pytest.mark.parametrize("offset,limit", [(True, 2), (-1, 2), (4, 2), (0, False), (0, 1001)])
def test_invalid_page_ranges_fail(offset, limit):
    with pytest.raises(ValueError):
        result_page(table(3), offset, limit)


class PagingModel(ScriptedModel):
    def _generate(self, messages, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if results and results[-1].name == "analyze_csv":
            view = json.loads(results[-1].content)
            assert view["preview_only"] and view["page"]["next_offset"]
            message = call("read_analysis_rows", "page1", {"data_id": view["data_id"], "offset": view["page"]["next_offset"], "limit": 1000})
            return ChatResult(generations=[ChatGeneration(message=message)])
        return super()._generate(messages, **kwargs)


def test_real_graph_pages_and_export_preserve_all_groups(tmp_path):
    path = tmp_path / "categories.csv"
    path.write_text("category,value\n" + "".join(f"category-{i:05},1\n" for i in range(1500)), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    request = spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "total"}])
    model = PagingModel(responses=[call(), call("analyze_csv", "a1", {"spec": request}), answer("calculation", "完整分组结果已保存；分页只是查看部分记录。")])
    session = create_session(path, model)
    report = session.invoke("统计每个类别，不截取前几名")
    assert report["status"] == "completed", report.get("error")
    assert report["tool_counts"]["read_analysis_rows"] == 1
    assert len(report["analysis_results"][0]["rows"]) == 1500
    assert sum(row["total"] for row in report["analysis_results"][0]["rows"]) == 1500
    paired(report)
    saved = session.save(tmp_path / "runs")
    with open(saved["result_csv_paths"][0], encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1500 and len({row["column_1"] for row in rows}) == 1500
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_forged_page_id_cannot_read_other_data(tmp_path):
    path = tmp_path / "small.csv"
    path.write_text("category,value\na,1\n", encoding="utf-8")
    model = ScriptedModel(responses=[call(), call("analyze_csv", "a", {"spec": spec(groups=[1])}),
                                    call("read_analysis_rows", "page", {"data_id": "forged"}), answer("calculation")])
    report = create_session(path, model).invoke("统计所有类别")
    page = report["execution_ledger"][-1]["result"]
    assert page["status"] == "error" and page["error"]["code"] == "invalid_result_page"
    assert "rows" not in page
