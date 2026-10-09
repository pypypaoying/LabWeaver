"""Host-data chart templates, exported bytes and trustworthy plotting tables."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import struct
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from labweaver.tools.csv_analysis import analyze_csv
from labweaver.tools.distribution import prepare_distribution
from labweaver.tools.visualization import PlotSpec, render_chart


def aggregate(rows, *, top_k=None):
    snapshot = SimpleNamespace(source={"name": "synthetic.csv", "sha256": "a" * 64},
                               headers=("group", "number"), rows=tuple(tuple(row) for row in rows))
    result = analyze_csv(snapshot, {"group_by": [1], "metrics": [{"op": "sum", "column": 2, "alias": "total"}],
                                    "order_by": [{"field": "total", "direction": "desc"}], "top_k": top_k})
    result.update(kind="aggregate", data_id="data_synthetic", session_id="session", task_id="task")
    return result


def distribution(values, bins=3):
    snapshot = SimpleNamespace(source={"name": "synthetic.csv", "sha256": "a" * 64},
                               headers=("measurement",), rows=tuple((str(value),) for value in values))
    result = prepare_distribution(snapshot, 1, bins=bins)
    result.update(data_id="data_distribution", session_id="session", task_id="task")
    return result


def assert_asset(result):
    assert result["status"] == "completed", result
    metadata = result["metadata"]
    json.loads(json.dumps(metadata, allow_nan=False))
    assert result["png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", result["png"][16:24]) == (1500, 900)
    svg = ET.fromstring(result["svg"])
    assert svg.tag.endswith("svg")
    assert b"<path" in result["svg"]
    assert b"<text" not in result["svg"]
    for field in ("png", "svg", "data_csv"):
        raw = result[field].encode("utf-8") if isinstance(result[field], str) else result[field]
        assert metadata["hashes"][field] == hashlib.sha256(raw).hexdigest()
    assert list(csv.DictReader(io.StringIO(result["data_csv"])))
    return metadata


def test_bar_preserves_real_order_and_marks_selected_range():
    data = aggregate([["Alpha", "2"], ["Beta", "7"], ["Gamma", "4"]], top_k=2)
    before = copy.deepcopy(data)
    metadata = assert_asset(render_chart(data, {"chart_type": "bar", "orientation": "horizontal"}))
    assert metadata["rows"] == [{"column_1": "Beta", "total": 7}, {"column_1": "Gamma", "total": 4}]
    assert "2 of 3" in metadata["range_note"]
    assert metadata["truncated"]
    assert metadata["source"] == data["source"]
    assert metadata["session_id"] == "session"
    assert data == before


def test_date_line_sorts_months_and_keeps_real_metric_values():
    data = aggregate([["2024-03", "4"], ["2024-01", "8"], ["2024-02", "2"]])
    metadata = assert_asset(render_chart(data, {"chart_type": "line"}))
    assert metadata["x_type"] == "date"
    assert metadata["spec"]["x_type"] == "date"
    assert metadata["rows"] == [{"column_1": "2024-01", "total": 8},
                                {"column_1": "2024-02", "total": 2},
                                {"column_1": "2024-03", "total": 4}]
    result = render_chart(data, {"chart_type": "line"})
    for month in ("2024-01", "2024-02", "2024-03"):
        assert result["svg"].decode("utf-8").count(f"<!-- {month} -->") == 1


def test_bar_preserves_all_280_groups_in_image_and_export():
    data = aggregate([[f"category-{i:03}", str(i)] for i in range(280)])
    asset = render_chart(data, {"chart_type": "bar"})
    metadata = assert_asset(asset)
    assert len(metadata["rows"]) == 280 and not metadata["truncated"]
    assert sum(row["total"] for row in metadata["rows"]) == sum(range(280))
    assert len(list(csv.DictReader(io.StringIO(asset["data_csv"])))) == 280
    assert b'patch_282' in asset["svg"]  # background + 280 bars; thinning labels doesn't remove bars


def test_numeric_line_sorts_raw_numeric_labels():
    data = aggregate([["10", "7"], ["2", "5"], ["1", "2"]])
    metadata = assert_asset(render_chart(data, {"chart_type": "line", "x_type": "numeric"}))
    assert [row["column_1"] for row in metadata["rows"]] == ["1", "2", "10"]


@pytest.mark.parametrize("x_type", ["auto", "date"])
@pytest.mark.parametrize("separator", ["/", "-"])
def test_year_first_line_dates_sort_by_calendar_preserving_raw_data(x_type, separator):
    labels = [value.replace("/", separator) for value in ("2024/12/10", "2024/12/2", "2025/1/1", "2024/12/1")]
    data = aggregate([[labels[0], "4"], [labels[1], "8"], [labels[2], "2"], [labels[3], "6"]])
    before = copy.deepcopy(data)
    result = render_chart(data, {"chart_type": "line", "x_type": x_type})
    metadata = assert_asset(result)
    assert metadata["x_type"] == "date"
    assert metadata["rows"] == [{"column_1": labels[index], "total": value}
                                for index, value in [(3, 6), (1, 8), (0, 4), (2, 2)]]
    assert [row["column_1"] for row in csv.DictReader(io.StringIO(result["data_csv"]))] == [labels[i] for i in (3, 1, 0, 2)]
    assert data == before


@pytest.mark.parametrize("labels,code", [
    (["2024/2/29", "2024/2/30"], "invalid_x_axis"),
    (["2023/2/29", "2023/3/1"], "invalid_x_axis"),
    (["2024/12/1", "2024-12-01"], "duplicate_x_axis"),
    (["2024/12/1", "12/02/2024"], "invalid_x_axis"),
    (["2024/12/1", "2024/12-2"], "invalid_x_axis"),
])
def test_year_first_dates_reject_invalid_ambiguous_or_duplicate_days(labels, code):
    output = render_chart(aggregate([[label, "2"] for label in labels]), {"chart_type": "line"})
    assert output["status"] == "error" and output["error"]["code"] == code
    assert "png" not in output


def test_histogram_uses_registered_hand_counted_intervals():
    data = distribution(range(7), bins=3)
    metadata = assert_asset(render_chart(data, {"chart_type": "histogram"}))
    assert metadata["rows"] == [{"bin_left": 0, "bin_right": 2, "count": 2},
                                {"bin_left": 2, "bin_right": 4, "count": 2},
                                {"bin_left": 4, "bin_right": 6, "count": 3}]
    assert metadata["columns"] == ["bin_left", "bin_right", "count"]


def test_missing_metric_exclusion_is_counted_and_reported():
    data = aggregate([["a", "1"], ["b", ""]])
    metadata = assert_asset(render_chart(data, {"chart_type": "bar"}))
    assert metadata["excluded_missing_count"] == 1
    assert metadata["rows"] == [{"column_1": "a", "total": 1}]
    assert "Excluded 1" in metadata["range_note"]


@pytest.mark.parametrize("spec", [
    {"chart_type": "scatter"}, {"chart_type": "bar", "code": "x"},
    {"chart_type": "bar", "rows": []}, {"chart_type": "bar", "title": "a" * 161},
    {"chart_type": "bar", "title": "bad\nlabel"}, {"chart_type": "line", "orientation": "horizontal"},
    {"chart_type": "bar", "x_type": "date"}, {"chart_type": "bar", "y": "missing"},
])
def test_unsupported_specs_and_model_supplied_data_are_rejected(spec):
    output = render_chart(aggregate([["a", "2"]]), spec)
    assert output["status"] == "error"
    assert "png" not in output


@pytest.mark.parametrize("labels,x_type,code", [
    (["a", "b"], "auto", "numeric_out_of_range"),
    (["2024-02-30", "2024-03-01"], "date", "invalid_x_axis"),
    (["2024-01", "2024-01-01"], "date", "duplicate_x_axis"),
    (["1", "01"], "numeric", "duplicate_x_axis"),
    (["", "2"], "numeric", "invalid_x_axis"),
    (["9007199254740993", "2"], "numeric", "numeric_precision_loss"),
])
def test_invalid_or_duplicate_line_axes_require_recomputation(labels, x_type, code):
    result = render_chart(aggregate([[label, "1"] for label in labels]), {"chart_type": "line", "x_type": x_type})
    assert result["status"] == "error"
    assert result["error"]["code"] == code


@pytest.mark.parametrize("value", [9007199254740993, 10 ** 400, float("inf"), float("nan"), True])
def test_metric_outside_float_range_or_precision_is_rejected(value):
    data = aggregate([["a", "1"]])
    data["rows"][0]["total"] = value
    result = render_chart(data, {"chart_type": "bar"})
    assert result["status"] == "error"
    assert "png" not in result


def test_exact_float_representable_large_integer_is_drawable():
    data = aggregate([["a", str(2 ** 60)]])
    metadata = assert_asset(render_chart(data, {"chart_type": "bar"}))
    assert metadata["rows"][0]["total"] == 2 ** 60


def test_sample_top_k_and_forged_histogram_counts_are_rejected():
    data = aggregate([["a", "1"]])
    assert render_chart(data, {"chart_type": "histogram"})["status"] == "error"
    histogram = distribution(range(7))
    histogram["rows"][0]["count"] += 1
    assert render_chart(histogram, {"chart_type": "histogram"})["status"] == "error"
    histogram = distribution(range(7))
    histogram["truncated"] = True
    assert render_chart(histogram, {"chart_type": "histogram"})["status"] == "error"


def test_empty_result_unregistered_data_and_missing_font(tmp_path):
    data = aggregate([["a", "1"]])
    del data["data_id"]
    assert render_chart(data, {"chart_type": "bar"})["error"]["code"] == "invalid_data"
    data = aggregate([["a", "1"]])
    data["rows"] = []
    assert render_chart(data, {"chart_type": "bar"})["error"]["code"] == "empty_result"
    result = render_chart(aggregate([["a", "1"]]), {"chart_type": "bar"}, font_path=tmp_path / "missing.ttf")
    assert result["error"]["code"] == "font_unavailable"


def test_cjk_is_supported_or_explicitly_rejected_and_override_checks_glyphs():
    from matplotlib import font_manager

    result = render_chart(aggregate([["中国", "2"], ["法国", "3"]]),
                          {"chart_type": "bar", "title": "部门比较", "x_label": "部门", "y_label": "数量"})
    if result["status"] == "completed":
        assert_asset(result)
    else:
        assert result["error"]["code"] == "font_unavailable"
    latin_font = font_manager.findfont("DejaVu Sans")
    result = render_chart(aggregate([["中国", "2"]]), {"chart_type": "bar"}, font_path=latin_font)
    assert result["error"]["code"] == "font_unavailable"


def test_renderer_writes_no_files_and_keeps_source_hash(tmp_path, monkeypatch):
    from labweaver.tools.csv_profile import load_csv_snapshot

    source = tmp_path / "source.csv"
    source.write_text("group,number\na,2\nb,3\n", encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot = load_csv_snapshot(source)
    data = analyze_csv(snapshot, {"group_by": [1], "metrics": [{"op": "sum", "column": 2, "alias": "total"}]})
    data.update(kind="aggregate", data_id="data_source")
    monkeypatch.chdir(tmp_path)
    assert_asset(render_chart(data, {"chart_type": "bar"}))
    assert list(tmp_path.iterdir()) == [source]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_plot_schema_rejects_extra_fields():
    with pytest.raises(ValueError):
        PlotSpec.model_validate({"chart_type": "bar", "file_path": "output.png"})
