"""Task-level, hand-calculated acceptance tests for read-only aggregates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from labweaver.tools.csv_analysis import analyze_csv


def snapshot(headers, rows):
    """Use the documented snapshot interface without requiring disk parsing."""
    return SimpleNamespace(source={"name": "synthetic.csv", "sha256": "a" * 64},
                           headers=tuple(headers), rows=tuple(tuple(row) for row in rows), profile={})


def spec(*, groups=None, metrics=None, filters=None, order=None, top_k=None):
    return {"group_by": [] if groups is None else groups,
            "metrics": [{"op": "count", "column": None, "alias": "rows"}] if metrics is None else metrics,
            "filters": [] if filters is None else filters,
            "order_by": [] if order is None else order, "top_k": top_k}


def result(data, request):
    output = analyze_csv(data, request)
    json.loads(json.dumps(output, allow_nan=False))
    assert output["status"] == "completed", output
    return output


def test_real_task_synthetic_medal_ranking_and_year_followup():
    import csv
    path = Path(__file__).parents[1] / "tests/fixtures/medals.csv"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        data = snapshot(next(reader), list(reader))
    data.source = {"name": path.name, "sha256": before}
    request = spec(groups=[2], metrics=[{"op": "sum", "column": 6, "alias": "medals"}], top_k=5)
    all_years = result(data, request)
    assert all_years["rows"] == [{"column_2": name, "medals": total} for name, total in
                                  [("Alpha", 22), ("Beta", 17), ("Gamma", 11), ("Delta", 7), ("Epsilon", 5)]]
    assert all_years["input_row_count"] == all_years["filtered_row_count"] == 12
    assert all_years["group_count"] == 6
    assert all_years["returned_row_count"] == 5
    assert all_years["truncated"]
    request["filters"] = [{"column": 7, "op": "eq", "value": 2024}]
    year_2024 = result(data, request)
    assert year_2024["rows"] == [{"column_2": name, "medals": total} for name, total in
                                   [("Alpha", 12), ("Beta", 9), ("Gamma", 6), ("Delta", 4), ("Epsilon", 3)]]
    assert year_2024["filtered_row_count"] == 6
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_integer_sum_and_count_are_exact_past_float_precision():
    data = snapshot(["group", "number"], [["a", "9007199254740993"], ["a", "9007199254740994"]])
    output = result(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "sum"},
                                                  {"op": "count", "column": None, "alias": "count"}]))
    assert output["rows"] == [{"column_1": "a", "sum": 18014398509481987, "count": 2}]
    assert type(output["rows"][0]["sum"]) is int


def test_all_aggregate_operations_missing_values_and_original_labels():
    data = snapshot(["name", "value"], [["001", "1"], ["001", "3"], ["001", " "], ["NA", "0"], ["NULL", ""]])
    metrics = [{"op": operation, "column": 2, "alias": operation} for operation in ("count", "sum", "mean", "min", "max")]
    metrics.append({"op": "count", "column": None, "alias": "records"})
    output = result(data, spec(groups=[1], metrics=metrics))
    assert output["rows"] == [
        {"column_1": "001", "count": 2, "sum": 4, "mean": 2, "min": 1, "max": 3, "records": 3},
        {"column_1": "NA", "count": 1, "sum": 0, "mean": 0, "min": 0, "max": 0, "records": 1},
        {"column_1": "NULL", "count": 0, "sum": None, "mean": None, "min": None, "max": None, "records": 1},
    ]


def test_decimal_sum_exact_before_json_conversion_and_default_min_order():
    data = snapshot(["group", "value"], [["a", "0.1"], ["a", "0.2"], ["b", "0.5"]])
    output = result(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "total"}]))
    assert output["rows"] == [{"column_1": "b", "total": 0.5}, {"column_1": "a", "total": 0.3}]
    minimum = result(data, spec(groups=[1], metrics=[{"op": "min", "column": 2, "alias": "lowest"}]))
    assert minimum["rows"][0] == {"column_1": "a", "lowest": 0.1}


def test_ranking_compares_exact_decimals_before_json_float_rounding():
    data = snapshot(["group", "value"], [["a", "0.10000000000000000001"], ["z", "0.10000000000000000002"]])
    output = result(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "total"}]))
    assert [row["column_1"] for row in output["rows"]] == ["z", "a"]


def test_analysis_uses_the_existing_snapshot_and_never_reopens_source(tmp_path):
    from labweaver.tools.csv_profile import load_csv_snapshot
    path = tmp_path / "input.csv"
    path.write_bytes("name;total\n中国;2\n中国;3\n法国;4\n".encode("utf-8"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    data = load_csv_snapshot(path)
    assert data.profile["row_count"] == 3
    assert data.profile["parsing"]["delimiter"] == ";"
    output = result(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "total"}]))
    assert output["rows"] == [{"column_1": "中国", "total": 5}, {"column_1": "法国", "total": 4}]
    assert output["source"]["sha256"] == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # Statistics remain bound to the parsed snapshot even if the selected
    # source subsequently disappears; the tool has no path argument or I/O.
    path.unlink()
    assert result(data, spec())["rows"] == [{"rows": 3}]


def test_position_addressing_handles_duplicate_and_empty_headers():
    data = snapshot(["score", "score", ""], [["x", "001", "2"], ["x", "002", "3"]])
    output = result(data, spec(groups=[1, 2], metrics=[{"op": "sum", "column": 3, "alias": "score"}]))
    assert output["columns"] == ["column_1", "column_2", "score"]
    assert output["group_columns"] == [{"field": "column_1", "position": 1, "name": "score"},
                                       {"field": "column_2", "position": 2, "name": "score"}]
    assert output["rows"] == [{"column_1": "x", "column_2": "002", "score": 3},
                              {"column_1": "x", "column_2": "001", "score": 2}]


@pytest.mark.parametrize("operation,value,expected", [
    ("eq", "01", 1), ("eq", 1, 2), ("ne", "01", 4),
    ("gt", 1, 1), ("gte", 1, 3), ("lt", 2, 2), ("lte", 2, 3),
    ("in", ["01", "2"], 2), ("not_in", ["01", "2"], 3),
])
def test_filter_operations(operation, value, expected):
    data = snapshot(["value"], [["01"], ["1"], ["2"], [""], [" "]])
    output = result(data, spec(filters=[{"column": 1, "op": operation, "value": value}]))
    assert output["rows"] == [{"rows": expected}]


def test_missing_filters_and_filters_exclude_invalid_numeric_before_aggregation():
    data = snapshot(["value", "keep"], [["2", "yes"], ["NA", "no"], ["", "yes"], [" ", "no"]])
    output = result(data, spec(filters=[{"column": 2, "op": "eq", "value": "yes"}],
                               metrics=[{"op": "sum", "column": 1, "alias": "sum"}]))
    assert output["rows"] == [{"sum": 2}]
    for operation, expected in [("is_missing", 2), ("not_missing", 2)]:
        output = result(data, spec(filters=[{"column": 1, "op": operation}]))
        assert output["rows"] == [{"rows": expected}]


@pytest.mark.parametrize("token", ["NA", "NULL", "NaN", "inf", "-Infinity", "oops"])
def test_invalid_unfiltered_numeric_values_fail_without_partial_table(token):
    data = snapshot(["group", "value"], [["a", "2"], ["b", token]])
    output = analyze_csv(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "sum"}]))
    assert output["status"] == "error"
    assert output["error"]["code"] == "invalid_numeric_value"
    assert "rows" not in output
    assert "Record 2, column 2" in output["error"]["message"]


def test_ties_preserve_historical_labels_and_have_stable_exact_top_k():
    data = snapshot(["country", "total"], [["USSR", "10"], ["United States", "10"], ["Soviet Union", "10"], ["China", "10"]])
    output = result(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "total"}], top_k=2))
    assert output["rows"] == [{"column_1": "China", "total": 10}, {"column_1": "Soviet Union", "total": 10}]
    assert output["group_count"] == 4
    assert output["truncated"]


def test_explicit_multiple_ordering_fields_nulls_last_in_both_directions():
    data = snapshot(["group", "value"], [["b", "2"], ["a", "2"], ["c", " "]])
    for direction in ("asc", "desc"):
        output = result(data, spec(groups=[1], metrics=[{"op": "sum", "column": 2, "alias": "total"}],
                                   order=[{"field": "total", "direction": direction}, {"field": "column_1", "direction": "desc"}]))
        assert [row["column_1"] for row in output["rows"]] == ["b", "a", "c"]


def test_blanks_share_null_group_while_labels_with_spaces_are_preserved():
    data = snapshot(["group"], [[""], [" "], [" a "], ["a"]])
    output = result(data, spec(groups=[1]))
    assert output["rows"] == [{"column_1": None, "rows": 2}, {"column_1": " a ", "rows": 1}, {"column_1": "a", "rows": 1}]


def test_no_matching_rows_and_header_only_inputs_return_explicit_empty_statistics():
    for data in [snapshot(["value"], []), snapshot(["value"], [["1"]])]:
        request = spec(filters=[{"column": 1, "op": "eq", "value": "never"}],
                       metrics=[{"op": "count", "column": None, "alias": "count"}, {"op": "sum", "column": 1, "alias": "sum"}])
        output = result(data, request)
        assert output["rows"] == [{"count": 0, "sum": None}]
        assert output["filtered_row_count"] == 0
        request["group_by"] = [1]
        assert result(data, request)["rows"] == []


def test_requested_large_top_k_is_not_capped_and_source_metadata_is_copied():
    data = snapshot(["group"], [[f"{index:03}"] for index in range(280)])
    output = result(data, spec(groups=[1], top_k=500))
    assert output["returned_row_count"] == 280
    assert output["requested_top_k"] == 500
    assert output["spec"]["top_k"] == 500
    assert not output["limit_applied"] and not output["truncated"]
    assert output["result_limit"] is None
    output["source"]["name"] = "changed"
    assert data.source["name"] == "synthetic.csv"


@pytest.mark.parametrize("top_k", [None, 1_000_001])
def test_all_groups_are_retained_without_an_artificial_cap(top_k):
    data = snapshot(["group"], [[f"{index:04}"] for index in range(1500)])
    request = spec(groups=[1], top_k=top_k)
    if top_k is None:
        request.pop("top_k")
    output = result(data, request)
    assert output["returned_row_count"] == 1500
    assert not output["truncated"]


@pytest.mark.parametrize("invalid_request", [
    None, [], {"execute": "print(1)"}, {"metrics": []}, {"metrics": "sum"},
    {"metrics": [{"op": [], "alias": "a"}]}, {"metrics": [{"op": "sum", "column": 1, "alias": " "}]},
    {"metrics": [{"op": "sum", "column": 1, "alias": "x"}, {"op": "count", "alias": "x"}]},
    {"metrics": [{"op": "sum", "alias": "a"}]}, {"metrics": [{"op": "count", "column": False, "alias": "a"}]},
    {"metrics": [{"op": "count", "alias": "a"}], "group_by": [1, 1]},
    {"metrics": [{"op": "count", "alias": "column_1"}], "group_by": [1]},
    {"metrics": [{"op": "count", "alias": "a"}], "top_k": 0},
    {"metrics": [{"op": "count", "alias": "a"}], "top_k": True},
    {"metrics": [{"op": "count", "alias": "a"}], "order_by": [{"field": "a", "direction": []}]},
    {"metrics": [{"op": "count", "alias": "a"}], "order_by": [{"field": "unknown"}]},
    {"metrics": [{"op": "count", "alias": "a"}], "filters": [{"column": 1, "op": []}]},
    {"metrics": [{"op": "count", "alias": "a"}], "filters": [{"column": 1, "op": "in", "value": "a"}]},
    {"metrics": [{"op": "count", "alias": "a"}], "filters": [{"column": 1, "op": "eq", "value": float("nan")}]},
    {"metrics": [{"op": "count", "alias": "a"}], "filters": [{"column": 1, "op": "is_missing", "value": ""}]},
])
def test_invalid_requests_are_structured_errors(invalid_request):
    output = analyze_csv(snapshot(["value"], [["1"]]), invalid_request)
    assert output["status"] == "error"
    assert output["error"]["code"] == "invalid_spec"
    assert "rows" not in output
    json.loads(json.dumps(output, allow_nan=False))


@pytest.mark.parametrize("token", ["1e999999999", "1e-999999999"])
def test_adversarial_exponents_fail_before_integer_expansion(token):
    output = analyze_csv(snapshot(["value"], [[token]]), spec(metrics=[{"op": "sum", "column": 1, "alias": "sum"}]))
    assert output["error"]["code"] == "numeric_out_of_range"


def test_noninteger_underflow_and_overflow_are_not_emitted_as_zero_or_infinity():
    for token in ("1e-400", "1e400"):
        # A huge integral sum remains exact; mean makes this nonintegral.
        data = snapshot(["value"], [[token], ["0"], ["0"]])
        output = analyze_csv(data, spec(metrics=[{"op": "mean", "column": 1, "alias": "mean"}]))
        assert output["error"]["code"] == "numeric_out_of_range"
