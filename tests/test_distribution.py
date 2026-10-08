"""Hand-counted intervals from full snapshots and shared filter semantics."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from labweaver.tools.distribution import prepare_distribution


def snapshot(rows, headers=("value", "group")):
    return SimpleNamespace(source={"name": "synthetic.csv", "sha256": "a" * 64},
                           headers=headers, rows=tuple(tuple(row) for row in rows))


def test_equal_width_boundary_counts_and_full_snapshot():
    data = snapshot([[str(value), "a"] for value in range(11)] + [[" ", "a"]])
    result = prepare_distribution(data, 1, bins=5)
    assert result["status"] == "completed"
    assert result["edges"] == [0, 2, 4, 6, 8, 10]
    assert [row["count"] for row in result["rows"]] == [2, 2, 2, 2, 3]
    assert result["valid_count"] == 11
    assert result["missing_count"] == 1
    assert result["filtered_row_count"] == result["input_row_count"] == 12
    assert result["group_count"] == result["returned_row_count"] == 5
    json.loads(json.dumps(result, allow_nan=False))


def test_exact_decimal_bin_assignment_and_numeric_filter():
    data = snapshot([["0.1", "a"], ["0.2", "a"], ["0.3", "a"], ["0.4", "b"], ["dirty", "c"]])
    result = prepare_distribution(data, 1, [{"column": 2, "op": "eq", "value": "a"}], 2)
    assert result["edges"] == [0.1, 0.2, 0.3]
    assert result["rows"] == [{"bin_left": 0.1, "bin_right": 0.2, "count": 1},
                              {"bin_left": 0.2, "bin_right": 0.3, "count": 2}]
    assert result["filtered_row_count"] == 3
    result = prepare_distribution(data, 1, [{"column": 1, "op": "in", "value": [0.1, 0.3]}], 1)
    assert result["valid_count"] == 2


def test_constant_column_expands_by_half_unit():
    result = prepare_distribution(snapshot([["4", "a"], ["4", "a"]]), 1, bins=2)
    assert result["edges"] == [3.5, 4, 4.5]
    assert [row["count"] for row in result["rows"]] == [0, 2]
    assert "0.5" in result["semantics"]["constant"]


@pytest.mark.parametrize("column,bins,filters", [
    (True, 10, None), (0, 10, None), (3, 10, None), (1, True, None),
    (1, 0, None), (1, 51, None), (1, 1.5, None), (1, 10, {}),
    (1, 10, [{"column": 1, "op": "code", "value": "x"}]),
])
def test_invalid_specs_produce_no_partial_table(column, bins, filters):
    result = prepare_distribution(snapshot([["1", "a"]]), column, filters, bins)
    assert result["status"] == "error"
    assert "rows" not in result


@pytest.mark.parametrize("cell", ["bad", "NA", "NULL", "nan", "inf", "-Infinity", "1e5000"])
def test_invalid_nonmissing_value_rejects_entire_distribution(cell):
    result = prepare_distribution(snapshot([["1", "a"], [cell, "a"]]), 1)
    assert result["status"] == "error"
    assert "rows" not in result


def test_empty_selection_missing_only_and_numeric_precision_collapse():
    assert prepare_distribution(snapshot([]), 1)["error"]["code"] == "empty_selection"
    assert prepare_distribution(snapshot([["", "a"], [" ", "a"]]), 1)["error"]["code"] == "no_valid_values"
    assert prepare_distribution(snapshot([["0.10000000000000000001", "a"],
                                           ["0.10000000000000000002", "a"]]), 1)["error"]["code"] == "numeric_out_of_range"


def test_source_is_read_once_and_never_modified(tmp_path):
    from labweaver.tools.csv_profile import load_csv_snapshot

    path = tmp_path / "source.csv"
    path.write_text("value,group\n0,a\n1,a\n2,b\n3,b\n", encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    data = load_csv_snapshot(path)
    result = prepare_distribution(data, 1, bins=3)
    assert result["source"]["sha256"] == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    path.unlink()
    assert prepare_distribution(data, 1, bins=3)["rows"] == result["rows"]
