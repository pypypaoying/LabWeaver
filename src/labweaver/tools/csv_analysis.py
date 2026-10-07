"""Deterministic, bounded statistics over an already parsed CSV snapshot.

The request language is deliberately small: positional filters, grouping and
aggregates. It cannot execute Python, SQL, modify rows, or open another file.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import cmp_to_key
import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .csv_profile import CsvSnapshot


MAX_RESULT_ROWS = 100
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_FILTER_OPERATIONS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "is_missing", "not_missing"}
_AGGREGATES = {"count", "sum", "mean", "min", "max"}
_ALIAS = re.compile(r"[^\x00-\x1f\x7f]{1,80}\Z")


class _AnalysisError(ValueError):
    """A user-readable validation/computation failure, never a partial table."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _position(value: Any, width: int, label: str) -> int:
    """Validate one-based positions, including rejection of bool-as-int."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= width:
        raise _AnalysisError("invalid_spec", f"{label} must be a column position from 1 to {width}.")
    return value


def _numeric(value: str, *, row: int, column: int) -> Fraction:
    """Parse finite decimal notation without first rounding it to a float."""
    text = value.strip()
    if not _NUMBER.fullmatch(text):
        raise _AnalysisError("invalid_numeric_value", f"Record {row}, column {column} contains a nonnumeric or nonfinite value.")
    try:
        number = Decimal(text)
        # Bound expansion of adversarial exponents before creating huge ints.
        # Python's strict JSON encoder cannot emit arbitrarily long integers.
        exponent = number.as_tuple().exponent
        digits = len(number.as_tuple().digits)
        if not number.is_finite() or abs(exponent) > 4096 or digits > 4096 or number.adjusted() > 4090:
            raise _AnalysisError("numeric_out_of_range", f"Record {row}, column {column} exceeds the supported numeric range.")
        return Fraction(number)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        if isinstance(exc, _AnalysisError):
            raise
        raise _AnalysisError("invalid_numeric_value", f"Record {row}, column {column} contains an invalid numeric value.") from exc


def _json_number(value: Fraction) -> int | float:
    """Keep integral results exact; reject overflow and nonzero underflow."""
    if value.denominator == 1:
        # 13,620 bits fit within the default 4,300-decimal-digit JSON limit.
        if value.numerator.bit_length() > 13_620:
            raise _AnalysisError("numeric_out_of_range", "An aggregate is too large for strict JSON serialization.")
        return value.numerator
    try:
        rendered = float(value)
    except OverflowError as exc:
        raise _AnalysisError("numeric_out_of_range", "An aggregate exceeds the finite JSON numeric range.") from exc
    if not math.isfinite(rendered) or rendered == 0.0 and value != 0:
        raise _AnalysisError("numeric_out_of_range", "An aggregate exceeds the finite JSON numeric range.")
    return rendered


def _validate_spec(spec: Any, width: int) -> dict[str, Any]:
    """Reject unknown keys and normalize the entire request before reading rows."""
    if not isinstance(spec, dict):
        raise _AnalysisError("invalid_spec", "spec must be a JSON object.")
    allowed = {"filters", "group_by", "metrics", "order_by", "top_k"}
    if set(spec) - allowed:
        raise _AnalysisError("invalid_spec", "spec contains unsupported fields.")
    normalized: dict[str, Any] = {}
    filters = spec.get("filters", [])
    if not isinstance(filters, list) or len(filters) > 20:
        raise _AnalysisError("invalid_spec", "filters must contain at most 20 filter objects.")
    normalized["filters"] = []
    for item in filters:
        if not isinstance(item, dict) or set(item) - {"column", "op", "value"}:
            raise _AnalysisError("invalid_spec", "Each filter accepts only column, op and value.")
        column = _position(item.get("column"), width, "Filter column")
        operation = item.get("op")
        if not isinstance(operation, str) or operation not in _FILTER_OPERATIONS:
            raise _AnalysisError("invalid_spec", "The requested filter operation is unsupported.")
        current = {"column": column, "op": operation}
        if operation not in {"is_missing", "not_missing"}:
            if "value" not in item:
                raise _AnalysisError("invalid_spec", "This filter requires a value.")
            value = item["value"]
            values = value if operation in {"in", "not_in"} else [value]
            if operation in {"in", "not_in"} and (not isinstance(value, list) or len(value) > 100):
                raise _AnalysisError("invalid_spec", "Set filters require a list with at most 100 values.")
            for member in values:
                if isinstance(member, bool) or not isinstance(member, (str, int, float)):
                    raise _AnalysisError("invalid_spec", "Filter values must be strings or finite numbers.")
                if isinstance(member, float) and not math.isfinite(member):
                    raise _AnalysisError("invalid_spec", "Filter numeric values must be finite.")
                if isinstance(member, str) and len(member) > 65_536:
                    raise _AnalysisError("invalid_spec", "A filter string exceeds the field length limit.")
                if operation in {"gt", "gte", "lt", "lte"}:
                    _numeric(str(member), row=0, column=column)
            current["value"] = list(value) if isinstance(value, list) else value
        elif "value" in item:
            raise _AnalysisError("invalid_spec", "Missing-value filters do not accept a value.")
        normalized["filters"].append(current)

    groups = spec.get("group_by", [])
    if not isinstance(groups, list) or len(groups) > width:
        raise _AnalysisError("invalid_spec", "group_by must be a list of distinct column positions.")
    normalized["group_by"] = [_position(value, width, "Grouping column") for value in groups]
    if len(set(groups)) != len(groups):
        raise _AnalysisError("invalid_spec", "Grouping positions must be distinct.")

    metrics = spec.get("metrics")
    if not isinstance(metrics, list) or not 1 <= len(metrics) <= 20:
        raise _AnalysisError("invalid_spec", "metrics must contain between 1 and 20 aggregate objects.")
    normalized["metrics"] = []
    aliases = {f"column_{value}" for value in groups}
    for item in metrics:
        if not isinstance(item, dict) or set(item) - {"op", "column", "alias"}:
            raise _AnalysisError("invalid_spec", "Each metric accepts only op, column and alias.")
        operation, alias = item.get("op"), item.get("alias")
        if not isinstance(operation, str) or operation not in _AGGREGATES:
            raise _AnalysisError("invalid_spec", "The requested aggregate operation is unsupported.")
        if not isinstance(alias, str) or not alias.strip() or not _ALIAS.fullmatch(alias) or alias in aliases:
            raise _AnalysisError("invalid_spec", "Metric aliases must be unique nonempty labels of at most 80 characters.")
        aliases.add(alias)
        column = item.get("column")
        if column is not None:
            column = _position(column, width, "Metric column")
        elif operation != "count":
            raise _AnalysisError("invalid_spec", "Numeric aggregates require a column position.")
        normalized["metrics"].append({"op": operation, "column": column, "alias": alias})

    order = spec.get("order_by", [])
    if not isinstance(order, list) or len(order) > len(aliases):
        raise _AnalysisError("invalid_spec", "order_by must be a list of result-field ordering objects.")
    normalized["order_by"] = []
    used = set()
    for item in order:
        if not isinstance(item, dict) or set(item) - {"field", "direction"}:
            raise _AnalysisError("invalid_spec", "Each ordering accepts only field and direction.")
        field, direction = item.get("field"), item.get("direction", "desc")
        if not isinstance(field, str) or field not in aliases or field in used or not isinstance(direction, str) or direction not in {"asc", "desc"}:
            raise _AnalysisError("invalid_spec", "Ordering fields must be distinct result labels and directions asc or desc.")
        used.add(field)
        normalized["order_by"].append({"field": field, "direction": direction})
    if not order:
        first = normalized["metrics"][0]
        normalized["order_by"] = [{"field": first["alias"], "direction": "asc" if first["op"] == "min" else "desc"}]

    top_k = spec.get("top_k", MAX_RESULT_ROWS)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 1_000_000:
        raise _AnalysisError("invalid_spec", "top_k must be an integer between 1 and 1000000; output is capped at 100.")
    normalized["top_k"] = min(top_k, MAX_RESULT_ROWS)
    normalized["requested_top_k"] = top_k
    return normalized


def _value_matches(cell: str, expected: str | int | float, *, row: int, column: int) -> bool:
    """Strings match preserved raw cells; JSON numbers match numeric values."""
    if isinstance(expected, str):
        return cell == expected
    if not cell.strip():
        return False
    try:
        return _numeric(cell, row=row, column=column) == _numeric(str(expected), row=0, column=column)
    except _AnalysisError:
        # Equality selection should be able to exclude dirty numeric cells.
        return False


def _matches(row: tuple[str, ...], record: int, filters: list[dict[str, Any]]) -> bool:
    """Apply conjunctions in request order without mutating source cell strings."""
    for item in filters:
        column, operation = item["column"], item["op"]
        cell = row[column - 1]
        missing = not cell.strip()
        if operation == "is_missing":
            passed = missing
        elif operation == "not_missing":
            passed = not missing
        elif operation in {"eq", "ne"}:
            passed = _value_matches(cell, item["value"], row=record, column=column)
            if operation == "ne":
                passed = not passed
        elif operation in {"in", "not_in"}:
            passed = any(_value_matches(cell, value, row=record, column=column) for value in item["value"])
            if operation == "not_in":
                passed = not passed
        elif missing:
            passed = False
        else:
            actual = _numeric(cell, row=record, column=column)
            expected = _numeric(str(item["value"]), row=0, column=column)
            passed = {"gt": actual > expected, "gte": actual >= expected,
                      "lt": actual < expected, "lte": actual <= expected}[operation]
        if not passed:
            return False
    return True


def _compare_rows(left: dict, right: dict, order: list[dict], group_fields: list[str]) -> int:
    """Sort nulls last in both directions, then stabilize ties by raw labels."""
    for item in order:
        a, b = left[item["field"]], right[item["field"]]
        if a is None or b is None:
            if a is None and b is None:
                continue
            return 1 if a is None else -1
        comparison = (a > b) - (a < b)
        if comparison:
            return comparison if item["direction"] == "asc" else -comparison
    for field in group_fields:
        a, b = left[field], right[field]
        if a is None or b is None:
            if a is None and b is None:
                continue
            return 1 if a is None else -1
        comparison = (a > b) - (a < b)
        if comparison:
            return comparison
    return 0


def analyze_csv(snapshot: CsvSnapshot, spec: dict[str, Any]) -> dict[str, Any]:
    """Filter, group and aggregate a bound snapshot with explicit execution data.

    Missing cells are whitespace-only; literal ``0``, ``NA`` and ``NULL`` are
    retained. Group labels preserve nonmissing raw strings, including historic
    names and leading zeros. Blank group cells share a JSON-null group. Numeric
    aggregates exclude missing cells and reject remaining dirty/nonfinite data.
    Filters use AND; string equality is exact, numeric equality is numeric.
    Returned rows are deterministic and capped at 100, with truncation recorded.
    """
    source = dict(snapshot.source)
    try:
        normalized = _validate_spec(spec, len(snapshot.headers))
        groups = normalized["group_by"]
        metrics = normalized["metrics"]
        accumulators: dict[tuple, list[dict[str, Any]]] = {}
        matched = 0
        for record, row in enumerate(snapshot.rows, start=1):
            if not _matches(row, record, normalized["filters"]):
                continue
            matched += 1
            key = tuple(row[position - 1] if row[position - 1].strip() else None for position in groups)
            if key not in accumulators:
                accumulators[key] = [{"count": 0, "sum": Fraction(0), "min": None, "max": None} for _ in metrics]
            for metric, accumulator in zip(metrics, accumulators[key]):
                column = metric["column"]
                cell = None if column is None else row[column - 1]
                if cell is not None and not cell.strip():
                    continue
                accumulator["count"] += 1
                if metric["op"] == "count":
                    continue
                numeric = _numeric(cell, row=record, column=column)
                accumulator["sum"] += numeric
                accumulator["min"] = numeric if accumulator["min"] is None else min(accumulator["min"], numeric)
                accumulator["max"] = numeric if accumulator["max"] is None else max(accumulator["max"], numeric)
        # A nongrouped empty selection still has meaningful count=0 statistics.
        if not groups and not accumulators:
            accumulators[()] = [{"count": 0, "sum": Fraction(0), "min": None, "max": None} for _ in metrics]

        group_fields = [f"column_{position}" for position in groups]
        result_rows = []
        for key, aggregates in accumulators.items():
            current = dict(zip(group_fields, key))
            for metric, accumulator in zip(metrics, aggregates):
                operation, count = metric["op"], accumulator["count"]
                if operation == "count":
                    result = count
                elif not count:
                    result = None
                elif operation == "sum":
                    result = accumulator["sum"]
                elif operation == "mean":
                    result = accumulator["sum"] / count
                else:
                    result = accumulator[operation]
                current[metric["alias"]] = result
            result_rows.append(current)
        result_rows.sort(key=cmp_to_key(lambda a, b: _compare_rows(a, b, normalized["order_by"], group_fields)))
        # Sort exact fractions before rendering. Two close decimal values can
        # round to the same JSON float, which must not alter their ranking.
        for current in result_rows:
            for alias in [metric["alias"] for metric in metrics]:
                if isinstance(current[alias], Fraction):
                    current[alias] = _json_number(current[alias])
        total_groups = len(result_rows)
        returned_rows = result_rows[:normalized["top_k"]]
        requested_top_k = normalized.pop("requested_top_k")
        return {
            "status": "completed", "source": source, "spec": normalized,
            "input_row_count": len(snapshot.rows), "filtered_row_count": matched,
            "group_count": total_groups, "returned_row_count": len(returned_rows),
            "requested_top_k": requested_top_k, "result_limit": MAX_RESULT_ROWS,
            "truncated": total_groups > len(returned_rows),
            "limit_applied": requested_top_k > MAX_RESULT_ROWS,
            "columns": group_fields + [metric["alias"] for metric in metrics],
            "group_columns": [{"field": field, "position": position, "name": snapshot.headers[position - 1]}
                              for field, position in zip(group_fields, groups)],
            "rows": returned_rows,
            "semantics": {
                "filters": "AND; string equality matches exact raw cells; numeric comparisons use exact decimal values",
                "missing": "whitespace-only cells; numeric aggregates exclude missing values",
                "group_labels": "raw nonmissing labels are preserved; missing labels form a null group; no implicit merging",
                "ordering": "specified fields, then group labels ascending; nulls last; exactly top_k rows when available",
                "numeric": "count and integral results are exact integers; nonintegral results are finite JSON numbers",
            },
        }
    except _AnalysisError as exc:
        return {"status": "error", "source": source, "error": {"code": exc.code, "message": str(exc)}}
