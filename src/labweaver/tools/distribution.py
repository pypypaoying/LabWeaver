"""Exact, read-only histogram preparation over a complete CSV snapshot."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

from .csv_analysis import _AnalysisError, _json_number, _matches, _numeric, _position, _validate_spec

if TYPE_CHECKING:
    from .csv_profile import CsvSnapshot


def prepare_distribution(
    snapshot: CsvSnapshot, column_position: int,
    filters: list[dict[str, Any]] | None = None, bins: int = 10,
) -> dict[str, Any]:
    """Count equal-width intervals from every selected source record.

    Filtering and numeric parsing share the aggregate tool's exact decimal
    semantics. Bins are left closed, right open, except the final bin, whose
    right endpoint is included. A constant column spans value +/- 0.5.
    Errors never contain a partially computed distribution.
    """
    source = dict(snapshot.source)
    try:
        column = _position(column_position, len(snapshot.headers), "Distribution column")
        if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 50:
            raise _AnalysisError("invalid_spec", "bins must be an integer between 1 and 50.")
        normalized = _validate_spec({
            "filters": [] if filters is None else filters,
            "metrics": [{"op": "count", "alias": "count"}],
        }, len(snapshot.headers))
        values: list[Fraction] = []
        matched = missing = 0
        for record, row in enumerate(snapshot.rows, start=1):
            if not _matches(row, record, normalized["filters"]):
                continue
            matched += 1
            cell = row[column - 1]
            if not cell.strip():
                missing += 1
                continue
            values.append(_numeric(cell, row=record, column=column))
        if not matched:
            raise _AnalysisError("empty_selection", "No source records match the distribution filters.")
        if not values:
            raise _AnalysisError("no_valid_values", "The selected column contains no nonmissing numeric values.")
        lower, upper = min(values), max(values)
        constant = lower == upper
        if constant:
            lower -= Fraction(1, 2)
            upper += Fraction(1, 2)
        width = (upper - lower) / bins
        exact_edges = [lower + index * width for index in range(bins + 1)]
        counts = [0] * bins
        for value in values:
            index = min(int((value - lower) // width), bins - 1)
            counts[index] += 1
        edges = [_json_number(value) for value in exact_edges]
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise _AnalysisError("numeric_out_of_range", "Histogram boundaries cannot be represented as distinct finite JSON numbers.")
        rows = [{"bin_left": edges[index], "bin_right": edges[index + 1], "count": count}
                for index, count in enumerate(counts)]
        return {
            "status": "completed", "kind": "distribution", "source": source,
            "spec": {"column_position": column, "filters": normalized["filters"], "bins": bins},
            "column_name": snapshot.headers[column - 1],
            "columns": ["bin_left", "bin_right", "count"], "rows": rows,
            "input_row_count": len(snapshot.rows), "filtered_row_count": matched,
            "valid_count": len(values), "missing_count": missing, "edges": edges,
            "group_count": bins, "returned_row_count": bins, "truncated": False,
            "semantics": {
                "filters": "AND; string equality matches exact raw cells; numeric comparisons use exact decimal values",
                "missing": "whitespace-only cells are excluded and counted; literal zero, NA and NULL are not missing",
                "numeric": "all nonmissing selected cells must be finite decimal numbers; exact rational arithmetic before JSON conversion",
                "bins": "equal width; [left, right), with the final right endpoint included",
                "constant": "constant columns span value - 0.5 to value + 0.5" if constant else "no range expansion",
                "scope": "every selected record in the complete CSV snapshot; no sampling or top-k selection",
            },
        }
    except _AnalysisError as exc:
        return {"status": "error", "kind": "distribution", "source": source,
                "error": {"code": exc.code, "message": str(exc)}}
