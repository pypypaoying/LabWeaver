"""Bound model context, never the authoritative result or its exported rows."""

import copy
import json

MAX_PAGE_BYTES = 64 * 1024


def chart_receipt(metadata: dict) -> dict:
    """A model-visible asset receipt; the full plotting table stays with the host."""
    receipt = copy.deepcopy({key: value for key, value in metadata.items() if key != "rows"})
    receipt["plotted_row_count"] = len(metadata.get("rows", []))
    return receipt


def result_page(result: dict, offset: int = 0, limit: int | None = None) -> dict:
    """Return a byte-bounded view with an explicit cursor into registered rows.

    returned_row_count and truncated describe the calculation, not this view.
    A preview is never a new calculation and must not be aggregated as a whole.
    """
    if result.get("status") != "completed":
        return copy.deepcopy(result)
    rows = result["rows"]
    if type(offset) is not int or not 0 <= offset <= len(rows):
        raise ValueError("offset must be between zero and the available row count")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1000):
        raise ValueError("page limit must be between 1 and 1000")
    output = copy.deepcopy({key: value for key, value in result.items() if key != "rows"})
    # Reserve space for paging metadata; even a very wide header must be bounded.
    remaining = MAX_PAGE_BYTES - len(json.dumps(output, ensure_ascii=False, allow_nan=False).encode("utf-8")) - 1024
    selected = []
    end = len(rows) if limit is None else min(len(rows), offset + limit)
    for index in range(offset, end):
        row = rows[index]
        size = len(json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 2
        if size > remaining:
            break
        selected.append(copy.deepcopy(row))
        remaining -= size
    if remaining < 0 or not selected and offset < len(rows):
        # Full results are still saved by the host; don't pretend an empty view
        # is the complete result or offer a cursor that cannot advance.
        return {"status": "completed", "data_id": result["data_id"],
                "source": copy.deepcopy(result["source"]), "rows": [],
                "preview_only": True, "available_row_count": len(rows),
                "page": {"offset": offset, "row_count": 0, "next_offset": None},
                "view_error": "A row or its metadata exceeds the model message budget. Full results remain available in the exported CSV."}
    next_offset = offset + len(selected)
    output.update(rows=selected, preview_only=offset != 0 or len(selected) != len(rows),
                  available_row_count=len(rows),
                  page={"offset": offset, "row_count": len(selected),
                        "next_offset": next_offset if next_offset < len(rows) else None})
    return output
