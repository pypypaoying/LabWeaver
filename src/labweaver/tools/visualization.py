"""Bounded chart templates for host-registered statistics; no source I/O."""

from __future__ import annotations

import csv
from datetime import date
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .csv_analysis import _AnalysisError, _numeric


_RENDER_LOCK = threading.RLock()
_MONTH = re.compile(r"[0-9]{4}-[0-9]{2}\Z")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")


class PlotSpec(BaseModel):
    """One chart, one metric; model-supplied data and code are forbidden."""

    model_config = ConfigDict(extra="forbid", strict=True)
    chart_type: Literal["bar", "line", "histogram"]
    x: str | None = Field(default=None, max_length=80)
    y: str | None = Field(default=None, max_length=80)
    title: str = Field(default="", max_length=160)
    x_label: str = Field(default="", max_length=80)
    y_label: str = Field(default="", max_length=80)
    orientation: Literal["vertical", "horizontal"] = "vertical"
    x_type: Literal["auto", "numeric", "date"] = "auto"

    @field_validator("x", "y", "title", "x_label", "y_label")
    @classmethod
    def plain_text(cls, value: str | None) -> str | None:
        if value is not None and any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Chart labels must not contain control characters.")
        return value

    @field_validator("x", "y")
    @classmethod
    def empty_field(cls, value: str | None) -> str | None:
        return value or None


class PlotError(ValueError):
    """A structured chart error without an incomplete asset."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PlotError("invalid_numeric_value", f"{label} must contain finite numbers.")
    try:
        exact = _numeric(str(value), row=0, column=0)
        rendered = float(exact)
    except (_AnalysisError, OverflowError, ValueError) as exc:
        raise PlotError("numeric_out_of_range", f"{label} cannot be represented on a finite numeric axis.") from exc
    if not math.isfinite(rendered) or rendered == 0 and exact != 0:
        raise PlotError("numeric_out_of_range", f"{label} exceeds the finite plotting range.")
    # Ordinary decimal floats are drawable. Reject integers and raw decimal
    # strings whose precision would silently change when converted to float.
    if not isinstance(value, float):
        represented = Fraction.from_float(rendered) if exact.denominator == 1 else Fraction(str(rendered))
        if represented != exact:
            raise PlotError("numeric_precision_loss", f"{label} loses numeric precision on a floating point axis.")
    return rendered


def _label(value: Any) -> str:
    rendered = "(missing)" if value is None else str(value)
    if len(rendered) > 160 or any(ord(char) < 32 or ord(char) == 127 for char in rendered):
        raise PlotError("invalid_label", "A category label is too long or contains control characters.")
    return rendered


def _date(value: Any) -> date:
    if not isinstance(value, str) or not (_MONTH.fullmatch(value) or _DATE.fullmatch(value)):
        raise PlotError("invalid_x_axis", "Date axes require ISO YYYY-MM or YYYY-MM-DD values.")
    try:
        return date.fromisoformat(value + "-01" if _MONTH.fullmatch(value) else value)
    except ValueError as exc:
        raise PlotError("invalid_x_axis", "The date axis contains an invalid calendar date.") from exc


def _font(text: list[str], font_path: str | Path | None):
    from matplotlib import font_manager

    required = {ord(char) for label in text for char in label if ord(char) >= 32}
    required.update(map(ord, "0123456789-+.eE"))
    if font_path is not None:
        path = Path(font_path)
        if not path.is_file():
            raise PlotError("font_unavailable", "The configured font_path is not a readable font file.")
        candidates = [str(path)]
    else:
        preferred = ("microsoft yahei", "noto sans cjk", "noto serif cjk", "simhei", "simsun", "pingfang", "arial unicode")
        has_nonlatin = any(code > 255 for code in required)
        entries = list(font_manager.fontManager.ttflist)
        def rank(entry):
            name = entry.name.lower()
            if not has_nonlatin and name == "dejavu sans":
                priority = -1
            else:
                priority = next((index for index, key in enumerate(preferred) if key in name), len(preferred))
            weight = entry.weight if isinstance(entry.weight, int) else 400
            return priority, entry.style != "normal", abs(weight - 400), entry.fname
        candidates = list(dict.fromkeys(entry.fname for entry in sorted(entries, key=rank)))
    for candidate in candidates:
        try:
            loaded = font_manager.get_font(candidate)
            if required <= loaded.get_charmap().keys():
                return font_manager.FontProperties(fname=candidate), loaded.family_name
        except (OSError, RuntimeError, ValueError):
            continue
    raise PlotError("font_unavailable", "No available font covers all chart characters. Install a CJK font or set font_path.")


def _table(data: dict, spec: PlotSpec) -> tuple[str, str, list[dict], list, list[float], int, str]:
    columns, rows = data.get("columns"), data.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list) or not rows:
        raise PlotError("empty_result", "A chart requires a nonempty registered result table.")
    if len(rows) > 100 or any(not isinstance(row, dict) for row in rows):
        raise PlotError("invalid_data", "The registered result table is malformed or exceeds 100 rows.")
    if data.get("kind") != "aggregate":
        raise PlotError("invalid_data", "Bar and line charts require a registered aggregate result.")
    groups = [item["field"] for item in data.get("group_columns", [])]
    metrics = [item["alias"] for item in data.get("spec", {}).get("metrics", [])]
    x = spec.x or (groups[0] if len(groups) == 1 else None)
    y = spec.y or (metrics[0] if len(metrics) == 1 else None)
    if x not in groups or y not in metrics or x not in columns or y not in columns:
        raise PlotError("invalid_fields", "Select one existing group field for x and one aggregate metric for y.")
    if len(groups) != 1:
        raise PlotError("ambiguous_groups", "Charts require one grouping field; recompute with a single group column.")
    selected, xs, ys = [], [], []
    missing = 0
    x_type = spec.x_type
    if spec.chart_type == "line" and x_type == "auto":
        if all(isinstance(row.get(x), str) and (_MONTH.fullmatch(row[x]) or _DATE.fullmatch(row[x])) for row in rows):
            x_type = "date"
        else:
            x_type = "numeric"
    seen = set()
    for row in rows:
        if x not in row or y not in row:
            raise PlotError("invalid_data", "A registered result row is missing a selected field.")
        if spec.chart_type == "line":
            if row[x] is None or isinstance(row[x], str) and not row[x].strip():
                raise PlotError("invalid_x_axis", "Line chart axes cannot contain missing values.")
            axis_x = _date(row[x]) if x_type == "date" else _number(row[x], "The x axis")
            if axis_x in seen:
                raise PlotError("duplicate_x_axis", "Line chart x values repeat. Recompute with a defined aggregation for each x value.")
            seen.add(axis_x)
        else:
            axis_x = _label(row[x])
        if row[y] is None:
            missing += 1
            continue
        ys.append(_number(row[y], "The metric"))
        xs.append(axis_x)
        selected.append({x: row[x], y: row[y]})
    if not selected:
        raise PlotError("no_valid_values", "No nonmissing metric values are available to plot.")
    if spec.chart_type == "line":
        order = sorted(range(len(xs)), key=xs.__getitem__)
        selected, xs, ys = ([selected[index] for index in order], [xs[index] for index in order], [ys[index] for index in order])
    return x, y, selected, xs, ys, missing, x_type


def _histogram(data: dict, spec: PlotSpec) -> tuple[list[dict], list[float], list[int]]:
    if data.get("kind") != "distribution" or data.get("truncated"):
        raise PlotError("invalid_data", "Histograms require a complete registered distribution, never samples or top-k results.")
    if spec.x not in {None, "bin_left"} or spec.y not in {None, "count"}:
        raise PlotError("invalid_fields", "A histogram uses bin_left, bin_right and count from the distribution result.")
    rows, edges = data.get("rows"), data.get("edges")
    if (data.get("columns") != ["bin_left", "bin_right", "count"] or not isinstance(rows, list)
            or not 1 <= len(rows) <= 50 or not isinstance(edges, list) or len(edges) != len(rows) + 1):
        raise PlotError("invalid_data", "The registered histogram intervals are malformed.")
    plotted_edges = [_number(value, "Histogram boundaries") for value in edges]
    if any(a >= b for a, b in zip(plotted_edges, plotted_edges[1:])):
        raise PlotError("numeric_precision_loss", "Histogram intervals collapse on the floating point axis.")
    counts = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("bin_left") != edges[index] or row.get("bin_right") != edges[index + 1]:
            raise PlotError("invalid_data", "Histogram rows do not match their registered boundaries.")
        count = row.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PlotError("invalid_data", "Histogram counts must be nonnegative integers.")
        _number(count, "Histogram frequencies")
        counts.append(count)
    if not sum(counts) or sum(counts) != data.get("valid_count"):
        raise PlotError("invalid_data", "Histogram frequencies do not match the registered valid record count.")
    widths = [b - a for a, b in zip(plotted_edges, plotted_edges[1:])]
    if not all(math.isclose(width, widths[0], rel_tol=1e-9, abs_tol=0) for width in widths):
        raise PlotError("invalid_data", "Histogram bins must have equal width.")
    return [dict(row) for row in rows], plotted_edges, counts


def render_chart(data: dict[str, Any], spec: dict[str, Any], *, font_path: str | Path | None = None) -> dict[str, Any]:
    """Return PNG/SVG bytes and the exact plotted CSV; write no output files.

    Authorization and stable data registration belong to the calling host.
    Expected validation and rendering failures return status=error, with no
    image bytes. Metadata contains only strict JSON values.
    """
    data_id = data.get("data_id") if isinstance(data, dict) else None
    try:
        if not isinstance(data, dict) or data.get("status") != "completed" or not isinstance(data_id, str) or not 1 <= len(data_id) <= 160:
            raise PlotError("invalid_data", "A chart requires a completed host-registered result with data_id.")
        request = PlotSpec.model_validate(spec)
        requested_spec = request.model_dump()
        if request.chart_type != "bar" and request.orientation != "vertical":
            raise PlotError("invalid_spec", "Only bar charts support horizontal orientation.")
        if request.chart_type != "line" and request.x_type != "auto":
            raise PlotError("invalid_spec", "x_type applies only to line charts.")
        source = json.loads(json.dumps(data.get("source", {}), allow_nan=False))
        data_spec = json.loads(json.dumps(data.get("spec", {}), allow_nan=False))
        data_semantics = json.loads(json.dumps(data.get("semantics", {}), allow_nan=False))
        if request.chart_type == "histogram":
            plotted_rows, xs, ys = _histogram(data, request)
            x, y, columns = "bin_left", "count", ["bin_left", "bin_right", "count"]
            missing = data.get("missing_count", 0)
            if isinstance(missing, bool) or not isinstance(missing, int) or missing < 0:
                raise PlotError("invalid_data", "The distribution missing count is invalid.")
            x_type = "numeric"
        else:
            x, y, plotted_rows, xs, ys, missing, x_type = _table(data, request)
            columns = [x, y]
        request = request.model_copy(update={"x": x, "y": y})
        resolved_spec = request.model_dump()
        title = request.title or {"bar": "Category comparison", "line": "Trend", "histogram": "Distribution"}[request.chart_type]
        group_name = next((item.get("name", x) for item in data.get("group_columns", []) if item.get("field") == x), x)
        x_label = request.x_label or (data.get("column_name", x) if request.chart_type == "histogram" else group_name)
        y_label = request.y_label or ("Frequency" if request.chart_type == "histogram" else y)
        resolved_spec.update(title=title, x_label=x_label, y_label=y_label)
        if request.chart_type == "line":
            resolved_spec["x_type"] = x_type
        notes = []
        if data.get("truncated"):
            notes.append(f"Showing {len(data['rows'])} of {data.get('group_count', '?')} groups (selected result range)")
        if missing:
            notes.append(f"Excluded {missing} missing metric values")
        range_note = "; ".join(notes)
        labels = [title, _label(x_label), _label(y_label), range_note]
        if request.chart_type == "bar":
            labels.extend(xs)
        table_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(table_buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(plotted_rows)
        data_csv = table_buffer.getvalue()
        identity = json.dumps({"data_id": data_id, "source": source, "spec": resolved_spec,
                               "rows": plotted_rows}, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        chart_id = "chart_" + hashlib.sha256(identity).hexdigest()[:24]
        # Object-oriented Agg rendering avoids pyplot global figure state.
        # Matplotlib's font and rc registries require a shared render lock.
        with _RENDER_LOCK:
            import matplotlib
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            import matplotlib.dates as mdates
            from matplotlib.ticker import MaxNLocator

            font, font_name = _font(labels, font_path)
            with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": chart_id,
                                        "axes.unicode_minus": False, "text.parse_math": False,
                                        "font.family": font_name}):
                figure = Figure(figsize=(10, 6), dpi=150, layout="constrained")
                FigureCanvasAgg(figure)
                axes = figure.subplots()
                if request.chart_type == "bar":
                    locations = list(range(len(xs)))
                    if request.orientation == "horizontal":
                        axes.barh(locations, ys, color="#4472C4")
                        axes.set_yticks(locations, xs)
                        axes.invert_yaxis()
                        axes.set_xlabel(y_label, fontproperties=font)
                        axes.set_ylabel(x_label, fontproperties=font)
                    else:
                        axes.bar(locations, ys, color="#4472C4")
                        axes.set_xticks(locations, xs, rotation=30, ha="right")
                        axes.set_xlabel(x_label, fontproperties=font)
                        axes.set_ylabel(y_label, fontproperties=font)
                elif request.chart_type == "line":
                    axes.plot(xs, ys, color="#4472C4", marker="o", linewidth=2)
                    axes.set_xlabel(x_label, fontproperties=font)
                    axes.set_ylabel(y_label, fontproperties=font)
                    if x_type == "date":
                        monthly = all(_MONTH.fullmatch(str(row[x])) for row in plotted_rows)
                        pattern = "%Y-%m" if monthly else "%Y-%m-%d"
                        axes.xaxis.set_major_formatter(mdates.DateFormatter(pattern))
                        if len(xs) == 1:
                            axes.set_xticks(xs)
                        elif monthly:
                            span = (xs[-1].year - xs[0].year) * 12 + xs[-1].month - xs[0].month
                            axes.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, math.ceil(span / 8))))
                        else:
                            span = (xs[-1] - xs[0]).days
                            axes.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, math.ceil(span / 8))))
                        axes.tick_params(axis="x", labelrotation=30)
                else:
                    axes.bar(xs[:-1], ys, width=[b - a for a, b in zip(xs, xs[1:])],
                             align="edge", color="#4472C4", edgecolor="white")
                    axes.set_xlabel(x_label, fontproperties=font)
                    axes.set_ylabel(y_label, fontproperties=font)
                    axes.set_xlim(xs[0], xs[-1])
                    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
                axes.set_title(title, fontproperties=font, fontsize=16, pad=16)
                axes.set_axisbelow(True)
                axes.grid(axis="x" if request.orientation == "horizontal" else "y", alpha=0.2)
                for tick in axes.get_xticklabels() + axes.get_yticklabels():
                    tick.set_fontproperties(font)
                for offset in (axes.xaxis.get_offset_text(), axes.yaxis.get_offset_text()):
                    offset.set_fontproperties(font)
                if range_note:
                    figure.supxlabel(range_note, fontsize=9, fontproperties=font)
                png_buffer, svg_buffer = io.BytesIO(), io.BytesIO()
                figure.savefig(png_buffer, format="png", dpi=150, metadata={"Software": "LabWeaver"})
                figure.savefig(svg_buffer, format="svg", metadata={"Date": None, "Creator": "LabWeaver"})
                png, svg = png_buffer.getvalue(), svg_buffer.getvalue()
                figure.clear()
        return {
            "status": "completed",
            "metadata": {"chart_id": chart_id, "data_id": data_id, "source": source,
                         "session_id": data.get("session_id"), "task_id": data.get("task_id"),
                         "spec": resolved_spec, "requested_spec": requested_spec,
                         "data_spec": data_spec, "data_semantics": data_semantics,
                         "columns": columns, "rows": plotted_rows, "x_type": x_type,
                         "excluded_missing_count": missing, "truncated": bool(data.get("truncated")),
                         "range_note": range_note, "font": font_name,
                         "size_inches": [10, 6], "dpi": 150,
                         "hashes": {"png": hashlib.sha256(png).hexdigest(),
                                    "svg": hashlib.sha256(svg).hexdigest(),
                                    "data_csv": hashlib.sha256(data_csv.encode("utf-8")).hexdigest()}},
            "png": png, "svg": svg, "data_csv": data_csv,
        }
    except ValidationError:
        return {"status": "error", "data_id": data_id,
                "error": {"code": "invalid_spec", "message": "Chart spec has unsupported fields, values or labels."}}
    except PlotError as exc:
        return {"status": "error", "data_id": data_id, "error": {"code": exc.code, "message": str(exc)}}
    except (ValueError, TypeError, KeyError, OverflowError, OSError, RuntimeError) as exc:
        # Avoid leaking private filesystem paths through library exception text.
        return {"status": "error", "data_id": data_id,
                "error": {"code": "render_failed", "message": f"The chart could not be rendered ({type(exc).__name__})."}}
