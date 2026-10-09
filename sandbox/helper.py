"""Input/output helpers inside the container. Analysis belongs in generated code."""

import json
import math
import re
from pathlib import Path
import uuid

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.findfont("Noto Sans CJK JP", fallback_to_default=False)

plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
    }
)
ROOT = Path("/work/artifacts")


def load_dataset():
    """Return raw strings with stable c1..cN columns; names in df.attrs['columns']."""
    payload = json.loads(Path("/input/dataset.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        payload["rows"], columns=[c["id"] for c in payload["columns"]], dtype=object
    )
    frame.attrs["columns"] = payload["columns"]
    return frame


def _entry(kind, deliverable_id, label):
    ROOT.mkdir(exist_ok=True)
    entry = ROOT / uuid.uuid4().hex
    entry.mkdir()
    metadata = {"kind": kind, "deliverable_id": deliverable_id, "label": label}
    return entry, metadata


def _finish(entry, metadata):
    (entry / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def emit_table(frame, deliverable_id, label="结果表"):
    """Export all rows, preserving column order. No truncation or aggregation."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("emit_table expects a DataFrame")
    if not frame.columns.is_unique or any(
        not isinstance(c, str) for c in frame.columns
    ):
        raise ValueError(
            "Output columns must be unique strings; use explicit output labels"
        )
    entry, meta = _entry("table", deliverable_id, label)
    frame.to_csv(
        entry / "table.csv", index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    meta.update(row_count=len(frame), columns=list(frame.columns))
    _finish(entry, meta)


def emit_metric(value, deliverable_id, label="指标"):
    """Export a JSON scalar/object, rejecting nonfinite values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Metric must be finite")
    entry, meta = _entry("metric", deliverable_id, label)
    (entry / "metric.json").write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    _finish(entry, meta)


def emit_figure(figure, data, deliverable_id, label="图表"):
    """Export PNG, SVG and the actual plotting DataFrame, without choosing a chart."""
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        raise TypeError("emit_figure requires a plotting DataFrame with unique columns")
    entry, meta = _entry("figure", deliverable_id, label)
    figure.savefig(entry / "figure.png", dpi=150, bbox_inches="tight")
    figure.savefig(entry / "figure.svg", bbox_inches="tight")
    svg = (entry / "figure.svg").read_text(encoding="utf-8")
    (entry / "figure.svg").write_text(
        re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.S), encoding="utf-8"
    )
    data.to_csv(
        entry / "data.csv", index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    meta.update(row_count=len(data), columns=[str(c) for c in data.columns])
    _finish(entry, meta)
    plt.close(figure)
