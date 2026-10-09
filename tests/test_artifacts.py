"""Adversarial output validation and uncapped complete-table export."""

import base64
import copy
import hashlib
import json
from pathlib import Path
import struct
import zlib
import pytest
from labweaver.runtime.artifacts import (
    artifact_view,
    validate_payload,
    validate_svg,
    validate_png,
)
from labweaver.runtime.records import save_run
from labweaver.tools.result_pages import result_page, MAX_PAGE_BYTES

SOURCE = {"name": "selected.csv", "sha256": "b" * 64}
PLAN = [{"id": "table", "kind": "table", "description": "full table"}]


def payload(rows=280):
    raw = "group,value\n" + "".join(f"{i:03},1\n" for i in range(rows))
    return [
        {
            "metadata": {
                "kind": "table",
                "deliverable_id": "table",
                "label": "结果表",
                "columns": ["group", "value"],
                "row_count": rows,
            },
            "files": {"table.csv": base64.b64encode(raw.encode()).decode()},
        }
    ]


def assets(data=None, **kwargs):
    return validate_payload(
        payload() if data is None else data,
        deliverables=PLAN,
        source=SOURCE,
        execution_id="execution-1",
        image_id="sha256:" + "c" * 64,
        **kwargs,
    )


def report_and_assets():
    registered = assets()
    for item in registered.values():
        item["metadata"].update(session_id="session", task_id="task")
    return {
        "report_version": 2,
        "status": "completed",
        "session_id": "session",
        "task_id": "task",
        "task": "统计",
        "source": SOURCE,
        "task_plan": PLAN,
        "artifacts": [a["metadata"] for a in registered.values()],
        "fulfilled_deliverable_ids": ["table"],
        "code_executions": [],
    }, registered


def test_280_groups_export_in_full_while_model_pages(tmp_path):
    report, registry = report_and_assets()
    asset = next(iter(registry.values()))
    preview = artifact_view(asset, limit=10)
    assert preview["available_row_count"] == 280 and len(preview["rows"]) == 10
    assert preview["page"]["next_offset"] == 10
    later = artifact_view(asset, offset=270, limit=20)
    assert len(later["rows"]) == 10 and later["page"]["next_offset"] is None
    path = save_run(report, tmp_path, artifact_assets=registry)
    saved = json.loads(path.read_text(encoding="utf-8"))
    raw = Path(saved["result_csv_paths"][0]).read_bytes()
    assert len(raw.decode().splitlines()) == 281
    assert (
        hashlib.sha256(raw).hexdigest()
        == asset["metadata"]["files"]["table.csv"]["sha256"]
    )
    assert "完整结果 280 行" in Path(saved["brief_path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    ["path", "extra", "kind", "delivery", "rows", "columns", "base64", "nan"],
)
def test_invalid_protocol_artifacts_rejected_atomically(mutation):
    data = payload(2)
    meta, files = data[0]["metadata"], data[0]["files"]
    if mutation == "path":
        files["../table.csv"] = files.pop("table.csv")
    elif mutation == "extra":
        files["payload.py"] = ""
    elif mutation == "kind":
        meta["kind"] = "executable"
    elif mutation == "delivery":
        meta["deliverable_id"] = "outside-scope"
    elif mutation == "rows":
        meta["row_count"] = 100
    elif mutation == "columns":
        meta["columns"] = ["wrong"]
    elif mutation == "base64":
        files["table.csv"] = "%%%%"
    else:
        data = [
            {
                "metadata": {
                    "kind": "metric",
                    "deliverable_id": "table",
                    "label": "invalid",
                },
                "files": {"metric.json": base64.b64encode(b"NaN").decode()},
            }
        ]
    with pytest.raises(ValueError):
        assets(data)


def test_total_byte_limit_is_not_a_row_limit():
    with pytest.raises(ValueError, match="50 MiB"):
        assets(remaining_bytes=10)
    assert next(iter(assets(payload(1000)).values()))["metadata"]["row_count"] == 1000


def test_output_field_can_exceed_input_field_limit_without_changing_global_limit():
    import csv
    from labweaver.runtime.artifacts import csv_table

    before = csv.field_size_limit()
    text = "x" * 200_000
    assert csv_table(("value\n" + text + "\n").encode())[1] == [{"value": text}]
    assert csv.field_size_limit() == before


def test_svg_local_quoted_reference_allowed_external_still_rejected():
    validate_svg(b"<svg><path style=\"clip-path:url('\x23clip')\"/></svg>")
    with pytest.raises(ValueError):
        validate_svg(
            b"<svg><path style=\"fill:url('https://example.invalid')\"/></svg>"
        )


@pytest.mark.parametrize(
    "svg",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg"><script>bad()</script></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg" onload="bad()"/>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><use href="https://example.invalid"/></svg>',
        b'<!DOCTYPE svg SYSTEM "file:///secret"><svg/>',
        b'<svg><rect style="fill:url(https://example.invalid)"/></svg>',
        b"<svg><foreignObject/></svg>",
    ],
)
def test_active_or_external_svg_rejected(svg):
    with pytest.raises(ValueError):
        validate_svg(svg)


def test_png_crc_and_structure():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\0\0\0"))
        + chunk(b"IEND", b"")
    )
    validate_png(raw)
    # Matplotlib raster/heatmap SVGs embed PNGs, not active SVG or external URLs.
    validate_svg(
        b'<svg><image href="data:image/png;base64,'
        + base64.b64encode(raw)
        + b'"/></svg>'
    )
    with pytest.raises(ValueError):
        validate_png(raw[:-1])


def test_export_rollback_and_tamper_refusal(tmp_path, monkeypatch):
    report, registry = report_and_assets()
    forged = copy.deepcopy(registry)
    next(iter(forged.values()))["files"]["table.csv"] += b"extra"
    with pytest.raises(ValueError):
        save_run(report, tmp_path, artifact_assets=forged)
    original = Path.open

    def failing(self, mode="r", *args, **kwargs):
        if self.suffix == ".md" and mode == "xb":
            raise OSError("fixture write failure")
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing)
    with pytest.raises(OSError):
        save_run(report, tmp_path, artifact_assets=registry)
    assert not list(tmp_path.rglob("*.csv")) and not list(tmp_path.rglob("*.json"))


def test_byte_bound_page_does_not_claim_complete():
    result = {
        "status": "completed",
        "source": SOURCE,
        "data_id": "id",
        "columns": ["x"],
        "rows": [{"x": "a" * MAX_PAGE_BYTES}],
    }
    view = result_page(result)
    assert view["preview_only"] and view["view_error"] and not view["rows"]


@pytest.mark.parametrize("identifier", ["../task", "a/b", "", "a" * 81])
def test_export_path_identifiers_are_host_safe(tmp_path, identifier):
    report, registry = report_and_assets()
    report["task_id"] = identifier
    with pytest.raises(ValueError):
        save_run(report, tmp_path, artifact_assets=registry)
