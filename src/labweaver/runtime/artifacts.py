"""Validate untrusted container output before registering host-owned artifacts."""

import base64
import csv
import hashlib
import io
import json
import re
import struct
import uuid
import xml.etree.ElementTree as ET
import zlib

MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
FILES = {
    "table": {"table.csv"},
    "metric": {"metric.json"},
    "figure": {"figure.png", "figure.svg", "data.csv"},
}


def strict_json(raw):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(raw, parse_constant=invalid)


def validate_png(raw):
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG signature")
    position, types = 8, []
    while position < len(raw):
        if position + 12 > len(raw):
            raise ValueError("Incomplete PNG")
        size = struct.unpack(">I", raw[position : position + 4])[0]
        kind = raw[position + 4 : position + 8]
        end = position + 12 + size
        if (
            end > len(raw)
            or zlib.crc32(raw[position + 4 : end - 4])
            != struct.unpack(">I", raw[end - 4 : end])[0]
        ):
            raise ValueError("Invalid PNG chunk")
        if kind == b"IHDR":
            width, height = struct.unpack(">II", raw[position + 8 : position + 16])
            if not 0 < width * height <= 100_000_000:
                raise ValueError("PNG dimensions exceed limits")
        types.append(kind)
        position = end
    if not types or types[0] != b"IHDR" or types[-1] != b"IEND" or b"IDAT" not in types:
        raise ValueError("Invalid PNG structure")


def validate_svg(raw):
    text = raw.decode("utf-8", errors="strict")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)|<\?xml-stylesheet", text, re.I):
        raise ValueError("SVG declarations are forbidden")
    root = ET.fromstring(text)
    allowed = {
        "svg",
        "g",
        "defs",
        "path",
        "rect",
        "circle",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "text",
        "tspan",
        "clipPath",
        "use",
        "image",
        "linearGradient",
        "radialGradient",
        "stop",
        "title",
        "desc",
        "style",
        "metadata",
        "RDF",
        "Work",
        "type",
        "date",
        "format",
        "creator",
        "Agent",
        "title",
    }
    if root.tag.split("}")[-1] != "svg":
        raise ValueError("Invalid SVG root")
    for node in root.iter():
        if node.tag.split("}")[-1] not in allowed:
            raise ValueError("Active or unsupported SVG element")
        for key, value in node.attrib.items():
            key = key.split("}")[-1].lower()
            if (
                key == "href"
                and node.tag.split("}")[-1] == "image"
                and value.startswith("data:image/png;base64,")
            ):
                validate_png(
                    base64.b64decode(
                        "".join(value.split(",", 1)[1].split()), validate=True
                    )
                )
                continue
            if (
                key.startswith("on")
                or key in {"href", "src"}
                and not re.fullmatch(r"#[\w.-]+", value)
            ):
                raise ValueError("SVG active content or external link")
            without_local_urls = re.sub(
                r"url\s*\(\s*(['\"]?)#[\w.-]+\1\s*\)", "", value, flags=re.I
            )
            if re.search(
                r"@import|(?:javascript|data):|url\s*\(", without_local_urls, re.I
            ):
                raise ValueError("SVG external resource")
        if node.tag.split("}")[-1] == "style" and re.search(
            r"@import|url\s*\(", node.text or "", re.I
        ):
            raise ValueError("SVG external style")


def csv_table(raw):
    # Output tables can exceed input field limits, but never the total byte budget.
    from labweaver.tools.csv_profile import _CSV_LOCK

    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("Artifacts exceed 50 MiB")
    with (
        _CSV_LOCK,
        io.StringIO(raw.decode("utf-8-sig", errors="strict"), newline="") as stream,
    ):
        previous_limit = csv.field_size_limit(MAX_ARTIFACT_BYTES)
        try:
            reader = csv.reader(stream, strict=True)
            columns = next(reader, None)
            if not columns or len(set(columns)) != len(columns):
                raise ValueError("Output table requires unique column labels")
            rows = []
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError("Invalid output table width")
                rows.append(dict(zip(columns, row)))
        finally:
            csv.field_size_limit(previous_limit)
    return columns, rows


def validate_payload(
    payload,
    *,
    deliverables,
    source,
    execution_id,
    image_id,
    remaining_bytes=MAX_ARTIFACT_BYTES,
    remaining_figures=2,
):
    """Decode an entire execution atomically. IDs, hashes and provenance are host-generated."""
    if not isinstance(payload, list) or len(payload) > 50:
        raise ValueError("Invalid artifact collection")
    allowed = {d["id"]: d["kind"] for d in deliverables}
    result, total, figures = {}, 0, 0
    for item in payload:
        meta, encoded = item["metadata"], item["files"]
        kind, did, label = (
            meta.get("kind"),
            meta.get("deliverable_id"),
            meta.get("label"),
        )
        if (
            kind not in FILES
            or did not in allowed
            or kind != ("metric" if allowed[did] == "explanation" else allowed[did])
            or not isinstance(label, str)
            or not 0 < len(label) <= 300
        ):
            raise ValueError("Artifact does not match an authorized deliverable")
        if not isinstance(encoded, dict) or set(encoded) != FILES[kind]:
            raise ValueError("Invalid artifact filenames or file types")
        files = {}
        for name, value in encoded.items():
            if (
                not isinstance(value, str)
                or len(value) > (remaining_bytes - total) * 4 // 3 + 4
            ):
                raise ValueError("Artifacts exceed 50 MiB")
            raw = base64.b64decode(value, validate=True)
            total += len(raw)
            if total > remaining_bytes:
                raise ValueError("Artifacts exceed 50 MiB")
            files[name] = raw
        metadata = {
            "id": "artifact-" + uuid.uuid4().hex,
            "kind": kind,
            "deliverable_id": did,
            "label": label,
            "source": source.copy(),
            "execution_id": execution_id,
            "image_id": image_id,
            "files": {
                name: {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
                for name, raw in files.items()
            },
        }
        if kind == "metric":
            metadata["value"] = strict_json(files["metric.json"])
        else:
            columns, rows = csv_table(
                files["table.csv" if kind == "table" else "data.csv"]
            )
            if meta.get("row_count") != len(rows) or meta.get("columns") != columns:
                raise ValueError("Artifact table metadata does not match its bytes")
            metadata.update(row_count=len(rows), columns=columns)
            if kind == "figure":
                figures += 1
                if figures > remaining_figures:
                    raise ValueError("At most two figures per task")
                validate_png(files["figure.png"])
                validate_svg(files["figure.svg"])
        result[metadata["id"]] = {"metadata": metadata, "files": files}
    return result


def artifact_view(asset, offset=0, limit=20):
    """Model view of a registered artifact. Full bytes stay on the host."""
    from labweaver.tools.result_pages import result_page

    meta = asset["metadata"]
    output = {"status": "completed", **meta}
    if meta["kind"] == "metric":
        raw = json.dumps(output, ensure_ascii=False, allow_nan=False).encode()
        if len(raw) > 64 * 1024:
            return {
                "status": "completed",
                "id": meta["id"],
                "kind": "metric",
                "preview_only": True,
                "view_error": "Metric exceeds model preview size; see exported JSON.",
            }
        return output
    columns, rows = csv_table(
        asset["files"]["table.csv" if meta["kind"] == "table" else "data.csv"]
    )
    return result_page(
        {**output, "data_id": meta["id"], "columns": columns, "rows": rows},
        offset,
        limit,
    )


def artifact_receipts(assets):
    """Bound the combined receipt; use read_artifact for additional actual rows."""
    receipts, total = [], 0
    for asset in assets:
        meta = asset["metadata"]
        view = artifact_view(asset, limit=5)
        size = len(json.dumps(view, ensure_ascii=False, allow_nan=False).encode())
        if total + size > 24 * 1024:
            view = {k: meta[k] for k in ("id", "kind", "deliverable_id", "label")}
            view.update(
                status="completed",
                preview_only=True,
                row_count=meta.get("row_count"),
                notice="Read this registered artifact by ID for values/rows; full output is saved.",
            )
            size = len(json.dumps(view, ensure_ascii=False).encode())
        receipts.append(view)
        total += size
    return receipts
