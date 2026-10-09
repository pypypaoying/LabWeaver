"""Trace and citation checks do not prove semantic correctness."""

import json
import math
import re
from typing import Any
from langchain_core.messages import ToolMessage


def _json_safe(value: Any) -> Any:
    """Keep reports JSON-compatible without serializing model objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return f"<{type(value).__name__}>"


def _profile_is_completed(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("status") != "completed":
        return False
    source = value.get("source")
    if not isinstance(source, dict):
        return False
    if not isinstance(source.get("name"), str) or not source["name"]:
        return False
    sha256 = source.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        return False
    if any(character not in "0123456789abcdefABCDEF" for character in sha256):
        return False
    for key in ("row_count", "column_count"):
        if type(value.get(key)) is not int or value[key] < 0:
            return False
    for key in ("columns", "sample_rows", "warnings"):
        if not isinstance(value.get(key), list):
            return False
    return len(value["columns"]) == value["column_count"]


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def parse_model_object(content: str, required_keys: set[str]) -> dict:
    """Read one terminal protocol object, allowing a prose/fence wrapper.

    Wrapper text is never used as an answer or evidence. Ambiguous protocol
    objects, trailing prose, duplicate keys and non-finite JSON are rejected.
    Callers still validate the schema and actual execution evidence.
    """
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON")

    decoder = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=invalid_constant)
    candidates = []
    for match in re.finditer(r"(?m)^[ \t]*(?:```(?:json)?[ \t]*\n[ \t]*)?(\{)", content):
        start = match.start(1)
        try:
            value, end = decoder.raw_decode(content, start)
        except ValueError:
            continue
        if isinstance(value, dict) and required_keys.issubset(value):
            candidates.append((value, content[end:].strip()))
    if len(candidates) != 1 or candidates[0][1] not in {"", "```"}:
        raise ValueError("Expected one terminal JSON protocol object")
    return candidates[0][0]


def _trace_messages(messages: list[Any]) -> list[dict]:
    trace = []
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            trace.append(
                {
                    "kind": "tool_call",
                    "name": call.get("name"),
                    "args": _json_safe(call.get("args", {})),
                    "id": call.get("id"),
                }
            )
        if isinstance(message, ToolMessage):
            trace.append(
                {
                    "kind": "tool_result",
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                    "content": _message_content(message),
                    "status": message.status,
                }
            )
    return trace


def _chunk_is_valid(chunk: Any) -> bool:
    if not isinstance(chunk, dict):
        return False
    chunk_id, source_id = chunk.get("id"), chunk.get("source_id")
    if not isinstance(chunk_id, str) or not re.fullmatch(
        r"D[1-9]\d*-C[1-9]\d*", chunk_id
    ):
        return False
    if source_id != chunk_id.split("-")[0]:
        return False
    if not isinstance(chunk.get("name"), str) or not chunk["name"]:
        return False
    digest = chunk.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        return False
    if not isinstance(chunk.get("text"), str) or not 0 < len(chunk["text"]) <= 800:
        return False
    location = chunk.get("location")
    if not isinstance(location, dict):
        return False
    if set(location) == {"page"}:
        return type(location["page"]) is int and 0 < location["page"] <= 100
    return (
        set(location) == {"line_start", "line_end"}
        and type(location["line_start"]) is int
        and type(location["line_end"]) is int
        and 0 < location["line_start"] <= location["line_end"]
    )


def _validate_citations(
    answer: str, chunks: list[dict]
) -> tuple[list[str], str | None]:
    """Check retrieved IDs and explicit source annotations, not semantic entailment."""
    available = {chunk["id"]: chunk for chunk in chunks}
    ids = []
    for match in re.finditer(r"\[(D\d+-C\d+)\]", answer):
        chunk_id = match.group(1)
        if chunk_id not in available:
            return ids, "invalid_citation"
        if chunk_id not in ids:
            ids.append(chunk_id)
        chunk = available[chunk_id]
        suffix = answer[match.end() :].lstrip()
        if suffix.startswith(("（", "(")):
            closing = "）" if suffix[0] == "（" else ")"
            end_position = suffix.find(closing, 1)
            if not 0 < end_position <= 300:
                return ids, "invalid_citation_source"
            source = suffix[1:end_position]
            named = re.search(r"([^，,；;\n]+\.(?:md|txt|pdf))", source, flags=re.I)
            if named and named.group(1).strip() != chunk["name"]:
                return ids, "invalid_citation_source"
            position = re.search(
                r"第\s*(\d+)\s*(?:[-–—~至]\s*(\d+))?\s*(行|页)", source
            )
            if position:
                start = int(position.group(1))
                end = int(position.group(2) or start)
                kind = position.group(3)
            else:
                english = re.search(
                    r"\b(lines?|pages?)\s*[:：]?\s*(\d+)\s*(?:[-–—~]\s*(\d+))?",
                    source,
                    re.I,
                )
                if english:
                    start = int(english.group(2))
                    end = int(english.group(3) or start)
                    kind = "页" if english.group(1).lower().startswith("page") else "行"
                elif re.search(r"第.*[行页]|\b(?:lines?|pages?)\b", source, re.I):
                    return ids, "invalid_citation_source"
            if position or english:
                location = chunk["location"]
                if kind == "页":
                    valid = start == end == location.get("page")
                else:
                    valid = (
                        "line_start" in location
                        and location["line_start"]
                        <= start
                        <= end
                        <= location["line_end"]
                    )
                if not valid:
                    return ids, "invalid_citation_source"
    if available and not ids:
        return ids, "missing_citation"
    if not available and not re.search(
        r"资料不足|无命中|没有.*(?:命中|相关|依据)|未.*(?:匹配|检索到|覆盖)|无法.*(?:确定|支持)",
        answer,
    ):
        return ids, "missing_insufficiency_notice"
    return ids, None


def _source_list(citations: list[str], chunks: list[dict]) -> str:
    """Attach exact source locations using recorded hits rather than model text."""
    by_id = {chunk["id"]: chunk for chunk in chunks}
    lines = []
    for chunk_id in citations:
        chunk = by_id[chunk_id]
        location = chunk["location"]
        position = (
            f"第{location['page']}页"
            if "page" in location
            else f"第{location['line_start']}–{location['line_end']}行"
        )
        lines.append(f"- [{chunk_id}] {chunk['name']}，{position}")
    return "\n\n引用来源（由执行记录生成）：\n" + "\n".join(lines) if lines else ""
