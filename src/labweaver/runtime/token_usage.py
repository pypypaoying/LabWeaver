"""Numeric-only request sizing and provider-reported cache measurements."""

import json


def _count(value):
    return value if type(value) is int and value >= 0 else None


def cache_measurements(messages):
    """Do not infer cache misses or prices from missing provider metadata."""
    hits, misses, writes = [], [], []
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("input_token_details") or {}
        details = details if isinstance(details, dict) else {}
        metadata = getattr(message, "response_metadata", None) or {}
        raw = metadata.get("token_usage") if isinstance(metadata, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        hit = _count(details.get("cache_read"))
        hits.append(hit if hit is not None else _count(raw.get("prompt_cache_hit_tokens")))
        misses.append(_count(raw.get("prompt_cache_miss_tokens")))
        writes.append(_count(details.get("cache_creation")))
    def total(values):
        return sum(values) if values and all(v is not None for v in values) else None
    return {"read_tokens": total(hits), "miss_tokens": total(misses), "write_tokens": total(writes)}


def context_footprint(messages, system_content, tools):
    """UTF-8 logical payload sizes, not exact provider tokenizer attribution.

    No prompt, code, tool argument or credential is retained by this function.
    Failure to size unusual custom tools must not interrupt actual execution.
    """
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    try:
        message_bytes = sum(size({"type": getattr(m, "type", "unknown"),
                                  "content": getattr(m, "content", ""),
                                  "tool_calls": getattr(m, "tool_calls", [])}) for m in messages)
    except Exception:
        message_bytes = None
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool
        schema_bytes = size([convert_to_openai_tool(t) for t in tools])
    except Exception:
        schema_bytes = None
    return {"system_bytes": len(system_content.encode("utf-8")),
            "message_bytes": message_bytes, "tool_schema_bytes": schema_bytes,
            "measurement": "logical_utf8_bytes_not_provider_tokens"}
