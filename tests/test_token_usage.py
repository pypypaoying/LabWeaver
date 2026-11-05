from types import SimpleNamespace

from labweaver.runtime.token_usage import cache_measurements, context_footprint


def test_absent_cache_metadata_is_unknown_and_zero_is_preserved():
    assert cache_measurements([SimpleNamespace()]) == {
        "read_tokens": None, "miss_tokens": None, "write_tokens": None}
    m = SimpleNamespace(usage_metadata={"input_token_details": {"cache_read": 0}},
                        response_metadata={"token_usage": {"prompt_cache_miss_tokens": 11}})
    assert cache_measurements([m]) == {"read_tokens": 0, "miss_tokens": 11, "write_tokens": None}


def test_partial_metadata_does_not_claim_complete_cache_totals():
    m = SimpleNamespace(usage_metadata={"input_token_details": {"cache_read": 8}}, response_metadata={})
    assert cache_measurements([m, SimpleNamespace()])["read_tokens"] is None


def test_invalid_cache_counts_are_not_accepted():
    for value in (-1, True, "3"):
        m = SimpleNamespace(response_metadata={"token_usage": {"prompt_cache_hit_tokens": value}})
        assert cache_measurements([m])["read_tokens"] is None


def test_footprint_contains_only_numeric_sizes_not_private_content():
    secret = "private-token-measurement-sentinel"
    message = SimpleNamespace(type="human", content=secret)
    result = context_footprint([message], "任务", [])
    assert result["system_bytes"] == len("任务".encode())
    assert result["message_bytes"] >= len(secret)
    assert secret not in str(result)
    assert result["measurement"] == "logical_utf8_bytes_not_provider_tokens"


def test_custom_provider_metadata_does_not_break_execution_measurement():
    for malformed in ("unknown", 123, [1]):
        m = SimpleNamespace(usage_metadata={"input_token_details": malformed},
                            response_metadata={"token_usage": malformed})
        assert cache_measurements([m])["read_tokens"] is None


def test_schema_measurement_runtime_failure_is_unknown(monkeypatch):
    import langchain_core.utils.function_calling as conversion
    def unsupported(tool):
        raise RuntimeError("Custom schema unavailable")
    monkeypatch.setattr(conversion, "convert_to_openai_tool", unsupported)
    assert context_footprint([], "rules", [object()])["tool_schema_bytes"] is None
