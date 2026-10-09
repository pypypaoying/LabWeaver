"""Model formatting tolerance must never accept ambiguous or invalid evidence."""

import pytest
from labweaver.runtime.evidence import parse_model_object


@pytest.mark.parametrize("content", [
    '{"status":"completed","artifact_ids":["a"]}',
    '```json\n{"status":"completed","artifact_ids":["a"]}\n```',
    '产物已生成。\n\n```json\n{"status":"completed","artifact_ids":["a"]}\n```',
    '指标：\n{"value":5}\n最终结果：\n{"status":"completed","artifact_ids":["a"]}',
])
def test_terminal_object_ignores_only_wrapper_text(content):
    assert parse_model_object(content, {"status"}) == {
        "status": "completed", "artifact_ids": ["a"]
    }


@pytest.mark.parametrize("content", [
    '{"status":"completed"}\n还有正文',
    '{"status":"error"}\n{"status":"completed"}',
    '{"status":"error","status":"completed"}',
    '{"status":"completed","value":NaN}',
    '伪造摘要，没有 JSON',
    '{"broken":\n{"status":"completed"}}',
])
def test_ambiguous_malformed_or_trailing_content_rejected(content):
    with pytest.raises(ValueError):
        parse_model_object(content, {"status"})
