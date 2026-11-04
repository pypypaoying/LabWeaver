"""Blind formatting tests; fake models only and no calls to online providers."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def formatter():
    path = Path(__file__).resolve().parents[1] / "evaluation/answer_format.py"
    spec = importlib.util.spec_from_file_location("blind_answer_format_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model:
    def __init__(self, content="@mean_fare[34.65]", *, usage=None, error=None):
        self.content = content
        self.usage = usage
        self.error = error
        self.calls = []

    def bind_tools(self, *args, **kwargs):
        raise AssertionError("The formatting model must not receive tools")

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content, usage_metadata=self.usage)


def question():
    # Extra source fields must not reach the formatting model.
    return {"question": "Calculate the mean fare.", "format": "@mean_fare[value]",
            "constraints": "Do not pass constraints as hidden-solving hints.",
            "common_answers": [["mean_fare", "SECRET_ORACLE_VALUE"]],
            "file_name": "PRIVATE_CSV_PATH", "csv": "PRIVATE_SOURCE_RECORDS",
            "artifacts": {"value": "PRIVATE_ARTIFACTS"}}


def test_formatter_passes_only_public_question_format_and_final_answer(formatter):
    model = Model()
    result = formatter.format_answer(model, question=question(), answer="平均票价为34.65。")
    assert result["response"] == "@mean_fare[34.65]"
    assert result["called"] is True
    assert len(model.calls) == 1
    messages = model.calls[0]
    assert [m.type for m in messages] == ["system", "human"]
    payload = json.loads(messages[1].content)
    assert payload == {"question": "Calculate the mean fare.", "requested_format": "@mean_fare[value]",
                       "analyst_answer": "平均票价为34.65。"}
    serialized = "\n".join(m.content for m in messages)
    assert all(value not in serialized for value in ["SECRET_ORACLE_VALUE", "PRIVATE_CSV_PATH", "PRIVATE_SOURCE_RECORDS", "PRIVATE_ARTIFACTS"])


def test_formatter_filters_unrequested_output_keys(formatter):
    model = Model("@unrequested[999]\n@mean_fare[34.65]\nOther text")
    result = formatter.format_answer(model, question=question(), answer="平均票价34.65。")
    assert result["response"] == "@mean_fare[34.65]"
    assert result["raw_response"] == model.content


@pytest.mark.parametrize("answer", ["", " ", "\n\t"])
def test_blank_answers_never_call_model(formatter, answer):
    model = Model(error=AssertionError("No call permitted"))
    result = formatter.format_answer(model, question=question(), answer=answer)
    assert result["response"] == "__NO_ANSWER__"
    assert result["called"] is False
    assert result["usage"] is None
    assert not model.calls


def test_formatter_never_retries_provider_failure(formatter):
    model = Model(error=TimeoutError("Fake provider timeout"))
    with pytest.raises(TimeoutError):
        formatter.format_answer(model, question=question(), answer="平均票价34.65。")
    assert len(model.calls) == 1


def test_token_usage_is_preserved_without_fabrication(formatter):
    usage = {"input_tokens": 121, "output_tokens": 13, "total_tokens": 134,
             "input_token_details": {"cache_read": 42}}
    model = Model(usage=usage)
    result = formatter.format_answer(model, question=question(), answer="平均票价34.65。")
    assert result["usage"] == usage
    assert result["usage"] is not usage


def test_absent_token_usage_remains_unknown(formatter):
    result = formatter.format_answer(Model(), question=question(), answer="平均票价34.65。")
    assert result["usage"] is None


@pytest.mark.parametrize("content", [None, ["unexpected list"], {}, "No stated answer"])
def test_invalid_or_empty_formatting_is_a_recorded_no_answer(formatter, content):
    model = Model(content)
    result = formatter.format_answer(model, question=question(), answer="没有计算结果。")
    assert result["response"] == "__NO_ANSWER__"
    assert result["called"] is True
    assert len(model.calls) == 1


def test_explicit_original_wrong_answer_cannot_be_corrected_by_formatter(formatter):
    result = formatter.format_answer(Model("@mean_fare[34.65]"), question=question(),
                                     answer="@mean_fare[999]\n另有参考数值34.65。")
    assert result["response"] == "@mean_fare[999]"


@pytest.mark.parametrize("answer, output", [
    ("两个输入值为3和4，未完成计算。", "@mean_fare[7]"),
    ("平均票价34.65。", "@mean_fare[34.65001]"),
    ("计算失败，暂无结果。", "@mean_fare[34.65]"),
    ("给出了50%，没有换算结果。", "@mean_fare[0.5]"),
])
def test_formatting_cannot_add_or_calculate_a_new_numeric_value(formatter, answer, output):
    result = formatter.format_answer(Model(output), question=question(), answer=answer)
    assert result["response"] == "__NO_ANSWER__"
    assert result["rejected"]


def test_conflicting_formatter_values_are_omitted_not_last_value_wins(formatter):
    result = formatter.format_answer(Model("@mean_fare[34.65]\n@mean_fare[35]"), question=question(),
                                     answer="报告包含34.65和35两个不兼容数值。")
    assert result["response"] == "__NO_ANSWER__"


@pytest.mark.parametrize("answer", ["平均票价为34.65。", "The mean fare is 34.65.", "Mean fare: **34.65**."])
def test_numeric_grounding_handles_chinese_and_sentence_punctuation(formatter, answer):
    result = formatter.format_answer(Model(), question=question(), answer=answer)
    assert result["response"] == "@mean_fare[34.65]"


def test_formatter_cannot_invent_a_non_numeric_answer(formatter):
    result = formatter.format_answer(Model("@mean_fare[London]"), question=question(), answer="没有城市名称或分析结果。")
    assert result["response"] == "__NO_ANSWER__"


def test_formatter_may_copy_a_city_explicitly_stated_in_answer(formatter):
    city_question = {"question": "Which city has the highest value?", "format": "@city[name]"}
    result = formatter.format_answer(Model("@city[London]"), question=city_question, answer="最高的城市为 London。")
    assert result["response"] == "@city[London]"
