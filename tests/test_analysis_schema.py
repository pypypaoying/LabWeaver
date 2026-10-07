"""Typed Agent arguments retain the deterministic tool's numeric semantics."""

import pytest
from pydantic import ValidationError
from labweaver.tools.analysis_schema import AnalysisArguments


def arguments(value, operation="eq"):
    return {
        "spec": {
            "filters": [{"column": 1, "op": operation, "value": value}],
            "group_by": [],
            "metrics": [{"op": "count", "column": None, "alias": "rows"}],
            "order_by": [],
            "top_k": 5,
        }
    }


@pytest.mark.parametrize("value", [True, False, [True], [False, "yes"]])
def test_boolean_filter_values_cannot_be_coerced_into_numeric_values(value):
    operation = "in" if isinstance(value, list) else "eq"
    with pytest.raises(ValidationError):
        AnalysisArguments.model_validate(arguments(value, operation))


@pytest.mark.parametrize(
    "value", ["001", 9007199254740993, 0.25, ["001", 9007199254740993, 0.25]]
)
def test_identifiers_and_large_integer_filter_values_keep_their_original_types(value):
    operation = "in" if isinstance(value, list) else "eq"
    parsed = AnalysisArguments.model_validate(arguments(value, operation)).model_dump()
    assert parsed["spec"]["filters"][0]["value"] == value
    original = value if isinstance(value, list) else [value]
    actual = parsed["spec"]["filters"][0]["value"]
    actual = actual if isinstance(actual, list) else [actual]
    assert [type(item) for item in actual] == [type(item) for item in original]


@pytest.mark.parametrize("path", ["filter", "group", "metric", "top_k"])
def test_bool_positions_and_limit_are_not_accepted_as_the_integer_one(path):
    request = arguments("x")
    if path == "filter":
        request["spec"]["filters"][0]["column"] = True
    elif path == "group":
        request["spec"]["group_by"] = [True]
    elif path == "metric":
        request["spec"]["metrics"][0]["column"] = True
    else:
        request["spec"]["top_k"] = True
    with pytest.raises(ValidationError):
        AnalysisArguments.model_validate(request)
