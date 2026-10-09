"""Typed tool arguments make the available statistics explicit to the model."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictFloat, StrictStr


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: StrictInt = Field(ge=1, description="One-based CSV column position")
    op: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "not_in",
        "is_missing",
        "not_missing",
    ] = Field(
        description="eq/ne compare values, gt/gte/lt/lte compare numeric bounds, in/not_in test membership, is_missing/not_missing test whitespace-only cells"
    )
    value: (
        StrictStr
        | StrictInt
        | StrictFloat
        | list[StrictStr | StrictInt | StrictFloat]
        | None
    ) = None


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["count", "sum", "mean", "min", "max"] = Field(
        description="sum totals values; count counts records, not values. Numeric aggregates skip whitespace-only missing cells and reject other invalid or nonfinite values"
    )
    column: StrictInt | None = Field(
        description="One-based value column; null only for row count"
    )
    alias: str = Field(min_length=1, max_length=80)


class OrderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(
        description="Metric alias or positional group key such as column_2"
    )
    direction: Literal["asc", "desc"]


class AnalysisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: list[FilterSpec] = Field(max_length=20)
    group_by: list[StrictInt] = Field(
        description="One-based group positions; preserve raw nonmissing labels, no implicit category merging"
    )
    metrics: list[MetricSpec] = Field(min_length=1, max_length=20)
    order_by: list[OrderSpec]
    top_k: StrictInt | None = Field(
        default=None,
        ge=1,
        description="Use null or omit to return ALL groups. Set a positive integer only when the user requests a top-N subset; no automatic row cap.",
    )


class AnalysisArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: AnalysisSpec
