"""Pydantic result schemas for analyzer outputs (opt-in, progressive strictness)."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

ValidationMode = Literal["off", "warn", "strict"]

_RESULT_SCHEMAS: dict[str, type[BaseModel]] = {}


def register_result_schema(analyzer_name: str, schema: type[BaseModel]) -> type[BaseModel]:
    """Decorator to associate a Pydantic schema with an analyzer result."""
    _RESULT_SCHEMAS[analyzer_name] = schema
    return schema


class AnalyzerResultRegistry:
    """Central registry of analyzer output schemas."""

    @classmethod
    def register(cls, analyzer_name: str, schema: type[BaseModel]) -> None:
        _RESULT_SCHEMAS[analyzer_name] = schema

    @classmethod
    def get(cls, analyzer_name: str) -> type[BaseModel] | None:
        return _RESULT_SCHEMAS.get(analyzer_name)

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(_RESULT_SCHEMAS.keys())


def get_validation_mode() -> ValidationMode:
    raw = os.getenv("ANALYZER_VALIDATION_MODE", "warn").strip().lower()
    if raw in ("off", "warn", "strict"):
        return raw  # type: ignore[return-value]
    return "warn"


def validate_analyzer_result(
    analyzer_name: str,
    raw: Any,
    mode: ValidationMode | None = None,
) -> Any:
    """Validate analyzer output; warn or strict based on mode."""
    effective = mode or get_validation_mode()
    if effective == "off":
        return raw

    schema = AnalyzerResultRegistry.get(analyzer_name)
    if schema is None:
        return raw

    try:
        if isinstance(raw, BaseModel):
            return raw.model_dump()
        validated = schema.model_validate(raw)
        return validated.model_dump()
    except ValidationError as exc:
        if effective == "strict":
            raise
        logger.warning("Result validation failed for '%s': %s", analyzer_name, exc)
        if isinstance(raw, dict):
            return {**raw, "_validation_warning": str(exc)}
        return raw


class EmpathyResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    overall_empathy: float = Field(ge=0, le=100)
    per_segment: list[dict[str, Any]] = Field(default_factory=list)
    coaching_tips: list[str] = Field(default_factory=list)


class CustomerEffortResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    overall_ces: float = Field(ge=0, le=100)
    per_segment: list[dict[str, Any]] = Field(default_factory=list)
    coaching_tips: list[str] = Field(default_factory=list)


class ComplianceRiskResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    overall_risk_level: Literal["low", "medium", "high"]
    flagged_segments: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str = ""


class ResolutionProbabilityResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    resolution_probability: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=100)
    recommended_action: str = ""


class NegationSegmentResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    has_negation: bool
    negation_count: int = Field(ge=0)


def get_typed_result(
    results: dict[str, Any],
    analyzer_name: str,
    schema: type[BaseModel] | None = None,
) -> BaseModel | None:
    """Parse a single analyzer result from report ``results`` dict."""
    raw = results.get(analyzer_name)
    if raw is None:
        return None
    model = schema or AnalyzerResultRegistry.get(analyzer_name)
    if model is None:
        return None
    try:
        return model.model_validate(raw)
    except ValidationError:
        return None


def _register_builtin_schemas() -> None:
    AnalyzerResultRegistry.register("empathy", EmpathyResult)
    AnalyzerResultRegistry.register("customer_effort", CustomerEffortResult)
    AnalyzerResultRegistry.register("compliance_risk", ComplianceRiskResult)
    AnalyzerResultRegistry.register("resolution_probability", ResolutionProbabilityResult)
    AnalyzerResultRegistry.register("negation", NegationSegmentResult)


_register_builtin_schemas()