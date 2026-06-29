"""Tests for registry extensions: profiles, graph, resources, validation, async."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from src.analysis.graph import (
    build_dependency_graph,
    compute_execution_levels,
    detect_cycles,
    to_mermaid,
)
from src.analysis.registry import (
    _ANALYZER_REGISTRY,
    ensure_analyzers_loaded,
    get_analyzer_registry,
    register_analyzer,
    register_analyzer_class,
    resolve_analyzers_for_profile,
    run_analyzers,
    run_analyzers_async,
)
from src.analysis.resources import ModelResourcePool
from src.analysis.schemas import AnalyzerResultRegistry, validate_analyzer_result
from src.core.models import AnalysisContext, Segment


@pytest.fixture(autouse=True)
def _reload_registry():
    saved = dict(_ANALYZER_REGISTRY)
    yield
    _ANALYZER_REGISTRY.clear()
    _ANALYZER_REGISTRY.update(saved)


def test_autodiscover_registers_core_analyzers():
    ensure_analyzers_loaded()
    names = set(get_analyzer_registry().keys())
    assert "sentiment" in names
    assert "empathy" in names
    assert "negation" in names
    assert "root_cause" in names
    assert "predictive" in names


def test_resolve_analyzers_for_profile_callcenter():
    selected = resolve_analyzers_for_profile("callcenter")
    assert selected is not None
    assert "sentiment" in selected
    assert "compliance_risk" in selected
    assert "spoken_normalizer" not in selected


def test_resolve_explicit_overrides_profile():
    selected = resolve_analyzers_for_profile("callcenter", explicit_selected=["sentiment"])
    assert selected == ["sentiment"]


def test_profile_default_none_for_unknown_without_yaml():
    selected = resolve_analyzers_for_profile("forum")
    assert selected is None


def test_execution_levels_no_cycle():
    ensure_analyzers_loaded()
    graph = build_dependency_graph(get_analyzer_registry())
    assert not detect_cycles(graph)
    levels = compute_execution_levels(graph)
    assert levels
    assert "sentiment" in levels[0]


def test_mermaid_export():
    ensure_analyzers_loaded()
    graph = build_dependency_graph(
        get_analyzer_registry(), selected={"trajectory", "sentiment", "emotion"}
    )
    mermaid = to_mermaid(graph)
    assert "flowchart BT" in mermaid
    assert "sentiment --> trajectory" in mermaid or "emotion --> trajectory" in mermaid


def test_resource_pool_reuses_sentiment_pipeline():
    pool = ModelResourcePool(maxsize=4)
    calls = {"n": 0}

    def _factory():
        calls["n"] += 1
        return object()

    key = ("sentiment", "mock-model", "cpu", "False")
    p1 = pool._get_or_create(key, _factory)
    p2 = pool._get_or_create(key, _factory)
    assert p1 is p2
    assert calls["n"] == 1


def test_selective_instantiation_count():
    from src.analysis.registry import get_analyzers_for_run

    ensure_analyzers_loaded()
    subset = get_analyzers_for_run({"sentiment", "intent"})
    assert set(subset.keys()) == {"sentiment", "intent"}


def test_result_validation_warn_mode():
    class StrictEmpathy(BaseModel):
        overall_empathy: float

    AnalyzerResultRegistry.register("test_empathy", StrictEmpathy)
    raw = {"overall_empathy": "not-a-number"}
    out = validate_analyzer_result("test_empathy", raw, mode="warn")
    assert "_validation_warning" in out


def test_result_validation_strict_mode():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        validate_analyzer_result("empathy", {"overall_empathy": "bad"}, mode="strict")


def test_async_execution_order():
    order: list[str] = []

    @register_analyzer("async_a")
    class A:
        @property
        def name(self) -> str:
            return "async_a"

        @property
        def requires(self) -> list[str]:
            return []

        def analyze(self, ctx: AnalysisContext) -> str:
            order.append("A")
            return "a"

    @register_analyzer("async_b")
    class B:
        @property
        def name(self) -> str:
            return "async_b"

        @property
        def requires(self) -> list[str]:
            return ["async_a"]

        def analyze(self, ctx: AnalysisContext) -> str:
            order.append("B")
            return "b"

    ctx = AnalysisContext()
    asyncio.run(run_analyzers_async(ctx, selected=["async_b"]))
    assert order == ["A", "B"]
    assert ctx.results["async_b"] == "b"


def test_run_analyzers_async_mode_flag():
    @register_analyzer("sync_x")
    class X:
        @property
        def name(self) -> str:
            return "sync_x"

        @property
        def requires(self) -> list[str]:
            return []

        def analyze(self, ctx: AnalysisContext) -> int:
            return 1

    ctx = AnalysisContext()
    run_analyzers(ctx, selected=["sync_x"], async_mode=True)
    assert ctx.results["sync_x"] == 1


def test_dynamic_register_analyzer_class():
    class Dyn:
        @property
        def name(self) -> str:
            return "dyn_test"

        @property
        def requires(self) -> list[str]:
            return []

        def analyze(self, ctx: AnalysisContext) -> dict:
            return {"ok": True}

    register_analyzer_class("dyn_test", Dyn)
    ctx = AnalysisContext()
    run_analyzers(ctx, selected=["dyn_test"])
    assert ctx.results["dyn_test"]["ok"] is True


def test_strict_validation_env(monkeypatch):
    monkeypatch.setenv("ANALYZER_VALIDATION_MODE", "off")
    from src.analysis.schemas import get_validation_mode

    assert get_validation_mode() == "off"


def test_callcenter_profile_strict_validation(monkeypatch):
    """Callcenter default analyzers must pass registered schema validation in strict mode."""
    ensure_analyzers_loaded()
    from src.analysis.intent import IntentAnalyzer
    from src.analysis.sentiment import SentimentAnalyzer

    monkeypatch.setattr(
        SentimentAnalyzer,
        "analyze",
        lambda self, ctx: [{"label": "neutral", "score": 0.5} for _ in (ctx.segments or [])],
    )
    monkeypatch.setattr(
        IntentAnalyzer,
        "analyze",
        lambda self, ctx: [
            {"intent": "billing_inquiry", "confidence": 0.8} for _ in (ctx.segments or [])
        ],
    )

    selected = resolve_analyzers_for_profile("callcenter")
    assert selected is not None
    segments = [
        Segment(
            start=0.0, end=1.0, text="Hej, jag har en fråga om min faktura.", speaker="SPEAKER_0"
        ),
        Segment(start=1.0, end=2.0, text="Absolut, jag hjälper dig gärna.", speaker="SPEAKER_1"),
    ]
    ctx = AnalysisContext(segments=segments)
    run_analyzers(ctx, selected=selected, validation_mode="strict")
    for name in selected:
        assert name in ctx.results


def test_yaml_plugin_allowlist_blocks_unknown_module(tmp_path):
    cfg = tmp_path / "analyzers.yaml"
    cfg.write_text(
        """
allowlist_prefixes: [src.analysis.]
plugins:
  - module: evil.plugin
    class: Bad
    enabled: true
""",
        encoding="utf-8",
    )
    from src.analysis.registry import load_analyzers_from_config

    before = set(get_analyzer_registry().keys())
    load_analyzers_from_config(cfg, graceful=True)
    after = set(get_analyzer_registry().keys())
    assert "Bad" not in after
    assert after >= before


def test_graph_snapshot_stable_subset():
    """CI guard: mermaid export for callcenter subset must contain core edges."""
    ensure_analyzers_loaded()
    selected = resolve_analyzers_for_profile("callcenter")
    graph = build_dependency_graph(get_analyzer_registry(), selected=set(selected or []))
    mermaid = to_mermaid(graph)
    assert "sentiment" in mermaid
    assert "compliance_risk" in mermaid or "customer_effort" in mermaid


def test_builtin_schemas_validate_empathy_shape():
    from src.analysis.schemas import EmpathyResult, validate_analyzer_result

    sample = {"overall_empathy": 72.5, "per_segment": [], "coaching_tips": ["Bra jobbat"]}
    out = validate_analyzer_result("empathy", sample, mode="strict")
    assert EmpathyResult.model_validate(out).overall_empathy == 72.5
