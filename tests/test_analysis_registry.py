"""Tests for the modular analysis registry and topological execution."""

from __future__ import annotations

import pytest

from src.analysis.base import Analyzer
from src.analysis.registry import _ANALYZER_REGISTRY, ensure_analyzers_loaded, register_analyzer, run_analyzers
from src.core.errors import AnalysisError
from src.core.models import AnalysisContext


@pytest.fixture(autouse=True)
def clean_registry():
    """Fixture to ensure the global registry is cleared of test-specific mock analyzers."""
    ensure_analyzers_loaded()
    saved_registry = dict(_ANALYZER_REGISTRY)
    yield
    _ANALYZER_REGISTRY.clear()
    _ANALYZER_REGISTRY.update(saved_registry)


def test_topological_execution():
    """Test that analyzers execute in correct topological order based on dependencies."""
    execution_order = []

    @register_analyzer("mock_c")
    class AnalyzerC(Analyzer):
        @property
        def name(self) -> str:
            return "mock_c"

        @property
        def requires(self) -> list[str]:
            return ["mock_b"]

        def analyze(self, ctx: AnalysisContext) -> str:
            execution_order.append("C")
            return "result_c"

    @register_analyzer("mock_b")
    class AnalyzerB(Analyzer):
        @property
        def name(self) -> str:
            return "mock_b"

        @property
        def requires(self) -> list[str]:
            return ["mock_a"]

        def analyze(self, ctx: AnalysisContext) -> str:
            execution_order.append("B")
            return "result_b"

    @register_analyzer("mock_a")
    class AnalyzerA(Analyzer):
        @property
        def name(self) -> str:
            return "mock_a"

        @property
        def requires(self) -> list[str]:
            return []

        def analyze(self, ctx: AnalysisContext) -> str:
            execution_order.append("A")
            return "result_a"

    ctx = AnalysisContext()
    # Execute only mock_c, which should transitively pull in mock_b and mock_a
    results = run_analyzers(ctx, selected=["mock_c"])

    # Verify order of execution: A must run before B, which must run before C
    assert execution_order == ["A", "B", "C"]
    assert results["mock_a"] == "result_a"
    assert results["mock_b"] == "result_b"
    assert results["mock_c"] == "result_c"


def test_circular_dependency_detection():
    """Test that circular dependencies among analyzers raise an AnalysisError."""

    # We define cyclic dependencies: D requires E, E requires D
    @register_analyzer("mock_d")
    class AnalyzerD(Analyzer):
        @property
        def name(self) -> str:
            return "mock_d"

        @property
        def requires(self) -> list[str]:
            return ["mock_e"]

        def analyze(self, ctx: AnalysisContext) -> str:
            return "d"

    @register_analyzer("mock_e")
    class AnalyzerE(Analyzer):
        @property
        def name(self) -> str:
            return "mock_e"

        @property
        def requires(self) -> list[str]:
            return ["mock_d"]

        def analyze(self, ctx: AnalysisContext) -> str:
            return "e"

    ctx = AnalysisContext()
    with pytest.raises(AnalysisError, match="Circular dependency detected"):
        run_analyzers(ctx, selected=["mock_d"])


def test_analyzer_error_isolation():
    """Test that a failure in one analyzer does not crash the entire run."""

    @register_analyzer("mock_failing")
    class FailingAnalyzer(Analyzer):
        @property
        def name(self) -> str:
            return "mock_failing"

        @property
        def requires(self) -> list[str]:
            return []

        def analyze(self, ctx: AnalysisContext) -> str:
            raise ValueError("Something went wrong!")

    @register_analyzer("mock_successful")
    class SuccessfulAnalyzer(Analyzer):
        @property
        def name(self) -> str:
            return "mock_successful"

        @property
        def requires(self) -> list[str]:
            return []

        def analyze(self, ctx: AnalysisContext) -> str:
            return "success"

    ctx = AnalysisContext()
    results = run_analyzers(ctx, selected=["mock_failing", "mock_successful"])

    # mock_failing should NOT have written results because it crashed,
    # but mock_successful should have run and stored its output safely.
    assert "mock_failing" not in results
    assert results["mock_successful"] == "success"
