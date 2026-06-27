"""Base classes and protocols for text analyzers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.models import AnalysisContext


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for conversation and text analyzers."""

    @property
    def name(self) -> str:
        """Unique identifier/name for this analyzer."""
        ...

    @property
    def requires(self) -> list[str]:
        """List of names of analyzers that must run before this analyzer."""
        ...

    def analyze(self, ctx: AnalysisContext) -> Any:
        """Run the analysis step on the context and return the result.

        Args:
            ctx: The shared AnalysisContext carrying the transcript and previous results.

        Returns:
            The analysis result, which will be stored in ctx.results[self.name].
        """
        ...
