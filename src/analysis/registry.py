"""Registry for managing and executing text analyzers with topological dependency resolution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ..core.errors import AnalysisError
from ..core.models import AnalysisContext
from .base import Analyzer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type[Analyzer])

# Global storage for registered analyzer classes or factories
_ANALYZER_REGISTRY: dict[str, type[Analyzer] | Callable[[], Analyzer]] = {}


def register_analyzer(name: str) -> Callable[[T], T]:
    """Decorator to register an Analyzer class with a unique name."""

    def decorator(cls: T) -> T:
        _ANALYZER_REGISTRY[name] = cls
        logger.debug("Registered analyzer: %s", name)
        return cls

    return decorator


def get_registered_analyzers(
    analyzer_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Analyzer]:
    """Instantiate and return all registered analyzers.

    Args:
        analyzer_configs: Optional mapping of analyzer name → keyword arguments
            passed to the analyzer's constructor.  Unrecognised names are
            silently ignored; unregistered names are not required to appear.

    Returns:
        Dict of instantiated analyzers keyed by their registered name.
    """
    configs = analyzer_configs or {}
    analyzers: dict[str, Analyzer] = {}
    for name, factory in _ANALYZER_REGISTRY.items():
        kwargs = configs.get(name, {})
        try:
            analyzers[name] = factory(**kwargs)
        except Exception as e:
            logger.error("Failed to instantiate analyzer '%s': %s", name, e)
    return analyzers


def run_analyzers(
    ctx: AnalysisContext,
    selected: list[str] | None = None,
    analyzer_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute analyzers in topological order, respecting their dependencies.

    Args:
        ctx: The shared AnalysisContext carrying inputs and accumulating results.
        selected: Optional list of analyzer names to run.  If *None*, runs all
            registered analyzers.  If specified, their transitive dependencies
            are resolved and included automatically.
        analyzer_configs: Optional mapping of analyzer name → constructor kwargs
            forwarded to :func:`get_registered_analyzers`.  Use this to pass
            model names, devices, or lexicon settings without modifying the
            registry itself.

    Returns:
        The results dictionary containing analyzer outputs (``ctx.results``).
    """
    all_analyzers = get_registered_analyzers(analyzer_configs)

    # 1. Resolve which analyzers to run (selected + their transitives)
    to_run: set[str] = set()
    if selected is None:
        to_run = set(all_analyzers.keys())
    else:
        def resolve_deps(name: str) -> None:
            if name in to_run:
                return
            to_run.add(name)
            analyzer = all_analyzers.get(name)
            if analyzer is not None:
                for req in analyzer.requires:
                    if req not in all_analyzers:
                        logger.warning(
                            "Required analyzer '%s' for '%s' is not registered!", req, name
                        )
                    resolve_deps(req)

        for s in selected:
            if s not in all_analyzers:
                logger.error("Selected analyzer '%s' is not registered!", s)
                continue
            resolve_deps(s)

    # Filter to only the resolved set
    active_analyzers = {name: all_analyzers[name] for name in to_run if name in all_analyzers}

    # 2. Topological sort (DFS algorithm)
    visited: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
    execution_order: list[str] = []

    def dfs(name: str) -> None:
        state = visited.get(name, 0)
        if state == 1:
            raise AnalysisError(f"Circular dependency detected involving '{name}'")
        if state == 2:
            return

        visited[name] = 1  # visiting
        analyzer = active_analyzers.get(name)
        if analyzer is not None:
            for req in analyzer.requires:
                if req in active_analyzers:
                    dfs(req)
        visited[name] = 2  # visited
        execution_order.append(name)

    for name in active_analyzers:
        if name not in visited:
            dfs(name)

    logger.info("Executing analyzers in order: %s", " -> ".join(execution_order))

    # 3. Execute in order with error isolation
    for name in execution_order:
        analyzer = active_analyzers[name]
        logger.debug("Running analyzer: %s", name)
        try:
            # Verify dependencies succeeded before running
            deps_ok = True
            for req in analyzer.requires:
                if req not in ctx.results:
                    logger.warning(
                        "Skipping analyzer '%s' because dependency '%s' did not produce results.",
                        name,
                        req,
                    )
                    deps_ok = False
                    break

            if deps_ok:
                res = analyzer.analyze(ctx)
                ctx.results[name] = res
                logger.debug("Analyzer '%s' completed successfully", name)
        except Exception as e:
            logger.error("Analyzer '%s' failed: %s", name, e, exc_info=True)
            # Isolate the error so subsequent, independent analyzers can still run

    return ctx.results
