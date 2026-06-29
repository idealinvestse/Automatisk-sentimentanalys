"""Dependency graph utilities for the analyzer registry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import AnalysisError
from .base import Analyzer


@dataclass
class AnalyzerNode:
    name: str
    requires: list[str] = field(default_factory=list)
    module: str | None = None


def _requires_of(factory: type[Analyzer] | Callable[[], Analyzer]) -> list[str]:
    try:
        probe = factory() if callable(factory) and not isinstance(factory, type) else factory()
        return list(probe.requires)
    except TypeError:
        inst = factory()  # type: ignore[misc]
        return list(inst.requires)


def build_dependency_graph(
    registry: dict[str, type[Analyzer] | Callable[[], Analyzer]],
    selected: set[str] | None = None,
) -> dict[str, AnalyzerNode]:
    """Build adjacency metadata for all or selected analyzers."""
    names = set(selected) if selected is not None else set(registry.keys())
    graph: dict[str, AnalyzerNode] = {}
    for name in names:
        factory = registry.get(name)
        if factory is None:
            continue
        module = getattr(factory, "__module__", None)
        requires = _requires_of(factory)
        graph[name] = AnalyzerNode(name=name, requires=requires, module=module)
    return graph


def detect_cycles(graph: dict[str, AnalyzerNode]) -> list[list[str]]:
    """Return cycles found via DFS (empty if acyclic)."""
    visited: dict[str, int] = {}
    cycles: list[list[str]] = []
    path: list[str] = []

    def dfs(name: str) -> None:
        state = visited.get(name, 0)
        if state == 1:
            if name in path:
                idx = path.index(name)
                cycles.append(path[idx:] + [name])
            return
        if state == 2:
            return
        visited[name] = 1
        path.append(name)
        node = graph.get(name)
        if node:
            for req in node.requires:
                if req in graph:
                    dfs(req)
        path.pop()
        visited[name] = 2

    for name in graph:
        if name not in visited:
            dfs(name)
    return cycles


def topological_sort(
    graph: dict[str, AnalyzerNode],
) -> list[str]:
    """Topological order; raises AnalysisError on cycle."""
    visited: dict[str, int] = {}
    order: list[str] = []

    def dfs(name: str) -> None:
        state = visited.get(name, 0)
        if state == 1:
            raise AnalysisError(f"Circular dependency detected involving '{name}'")
        if state == 2:
            return
        visited[name] = 1
        node = graph.get(name)
        if node:
            for req in node.requires:
                if req in graph:
                    dfs(req)
        visited[name] = 2
        order.append(name)

    for name in graph:
        if name not in visited:
            dfs(name)
    return order


def compute_execution_levels(graph: dict[str, AnalyzerNode]) -> list[list[str]]:
    """Group analyzers into parallel execution levels respecting dependencies."""
    order = topological_sort(graph)
    level_of: dict[str, int] = {}
    for name in order:
        node = graph[name]
        deps_in_graph = [r for r in node.requires if r in graph]
        level = 0 if not deps_in_graph else max(level_of[r] for r in deps_in_graph) + 1
        level_of[name] = level
    if not level_of:
        return []
    max_level = max(level_of.values())
    levels: list[list[str]] = [[] for _ in range(max_level + 1)]
    for name in order:
        levels[level_of[name]].append(name)
    return [lvl for lvl in levels if lvl]


def to_mermaid(
    graph: dict[str, AnalyzerNode],
    highlight: set[str] | None = None,
) -> str:
    """Render dependency graph as Mermaid flowchart (BT = bottom-to-top)."""
    lines = ["flowchart BT"]
    highlight = highlight or set()
    for name, node in sorted(graph.items()):
        if name in highlight:
            pass
        for req in node.requires:
            if req in graph:
                lines.append(f"    {req} --> {name}")
    if len(lines) == 1:
        lines.append("    empty[No analyzers]")
    return "\n".join(lines)


def to_text_summary(graph: dict[str, AnalyzerNode]) -> str:
    """Human-readable summary with execution levels."""
    levels = compute_execution_levels(graph)
    lines = [f"Analyzers: {len(graph)}", f"Execution levels: {len(levels)}"]
    for i, lvl in enumerate(levels):
        lines.append(f"  L{i}: {', '.join(lvl)}")
    cycles = detect_cycles(graph)
    if cycles:
        lines.append(f"Cycles detected: {cycles}")
    return "\n".join(lines)


def to_json(graph: dict[str, AnalyzerNode]) -> str:
    payload: dict[str, Any] = {
        name: {"requires": node.requires, "module": node.module} for name, node in graph.items()
    }
    payload["_execution_levels"] = compute_execution_levels(graph)
    return json.dumps(payload, indent=2, ensure_ascii=False)
