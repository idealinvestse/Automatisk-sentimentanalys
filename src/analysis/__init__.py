"""Analysis module that registers and manages text and conversation analyzers."""

from __future__ import annotations

from .base import Analyzer
from .graph import (
    AnalyzerNode,
    build_dependency_graph,
    compute_execution_levels,
    to_json,
    to_mermaid,
    to_text_summary,
)
from .registry import (
    IO_BOUND_ANALYZERS,
    autodiscover,
    ensure_analyzers_loaded,
    get_analyzer_registry,
    get_analyzers_for_run,
    get_registered_analyzers,
    load_analyzers_from_config,
    load_entry_point_analyzers,
    register_analyzer,
    register_analyzer_class,
    resolve_analyzers_for_profile,
    run_analyzers,
    run_analyzers_async,
)
from .resources import ModelResourcePool, get_pool
from .schemas import (
    AnalyzerResultRegistry,
    get_typed_result,
    get_validation_mode,
    validate_analyzer_result,
)

# Trigger autodiscovery of all @register_analyzer modules on package import.
ensure_analyzers_loaded()

__all__ = [
    "Analyzer",
    "AnalyzerNode",
    "AnalyzerResultRegistry",
    "IO_BOUND_ANALYZERS",
    "ModelResourcePool",
    "autodiscover",
    "build_dependency_graph",
    "compute_execution_levels",
    "ensure_analyzers_loaded",
    "get_analyzer_registry",
    "get_analyzers_for_run",
    "get_pool",
    "get_registered_analyzers",
    "get_typed_result",
    "get_validation_mode",
    "load_analyzers_from_config",
    "load_entry_point_analyzers",
    "register_analyzer",
    "register_analyzer_class",
    "resolve_analyzers_for_profile",
    "run_analyzers",
    "run_analyzers_async",
    "to_json",
    "to_mermaid",
    "to_text_summary",
    "validate_analyzer_result",
]