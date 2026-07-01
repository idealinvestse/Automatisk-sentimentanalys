"""Registry for managing and executing text analyzers with topological dependency resolution."""

from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import yaml

from ..core.errors import AnalysisError
from ..core.metrics import record_analyzer_duration
from ..core.models import AnalysisContext
from ..core.status import get_status_reporter
from .base import Analyzer
from .graph import AnalyzerNode, build_dependency_graph, compute_execution_levels, topological_sort
from .schemas import ValidationMode, validate_analyzer_result

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type[Analyzer])

_ANALYZER_REGISTRY: dict[str, type[Analyzer] | Callable[[], Analyzer]] = {}
_LOADED = False
_THREAD_POOL: ThreadPoolExecutor | None = None
_THREAD_POOL_LOCK = threading.Lock()

# Analyzers that perform external I/O (LLM calls) — safe to parallelize within a level.
IO_BOUND_ANALYZERS: frozenset[str] = frozenset({"llm_judge"})

_SKIP_AUTODISCOVER = frozenset(
    {"base", "registry", "graph", "resources", "schemas", "templates", "__init__"}
)


def get_analyzer_registry() -> dict[str, type[Analyzer] | Callable[[], Analyzer]]:
    """Return the live analyzer registry (after ensure_analyzers_loaded)."""
    ensure_analyzers_loaded()
    return _ANALYZER_REGISTRY


def register_analyzer(name: str, *, strict: bool = False) -> Callable[[T], T]:
    """Decorator to register an Analyzer class with a unique name."""

    def decorator(cls: T) -> T:
        if name in _ANALYZER_REGISTRY:
            msg = f"Analyzer '{name}' already registered (overwrite by {cls})"
            if strict:
                raise AnalysisError(msg)
            logger.warning(msg)
        _ANALYZER_REGISTRY[name] = cls
        logger.debug("Registered analyzer: %s", name)
        return cls

    return decorator


def register_analyzer_class(
    name: str,
    cls: type[Analyzer] | Callable[[], Analyzer],
    *,
    strict: bool = False,
) -> None:
    """Register an analyzer class or factory at runtime."""
    if name in _ANALYZER_REGISTRY:
        msg = f"Analyzer '{name}' already registered"
        if strict:
            raise AnalysisError(msg)
        logger.warning(msg)
    _ANALYZER_REGISTRY[name] = cls
    logger.info("Dynamically registered analyzer: %s", name)


def autodiscover(package: str = "src.analysis") -> None:
    """Import all analyzer modules in a package to trigger @register_analyzer decorators."""
    pkg = importlib.import_module(package)
    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        return
    for _finder, mod_name, _ispkg in pkgutil.iter_modules(pkg_path):
        if mod_name.startswith("_") or mod_name in _SKIP_AUTODISCOVER:
            continue
        full_name = f"{package}.{mod_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:
            logger.error("Autodiscover failed for %s: %s", full_name, exc)


def _configs_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "configs"


def _is_allowlisted(module_path: str, prefixes: list[str]) -> bool:
    return any(module_path.startswith(p) for p in prefixes)


def load_analyzers_from_config(
    path: Path | None = None,
    *,
    graceful: bool = True,
) -> None:
    """Load optional plugin analyzers from YAML config with module allowlist."""
    config_path = path or (_configs_dir() / "analyzers.yaml")
    if not config_path.is_file():
        return
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    prefixes = list(data.get("allowlist_prefixes") or ["src.analysis."])
    for entry in data.get("plugins") or []:
        if not entry.get("enabled", False):
            continue
        module_path = str(entry.get("module", ""))
        class_name = str(entry.get("class", ""))
        reg_name = entry.get("name")
        if not module_path or not class_name:
            continue
        if not _is_allowlisted(module_path, prefixes):
            msg = f"Plugin module not allowlisted: {module_path}"
            if graceful:
                logger.error(msg)
                continue
            raise AnalysisError(msg)
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            name = reg_name or getattr(cls, "name", None)
            if callable(name):
                name = name()
            if not name:
                name = class_name.lower()
            register_analyzer_class(str(name), cls)
        except Exception as exc:
            msg = f"Failed to load plugin {module_path}.{class_name}: {exc}"
            if graceful:
                logger.error(msg)
            else:
                raise AnalysisError(msg) from exc


def load_entry_point_analyzers(*, graceful: bool = True) -> None:
    """Load analyzers registered via setuptools entry points group ``sentiment_analyzers``."""
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return

    try:
        eps = entry_points(group="sentiment_analyzers")
    except TypeError:
        eps = entry_points().get("sentiment_analyzers", [])

    for ep in eps:
        try:
            loaded = ep.load()
            if isinstance(loaded, type):
                name = getattr(loaded, "name", ep.name)
                if callable(name):
                    inst = loaded()
                    name = inst.name
                register_analyzer_class(str(name), loaded)
            elif callable(loaded):
                register_analyzer_class(ep.name, loaded)
        except Exception as exc:
            msg = f"Entry point '{ep.name}' failed: {exc}"
            if graceful:
                logger.error(msg)
            else:
                raise AnalysisError(msg) from exc


def ensure_analyzers_loaded() -> None:
    """Idempotent: autodiscover built-ins, YAML plugins, and entry points."""
    global _LOADED
    if _LOADED:
        return
    autodiscover()
    load_analyzers_from_config(graceful=True)
    load_entry_point_analyzers(graceful=True)
    _LOADED = True


def _instantiate_one(
    name: str,
    factory: type[Analyzer] | Callable[[], Analyzer],
    kwargs: dict[str, Any],
) -> Analyzer | None:
    try:
        if isinstance(factory, type):
            return factory(**kwargs)
        return factory()
    except TypeError:
        try:
            return factory()  # type: ignore[misc]
        except Exception as e:
            logger.error("Failed to instantiate analyzer '%s': %s", name, e)
            return None
    except Exception as e:
        logger.error("Failed to instantiate analyzer '%s': %s", name, e)
        return None


def get_analyzers_for_run(
    names: set[str],
    analyzer_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Analyzer]:
    """Instantiate only analyzers needed for this run."""
    ensure_analyzers_loaded()
    configs = analyzer_configs or {}
    analyzers: dict[str, Analyzer] = {}
    for name in names:
        factory = _ANALYZER_REGISTRY.get(name)
        if factory is None:
            continue
        inst = _instantiate_one(name, factory, configs.get(name, {}))
        if inst is not None:
            analyzers[name] = inst
    return analyzers


def get_registered_analyzers(
    analyzer_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Analyzer]:
    """Instantiate and return all registered analyzers."""
    ensure_analyzers_loaded()
    return get_analyzers_for_run(set(_ANALYZER_REGISTRY.keys()), analyzer_configs)


def resolve_analyzers_for_profile(
    profile_name: str,
    explicit_selected: list[str] | None = None,
    profile_spec: dict[str, Any] | None = None,
) -> list[str] | None:
    """Resolve analyzer selection: explicit > profile default > all (None)."""
    if explicit_selected is not None:
        return explicit_selected

    from ..profiles import get_analyzer_profile_spec

    spec = profile_spec if profile_spec is not None else get_analyzer_profile_spec(profile_name)
    analyzers_cfg = (spec or {}).get("analyzers") or {}
    default_selected = analyzers_cfg.get("default_selected")
    disabled = set(analyzers_cfg.get("disabled") or [])

    if default_selected is None:
        return None
    if not isinstance(default_selected, list):
        return None
    return [a for a in default_selected if a not in disabled]


def _requires_map() -> dict[str, list[str]]:
    graph = build_dependency_graph(_ANALYZER_REGISTRY)
    return {name: list(node.requires) for name, node in graph.items()}


def _resolve_to_run(
    registered_names: set[str],
    selected: list[str] | None,
) -> set[str]:
    to_run: set[str] = set()
    if selected is None:
        return set(registered_names)

    req_map = _requires_map()

    def resolve_deps(name: str) -> None:
        if name in to_run:
            return
        to_run.add(name)
        for req in req_map.get(name, []):
            if req not in registered_names:
                logger.warning("Required analyzer '%s' for '%s' is not registered!", req, name)
            resolve_deps(req)

    for s in selected:
        if s not in registered_names:
            logger.error("Selected analyzer '%s' is not registered!", s)
            continue
        resolve_deps(s)
    return to_run


def _deps_satisfied(analyzer: Analyzer, ctx: AnalysisContext) -> bool:
    for req in analyzer.requires:
        if req not in ctx.results:
            logger.warning(
                "Skipping analyzer '%s' because dependency '%s' did not produce results.",
                analyzer.name,
                req,
            )
            return False
    return True


def _store_result(
    name: str,
    raw: Any,
    validation_mode: ValidationMode,
) -> Any:
    return validate_analyzer_result(name, raw, mode=validation_mode)


def _analyzer_error_result(name: str, exc: Exception) -> dict[str, Any]:
    return {
        name: {
            "error": str(exc),
            "error_code": "analysis_failed",
            "fallback": True,
        }
    }


def _run_single_sync(
    name: str, analyzer: Analyzer, ctx: AnalysisContext, validation_mode: ValidationMode
) -> None:
    if not _deps_satisfied(analyzer, ctx):
        return
    status = get_status_reporter()
    status.phase("analyzer", name, f"Startar analyzer '{name}'")
    started = time.perf_counter()
    try:
        res = analyzer.analyze(ctx)
        ctx.results[name] = _store_result(name, res, validation_mode)
        duration = time.perf_counter() - started
        logger.info("Analyzer '%s' completed in %.2fs", name, duration)
        status.info("analyzer", name, f"Analyzer '{name}' klar", duration_s=round(duration, 2))
    except Exception as exc:
        duration = time.perf_counter() - started
        logger.error("Analyzer '%s' failed: %s", name, exc, exc_info=True)
        status.error("analyzer", name, f"Analyzer '{name}' misslyckades: {exc}")
        ctx.results.update(_analyzer_error_result(name, exc))
    finally:
        record_analyzer_duration(name, time.perf_counter() - started)


async def _run_single_async(
    name: str,
    analyzer: Analyzer,
    ctx: AnalysisContext,
    validation_mode: ValidationMode,
) -> None:
    if not _deps_satisfied(analyzer, ctx):
        return
    status = get_status_reporter()
    status.phase("analyzer", name, f"Startar analyzer '{name}' (async)")
    started = time.perf_counter()
    try:
        analyze_async = getattr(analyzer, "analyze_async", None)
        if analyze_async is not None and asyncio.iscoroutinefunction(analyze_async):
            res = await analyze_async(ctx)
        elif name in IO_BOUND_ANALYZERS:
            res = await asyncio.to_thread(analyzer.analyze, ctx)
        else:
            loop = asyncio.get_event_loop()
            global _THREAD_POOL, _THREAD_POOL_LOCK
            if _THREAD_POOL is None:
                with _THREAD_POOL_LOCK:
                    if _THREAD_POOL is None:
                        _THREAD_POOL = ThreadPoolExecutor(max_workers=4)
            res = await loop.run_in_executor(_THREAD_POOL, analyzer.analyze, ctx)
        ctx.results[name] = _store_result(name, res, validation_mode)
        duration = time.perf_counter() - started
        logger.info("Analyzer '%s' completed in %.2fs (async)", name, duration)
        status.info("analyzer", name, f"Analyzer '{name}' klar", duration_s=round(duration, 2))
    except Exception as exc:
        logger.error("Analyzer '%s' failed: %s", name, exc, exc_info=True)
        status.error("analyzer", name, f"Analyzer '{name}' misslyckades: {exc}")
        ctx.results.update(_analyzer_error_result(name, exc))
    finally:
        record_analyzer_duration(name, time.perf_counter() - started)


def _build_execution_plan(
    active_analyzers: dict[str, Analyzer],
) -> tuple[list[str], list[list[str]]]:
    graph = build_dependency_graph(_ANALYZER_REGISTRY, selected=set(active_analyzers.keys()))
    # Re-bind requires from live instances
    for name, analyzer in active_analyzers.items():
        if name in graph:
            graph[name] = AnalyzerNode(
                name=name,
                requires=list(analyzer.requires),
                module=graph[name].module,
            )
    order = topological_sort(graph)
    levels = compute_execution_levels(graph)
    return order, levels


def run_analyzers(
    ctx: AnalysisContext,
    selected: list[str] | None = None,
    analyzer_configs: dict[str, dict[str, Any]] | None = None,
    *,
    async_mode: bool = False,
    validation_mode: ValidationMode | None = None,
    skip_llm_superseded: bool = False,
) -> dict[str, Any]:
    """Execute analyzers in topological order, respecting dependencies."""
    ensure_analyzers_loaded()
    effective_validation: ValidationMode = validation_mode or "warn"  # type: ignore[assignment]

    from .deep_path import filter_superseded

    selected = filter_superseded(selected, skip=skip_llm_superseded)
    registered = set(_ANALYZER_REGISTRY.keys())
    to_run = _resolve_to_run(registered, selected)
    active_analyzers = get_analyzers_for_run(to_run, analyzer_configs)

    execution_order, levels = _build_execution_plan(active_analyzers)
    logger.info("Executing analyzers in order: %s", " -> ".join(execution_order))
    status = get_status_reporter()
    status.phase(
        "pipeline",
        "run_analyzers",
        f"Kör {len(execution_order)} analyzers",
        analyzers=execution_order,
    )

    if "spoken_normalizer" in to_run:
        sn = active_analyzers.get("spoken_normalizer")
        if sn is not None and "spoken_normalizer" not in ctx.results:
            try:
                _run_single_sync("spoken_normalizer", sn, ctx, effective_validation)
            except Exception as e:
                logger.error("Analyzer 'spoken_normalizer' failed: %s", e, exc_info=True)
                ctx.results.update(_analyzer_error_result("spoken_normalizer", e))

    if async_mode:
        return asyncio.run(
            _run_analyzers_async_body(active_analyzers, levels, ctx, effective_validation)
        )

    total = len(execution_order)
    for index, name in enumerate(execution_order, start=1):
        if name == "spoken_normalizer" and "spoken_normalizer" in ctx.results:
            continue
        analyzer = active_analyzers.get(name)
        if analyzer is None:
            continue
        status.progress("pipeline", "run_analyzers", index - 1, total, f"Kör {name}")
        logger.debug("Running analyzer: %s", name)
        _run_single_sync(name, analyzer, ctx, effective_validation)

    status.progress("pipeline", "run_analyzers", total, total, "Analyzers klara")
    return ctx.results


async def run_analyzers_async(
    ctx: AnalysisContext,
    selected: list[str] | None = None,
    analyzer_configs: dict[str, dict[str, Any]] | None = None,
    *,
    validation_mode: ValidationMode | None = None,
    skip_llm_superseded: bool = False,
) -> dict[str, Any]:
    """Async execution with parallel levels."""
    ensure_analyzers_loaded()
    effective_validation: ValidationMode = validation_mode or "warn"  # type: ignore[assignment]

    from .deep_path import filter_superseded

    selected = filter_superseded(selected, skip=skip_llm_superseded)
    registered = set(_ANALYZER_REGISTRY.keys())
    to_run = _resolve_to_run(registered, selected)
    active_analyzers = get_analyzers_for_run(to_run, analyzer_configs)
    _, levels = _build_execution_plan(active_analyzers)
    if "spoken_normalizer" in to_run:
        sn = active_analyzers.get("spoken_normalizer")
        if sn is not None and "spoken_normalizer" not in ctx.results:
            try:
                await _run_single_async("spoken_normalizer", sn, ctx, effective_validation)
            except Exception as e:
                logger.error("Analyzer 'spoken_normalizer' failed: %s", e, exc_info=True)
                ctx.results.update(_analyzer_error_result("spoken_normalizer", e))
    return await _run_analyzers_async_body(active_analyzers, levels, ctx, effective_validation)


async def _run_analyzers_async_body(
    active_analyzers: dict[str, Analyzer],
    levels: list[list[str]],
    ctx: AnalysisContext,
    validation_mode: ValidationMode,
) -> dict[str, Any]:
    status = get_status_reporter()
    flat_names = [name for level in levels for name in level]
    total = max(len(flat_names), 1)
    completed = 0
    for level in levels:
        tasks = []
        for name in level:
            if name == "spoken_normalizer" and "spoken_normalizer" in ctx.results:
                continue
            analyzer = active_analyzers.get(name)
            if analyzer is None:
                continue
            logger.debug("Scheduling analyzer (async level): %s", name)

            async def _run(n: str = name, a: Analyzer = analyzer) -> None:
                await _run_single_async(n, a, ctx, validation_mode)

            tasks.append(_run())
        if tasks:
            await asyncio.gather(*tasks)
            completed += len(tasks)
            status.progress("pipeline", "run_analyzers", completed, total, "Async analyzers")
    return ctx.results
