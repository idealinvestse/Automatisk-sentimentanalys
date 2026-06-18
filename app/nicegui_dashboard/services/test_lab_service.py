"""Test lab helpers: audio catalog, benchmarks, preflight, and API path mapping."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.benchmarks.audio_catalog import default_audio_root, load_catalog
from src.benchmarks.audio_models import SampleFilter, ScenarioId, ValidationReport
from src.benchmarks.audio_runner import (
    _run_asr_on_sample,
    _run_pipeline_on_sample,
    _run_sentiment_on_text,
    run_scenario,
)
from src.install.preflight import PreflightReport, run_preflight
from src.install.user_config import load_user_config


def repo_root() -> Path:
    """Project root (parent of samples/audio)."""
    return default_audio_root().parent.parent.resolve()


def examples_txt_path() -> Path:
    return repo_root() / "samples" / "examples.txt"


def reports_dir() -> Path:
    return repo_root() / "reports"


def load_examples_txt() -> list[str]:
    path = examples_txt_path()
    if not path.is_file():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def list_pack_options() -> list[dict[str, str]]:
    catalog = load_catalog()
    active = catalog.active_packs()
    return [
        {"label": pack.label, "value": pack_id}
        for pack_id, pack in sorted(active.items(), key=lambda x: x[1].label)
    ]


def list_emotion_options() -> list[str]:
    catalog = load_catalog()
    emotions: set[str] = set()
    for sample in catalog.discover(SampleFilter(limit=5000)):
        if sample.metadata.emotion:
            emotions.add(sample.metadata.emotion)
    return sorted(emotions)


def list_sample_rows(
    *,
    pack_id: str | None = None,
    emotions: list[str] | None = None,
    actors: list[str] | None = None,
    tags: list[str] | None = None,
    search: str | None = None,
    limit: int | None = 200,
) -> list[dict[str, Any]]:
    catalog = load_catalog()
    samples = catalog.discover(
        SampleFilter(
            pack_ids=[pack_id] if pack_id else None,
            emotions=emotions or None,
            actors=actors or None,
            tags=tags or None,
            limit=limit,
        )
    )
    query = (search or "").strip().lower()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if query and query not in sample.relative_path.lower():
            continue
        rows.append(
            {
                "id": sample.relative_path,
                "pack": sample.pack_id,
                "path": sample.relative_path,
                "emotion": sample.metadata.emotion or "—",
                "expected": sample.expected_sentiment or "—",
                "statement": (sample.metadata.statement_text or "—")[:60],
                "actor": sample.metadata.actor or "—",
                "abs_path": sample.path,
                "language": sample.language,
            }
        )
    return rows


def validate_catalog() -> ValidationReport:
    return load_catalog().validate()


def default_run_settings() -> dict[str, Any]:
    cfg = load_user_config(repo_root())
    return {
        "backend": cfg.asr.backend,
        "model": cfg.asr.model,
        "device": cfg.device,
        "language": "en",
    }


def resolve_api_audio_path(
    local_path: str,
    *,
    media_root: str | None = None,
) -> tuple[str, str | None]:
    """Map a local absolute path to a server-relative path under API_MEDIA_ROOT."""
    resolved = Path(local_path).resolve()
    root_str = media_root if media_root is not None else os.environ.get("API_MEDIA_ROOT")
    if root_str:
        root = Path(root_str).resolve()
        try:
            rel = resolved.relative_to(root)
            return str(rel).replace("\\", "/"), None
        except ValueError:
            return (
                str(resolved),
                f"Sökvägen ligger utanför API_MEDIA_ROOT ({root}). "
                f"Sätt API_MEDIA_ROOT till projektroten.",
            )
    try:
        rel = resolved.relative_to(repo_root())
        return str(rel).replace("\\", "/"), (
            "API_MEDIA_ROOT ej satt — använder relativ sökväg från projektrot. "
            "Sätt API_MEDIA_ROOT till projektroten om API nekar åtkomst."
        )
    except ValueError:
        return str(resolved), "Filen ligger utanför projektroten."


def run_single_asr(
    path: str,
    *,
    backend: str = "faster",
    device: str = "cpu",
    language: str = "en",
) -> dict[str, Any]:
    text, elapsed = _run_asr_on_sample(path, backend=backend, device=device, language=language)
    return {
        "ok": bool(text.strip()),
        "transcript": text,
        "latency_s": round(elapsed, 3),
    }


def run_single_pipeline(
    path: str,
    *,
    backend: str = "faster",
    device: str = "cpu",
    language: str = "en",
) -> dict[str, Any]:
    ok, text = _run_pipeline_on_sample(path, backend=backend, device=device, language=language)
    sentiment = _run_sentiment_on_text(text or "", device=device) if ok and text else None
    return {
        "ok": ok,
        "transcript": text,
        "sentiment": sentiment,
    }


def run_single_sentiment_chain(
    path: str,
    *,
    backend: str = "faster",
    device: str = "cpu",
    language: str = "en",
    expected_sentiment: str | None = None,
) -> dict[str, Any]:
    text, elapsed = _run_asr_on_sample(path, backend=backend, device=device, language=language)
    pred = _run_sentiment_on_text(text, device=device) if text.strip() else None
    return {
        "ok": bool(text.strip()),
        "transcript": text,
        "latency_s": round(elapsed, 3),
        "sentiment_pred": pred,
        "expected_sentiment": expected_sentiment,
        "match": pred == expected_sentiment if expected_sentiment and pred else None,
    }


def run_scenario_ui(
    scenario: ScenarioId,
    *,
    pack_ids: list[str] | None = None,
    emotions: list[str] | None = None,
    actors: list[str] | None = None,
    limit: int | None = None,
    device: str = "cpu",
    backend: str = "faster",
    language: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    report = run_scenario(
        scenario,
        pack_ids=pack_ids,
        emotions=emotions,
        actors=actors,
        limit=limit,
        device=device,
        backend=backend,
        language=language,
        dry_run=dry_run,
    )
    return report.model_dump()


def run_doctor_check(*, require_openrouter: bool = False) -> PreflightReport:
    cfg = load_user_config(repo_root())
    return run_preflight(cfg, require_openrouter=require_openrouter)


def list_audio_reports(*, limit: int = 20) -> list[dict[str, Any]]:
    directory = reports_dir()
    if not directory.is_dir():
        return []
    files = sorted(
        directory.glob("audio_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    rows: list[dict[str, Any]] = []
    for path in files:
        summary: dict[str, Any] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = data.get("summary") or {}
            scenario = data.get("scenario", "?")
            n_files = data.get("n_files", 0)
        except Exception:
            scenario = "?"
            n_files = 0
        rows.append(
            {
                "id": path.name,
                "file": path.name,
                "scenario": scenario,
                "files": n_files,
                "accuracy": summary.get("sentiment_accuracy", "—"),
                "success": summary.get("n_success", "—"),
                "path": str(path),
            }
        )
    return rows


def load_report_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))