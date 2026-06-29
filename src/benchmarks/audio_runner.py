"""Execute audio benchmark scenarios against real or dry-run sample sets."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from .audio_catalog import load_catalog
from .audio_models import AudioRunReport, FileResult, ScenarioId
from .audio_scenarios import resolve_samples, scenario_requires_ml

logger = logging.getLogger(__name__)


def _preview_text(text: str, max_len: int = 120) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _aggregate_sentiment(scores: list[dict[str, Any]]) -> str | None:
    if not scores:
        return None
    first = scores[0]
    if isinstance(first, dict) and "label" in first:
        return str(first["label"])
    if isinstance(first, list) and first:
        best = max(first, key=lambda x: float(x.get("score", 0)))
        return str(best.get("label"))
    return None


def _run_asr_on_sample(
    sample_path: str,
    *,
    backend: str,
    device: str,
    language: str,
) -> tuple[str, float]:
    from ..transcription import get_transcriber

    transcriber = get_transcriber(backend=backend, device=device)
    start = time.time()
    result = transcriber.transcribe(audio_path=sample_path, language=language)
    elapsed = time.time() - start
    segments = getattr(result, "segments", None) or []
    if segments:
        text = " ".join((getattr(seg, "text", "") or "").strip() for seg in segments).strip()
    else:
        text = getattr(result, "text", "") or ""
    return text, elapsed


def _run_pipeline_on_sample(
    sample_path: str,
    *,
    backend: str,
    device: str,
    language: str,
) -> tuple[bool, str | None]:
    from ..pipeline import CallAnalysisPipeline

    pipeline = CallAnalysisPipeline(device=device, asr_backend=backend)
    report = pipeline.analyze_audio(
        audio_path=sample_path, language=language, run_diarization=False
    )
    segments = report.segments or []
    texts: list[str] = []
    for seg in segments:
        if isinstance(seg, dict):
            texts.append(str(seg.get("text") or "").strip())
        else:
            texts.append(str(getattr(seg, "text", "") or "").strip())
    text = " ".join(texts).strip()
    return True, text or None


def _run_sentiment_on_text(text: str, *, device: str) -> str | None:
    if not text.strip():
        return None
    from ..sentiment import analyze_smart

    results, _meta = analyze_smart([text], device=device)
    return _aggregate_sentiment(results)


def run_scenario(
    scenario: ScenarioId,
    *,
    audio_root: str | None = None,
    pack_ids: list[str] | None = None,
    tags: list[str] | None = None,
    emotions: list[str] | None = None,
    actors: list[str] | None = None,
    limit: int | None = None,
    subset: str | None = None,
    device: str = "cpu",
    backend: str = "faster",
    language: str | None = None,
    dry_run: bool = False,
) -> AudioRunReport:
    catalog = load_catalog(audio_root)
    samples = resolve_samples(
        catalog,
        scenario,
        pack_ids=pack_ids,
        tags=tags,
        emotions=emotions,
        actors=actors,
        limit=limit,
        subset=subset,
    )
    active_pack_ids = sorted({s.pack_id for s in samples})
    start = time.time()
    file_results: list[FileResult] = []
    errors: list[str] = []

    if scenario == "catalog":
        for sample in samples:
            file_results.append(
                FileResult(
                    path=sample.path,
                    relative_path=sample.relative_path,
                    pack_id=sample.pack_id,
                    metadata=sample.metadata.model_dump(),
                    ok=True,
                    expected_sentiment=sample.expected_sentiment,
                )
            )
        duration = time.time() - start
        return AudioRunReport(
            timestamp=datetime.now(UTC).isoformat(),
            scenario=scenario,
            packs=active_pack_ids,
            n_files=len(samples),
            duration_s=round(duration, 3),
            dry_run=dry_run,
            device=device,
            backend=backend,
            files=file_results,
            summary={
                "catalog_only": True,
                "file_count": len(samples),
            },
            errors=errors,
        )

    if not samples:
        errors.append("No audio samples matched the selection.")
        return AudioRunReport(
            timestamp=datetime.now(UTC).isoformat(),
            scenario=scenario,
            packs=active_pack_ids,
            n_files=0,
            duration_s=0.0,
            dry_run=dry_run,
            device=device,
            backend=backend,
            files=[],
            summary={"error": "no_samples"},
            errors=errors,
        )

    if dry_run:
        for sample in samples:
            pack = catalog.active_packs().get(sample.pack_id)
            lang = language or (pack.default_asr_language if pack else sample.language)
            file_results.append(
                FileResult(
                    path=sample.path,
                    relative_path=sample.relative_path,
                    pack_id=sample.pack_id,
                    metadata=sample.metadata.model_dump(),
                    ok=True,
                    expected_sentiment=sample.expected_sentiment,
                    language_used=lang,
                )
            )
        duration = time.time() - start
        return AudioRunReport(
            timestamp=datetime.now(UTC).isoformat(),
            scenario=scenario,
            packs=active_pack_ids,
            n_files=len(samples),
            duration_s=round(duration, 3),
            dry_run=True,
            device=device,
            backend=backend,
            files=file_results,
            summary={
                "dry_run": True,
                "selected_files": len(samples),
                "n_success": len(samples),
            },
            errors=errors,
        )

    if scenario_requires_ml(scenario):
        try:
            import torch  # noqa: F401
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError as exc:
            errors.append(f"ML dependencies missing for scenario '{scenario}': {exc}")
            return AudioRunReport(
                timestamp=datetime.now(UTC).isoformat(),
                scenario=scenario,
                packs=active_pack_ids,
                n_files=len(samples),
                duration_s=0.0,
                dry_run=False,
                device=device,
                backend=backend,
                files=[],
                summary={"error": "missing_ml_deps"},
                errors=errors,
            )

    asr_ok = 0
    pipeline_ok_count = 0
    sentiment_pairs: list[tuple[str | None, str | None]] = []

    for sample in samples:
        pack = catalog.active_packs().get(sample.pack_id)
        lang = language or (pack.default_asr_language if pack else sample.language)
        result = FileResult(
            path=sample.path,
            relative_path=sample.relative_path,
            pack_id=sample.pack_id,
            metadata=sample.metadata.model_dump(),
            expected_sentiment=sample.expected_sentiment,
            language_used=lang,
        )
        try:
            if scenario in {"smoke", "asr", "sentiment_chain", "language_sanity"}:
                transcript, elapsed = _run_asr_on_sample(
                    sample.path,
                    backend=backend,
                    device=device,
                    language=lang,
                )
                result.latency_s = round(elapsed, 3)
                result.transcript_preview = _preview_text(transcript)
                result.ok = bool(transcript.strip())
                if result.ok:
                    asr_ok += 1
                if scenario in {"sentiment_chain", "language_sanity"}:
                    pred = _run_sentiment_on_text(transcript, device=device)
                    result.sentiment_pred = pred
                    sentiment_pairs.append((sample.expected_sentiment, pred))

            elif scenario == "pipeline":
                ok, transcript = _run_pipeline_on_sample(
                    sample.path,
                    backend=backend,
                    device=device,
                    language=lang,
                )
                result.pipeline_ok = ok
                result.ok = ok
                result.transcript_preview = _preview_text(transcript or "")
                if ok:
                    pipeline_ok_count += 1
                    pred = _run_sentiment_on_text(transcript or "", device=device)
                    result.sentiment_pred = pred
                    sentiment_pairs.append((sample.expected_sentiment, pred))
            else:
                result.ok = True
        except Exception as exc:
            result.ok = False
            result.error = str(exc)
            errors.append(f"{sample.relative_path}: {exc}")
            logger.exception("Audio benchmark failed for %s", sample.relative_path)

        file_results.append(result)

    duration = time.time() - start
    summary: dict[str, Any] = {
        "n_success": sum(1 for f in file_results if f.ok),
        "n_failed": sum(1 for f in file_results if not f.ok),
    }
    if scenario in {"smoke", "asr", "sentiment_chain", "language_sanity"}:
        summary["asr_success_rate"] = round(asr_ok / len(samples), 4) if samples else 0.0
    if scenario == "pipeline":
        summary["pipeline_success_rate"] = (
            round(pipeline_ok_count / len(samples), 4) if samples else 0.0
        )

    comparable = [(exp, pred) for exp, pred in sentiment_pairs if exp and pred]
    if comparable:
        correct = sum(1 for exp, pred in comparable if exp == pred)
        summary["sentiment_accuracy"] = round(correct / len(comparable), 4)
        summary["sentiment_compared"] = len(comparable)

    return AudioRunReport(
        timestamp=datetime.now(UTC).isoformat(),
        scenario=scenario,
        packs=active_pack_ids,
        n_files=len(samples),
        duration_s=round(duration, 3),
        dry_run=False,
        device=device,
        backend=backend,
        files=file_results,
        summary=summary,
        errors=errors,
    )
