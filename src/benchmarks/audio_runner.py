"""Execute audio benchmark scenarios against real or dry-run sample sets."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from .audio_catalog import load_catalog
from .audio_models import (
    AudioCompareReport,
    AudioRunReport,
    CompareFileResult,
    FileResult,
    ScenarioId,
)
from .audio_scenarios import resolve_samples, scenario_requires_ml

logger = logging.getLogger(__name__)

# Deepgram pay-as-you-go placeholder (USD/min); update when contract rates change.
DEEPGRAM_USD_PER_MINUTE = 0.0043


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


def _transcript_text(transcript: object) -> str:
    segments = getattr(transcript, "segments", None) or []
    if segments:
        return " ".join((getattr(seg, "text", "") or "").strip() for seg in segments).strip()
    return getattr(transcript, "text", "") or ""


def _reference_transcript(sample: object) -> str | None:
    meta = getattr(sample, "metadata", None)
    if meta is None:
        return None
    statement = getattr(meta, "statement_text", None)
    if statement:
        return str(statement)
    extra = getattr(meta, "extra", None) or {}
    for key in ("reference_transcript", "transcript", "reference"):
        value = extra.get(key)
        if value:
            return str(value)
    return None


def _normalize_words(text: str) -> list[str]:
    return [part for part in text.strip().lower().split() if part]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_words(reference)
    hyp_words = _normalize_words(hypothesis)
    if not ref_words:
        return 0.0
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )
    return round(dist[rows - 1][cols - 1] / len(ref_words), 4)


def _estimate_cloud_cost_usd(duration_s: float | None, provider: str) -> float | None:
    if provider != "cloud" or duration_s is None:
        return None
    return round((duration_s / 60.0) * DEEPGRAM_USD_PER_MINUTE, 6)


def _require_cuda_if_requested(device: str) -> None:
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")


def _pipeline_transcript_text(report: object) -> str:
    segments = getattr(report, "segments", None) or []
    texts: list[str] = []
    for seg in segments:
        if isinstance(seg, dict):
            texts.append(str(seg.get("text") or "").strip())
        else:
            texts.append(str(getattr(seg, "text", "") or "").strip())
    return " ".join(texts).strip()


def _raise_if_pipeline_swallowed_cuda_oom(report: object) -> None:
    from ..transcription.oom_fallback import is_cuda_oom_error

    diarization = getattr(report, "diarization", None) or {}
    if diarization.get("backend") != "failed":
        return
    error = str(diarization.get("error") or "")
    if error and is_cuda_oom_error(RuntimeError(error)):
        raise RuntimeError(error)


def _expected_contains(sample: object) -> list[str]:
    meta = getattr(sample, "metadata", None)
    if meta is None:
        return []
    phrases = getattr(meta, "expected_transcript_contains", None) or []
    if not phrases:
        extra = getattr(meta, "extra", None) or {}
        phrases = extra.get("expected_transcript_contains") or []
    if isinstance(phrases, str):
        return [phrases] if phrases.strip() else []
    return [str(item) for item in phrases if str(item).strip()]


def _normalize_for_contains(text: str) -> str:
    return " ".join(text.casefold().split())


def _missing_expected_phrases(transcript: str, phrases: list[str]) -> list[str]:
    haystack = _normalize_for_contains(transcript)
    return [phrase for phrase in phrases if _normalize_for_contains(phrase) not in haystack]


def _assert_transcript_contract(
    transcript: str | None,
    sample: object,
    *,
    diarization: dict[str, Any] | None = None,
    allow_empty: bool = False,
) -> None:
    if diarization and diarization.get("backend") == "failed":
        raise RuntimeError(f"ASR/diarization failed: {diarization.get('error') or 'unknown'}")
    text = (transcript or "").strip()
    if not text:
        if allow_empty:
            return
        raise RuntimeError("empty transcript")
    missing = _missing_expected_phrases(text, _expected_contains(sample))
    if missing:
        raise RuntimeError(f"missing expected phrases: {missing}")


def _sample_allows_empty(sample: object) -> bool:
    meta = getattr(sample, "metadata", None)
    return bool(getattr(meta, "skip_ml", False))


def _run_asr_on_sample(
    sample_path: str,
    *,
    backend: str,
    device: str,
    language: str,
    provider: str = "local",
    model_name: str = "kb-whisper-large",
    oom_fallback: bool = True,
) -> tuple[str, float, str, bool]:
    from ..transcription.oom_fallback import transcribe_with_oom_fallback
    from ..transcription.router import AsrRouter

    _require_cuda_if_requested(device)

    start = time.time()
    router = AsrRouter()

    def _call(model: str):
        return router.transcribe(
            sample_path,
            provider=provider,
            backend=backend,
            device=device,
            language=language,
            model_name=model,
        )

    result = transcribe_with_oom_fallback(
        primary_model=model_name,
        fallback_model="kb-whisper-medium",
        allow_fallback=oom_fallback,
        transcribe_fn=_call,
    )
    elapsed = time.time() - start
    return _transcript_text(result.value), elapsed, result.model_used, result.fell_back


def _run_pipeline_on_sample(
    sample_path: str,
    *,
    backend: str,
    device: str,
    language: str,
    model_name: str = "kb-whisper-large",
    oom_fallback: bool = True,
    run_diarization: bool = False,
    profile: str = "callcenter",
    preprocess_mode: str = "callcenter",
    allow_empty: bool = False,
) -> tuple[bool, str | None, str, bool, dict[str, Any]]:
    from ..pipeline import CallAnalysisPipeline
    from ..transcription.oom_fallback import transcribe_with_oom_fallback

    _require_cuda_if_requested(device)

    def _analyze(model: str):
        pipeline = CallAnalysisPipeline(
            device=device, asr_backend=backend, asr_model=model, profile=profile
        )
        report = pipeline.analyze_audio(
            audio_path=sample_path,
            language=language,
            run_diarization=run_diarization,
            preprocess_mode=preprocess_mode,
            strict_asr=True,
        )
        _raise_if_pipeline_swallowed_cuda_oom(report)
        diarization = getattr(report, "diarization", None) or {}
        if diarization.get("backend") == "failed":
            raise RuntimeError(f"ASR/diarization failed: {diarization.get('error') or 'unknown'}")
        return report

    result = transcribe_with_oom_fallback(
        primary_model=model_name,
        fallback_model="kb-whisper-medium",
        allow_fallback=oom_fallback,
        transcribe_fn=_analyze,
    )
    text = _pipeline_transcript_text(result.value)
    diarization = getattr(result.value, "diarization", None) or {}
    if not text and not allow_empty:
        raise RuntimeError("empty transcript")
    return True, text or None, result.model_used, result.fell_back, diarization


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
    provider: str = "local",
    dry_run: bool = False,
    model_name: str = "kb-whisper-large",
    oom_fallback: bool = True,
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
    oom_fallbacks = 0
    sentiment_pairs: list[tuple[str | None, str | None]] = []

    for sample in samples:
        pack = catalog.active_packs().get(sample.pack_id)
        lang = language or sample.language or (pack.default_asr_language if pack else "sv")
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
                transcript, elapsed, model_used, fell_back = _run_asr_on_sample(
                    sample.path,
                    backend=backend,
                    device=device,
                    language=lang,
                    provider=provider,
                    model_name=model_name,
                    oom_fallback=oom_fallback,
                )
                result.metadata["model_used"] = model_used
                if fell_back:
                    result.metadata["oom_fell_back"] = True
                    oom_fallbacks += 1
                result.latency_s = round(elapsed, 3)
                result.transcript_preview = _preview_text(transcript)
                _assert_transcript_contract(
                    transcript, sample, allow_empty=_sample_allows_empty(sample)
                )
                result.ok = True
                asr_ok += 1
                if scenario in {"sentiment_chain", "language_sanity"}:
                    pred = _run_sentiment_on_text(transcript, device=device)
                    result.sentiment_pred = pred
                    sentiment_pairs.append((sample.expected_sentiment, pred))

            elif scenario == "pipeline":
                ok, pipeline_transcript, model_used, fell_back, diarization = (
                    _run_pipeline_on_sample(
                        sample.path,
                        backend=backend,
                        device=device,
                        language=lang,
                        model_name=model_name,
                        oom_fallback=oom_fallback,
                        allow_empty=_sample_allows_empty(sample),
                    )
                )
                result.metadata["model_used"] = model_used
                if fell_back:
                    result.metadata["oom_fell_back"] = True
                    oom_fallbacks += 1
                result.pipeline_ok = ok
                result.transcript_preview = _preview_text(pipeline_transcript or "")
                _assert_transcript_contract(
                    pipeline_transcript,
                    sample,
                    diarization=diarization,
                    allow_empty=_sample_allows_empty(sample),
                )
                result.ok = True
                pipeline_ok_count += 1
                pred = _run_sentiment_on_text(pipeline_transcript or "", device=device)
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
        if oom_fallbacks:
            summary["oom_fallbacks"] = oom_fallbacks
    if scenario == "pipeline":
        summary["pipeline_success_rate"] = (
            round(pipeline_ok_count / len(samples), 4) if samples else 0.0
        )
        if oom_fallbacks:
            summary["oom_fallbacks"] = oom_fallbacks

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


def run_compare(
    *,
    providers: list[str],
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
) -> AudioCompareReport:
    from ..transcription.router import AsrRouter, resolve_asr_provider

    resolved_providers = [resolve_asr_provider(provider) for provider in providers]
    catalog = load_catalog(audio_root)
    samples = resolve_samples(
        catalog,
        "smoke",
        pack_ids=pack_ids,
        tags=tags,
        emotions=emotions,
        actors=actors,
        limit=limit,
        subset=subset,
    )
    active_pack_ids = sorted({s.pack_id for s in samples})
    start = time.time()
    results: list[CompareFileResult] = []
    errors: list[str] = []

    if not samples:
        errors.append("No audio samples matched the selection.")
        return AudioCompareReport(
            timestamp=datetime.now(UTC).isoformat(),
            providers=resolved_providers,
            packs=active_pack_ids,
            n_files=0,
            n_runs=0,
            duration_s=0.0,
            dry_run=dry_run,
            device=device,
            backend=backend,
            results=[],
            summary={"error": "no_samples"},
            errors=errors,
        )

    if dry_run:
        for provider in resolved_providers:
            for sample in samples:
                results.append(
                    CompareFileResult(
                        provider=provider,
                        path=sample.path,
                        relative_path=sample.relative_path,
                        pack_id=sample.pack_id,
                        ok=True,
                    )
                )
        duration = time.time() - start
        return AudioCompareReport(
            timestamp=datetime.now(UTC).isoformat(),
            providers=resolved_providers,
            packs=active_pack_ids,
            n_files=len(samples),
            n_runs=len(results),
            duration_s=round(duration, 3),
            dry_run=True,
            device=device,
            backend=backend,
            results=results,
            summary={
                "dry_run": True,
                "providers": resolved_providers,
                "selected_files": len(samples),
                "planned_runs": len(results),
            },
            errors=errors,
        )

    router = AsrRouter()
    for provider in resolved_providers:
        for sample in samples:
            pack = catalog.active_packs().get(sample.pack_id)
            lang = language or (pack.default_asr_language if pack else sample.language)
            row = CompareFileResult(
                provider=provider,
                path=sample.path,
                relative_path=sample.relative_path,
                pack_id=sample.pack_id,
            )
            try:
                wall_start = time.time()
                transcript = router.transcribe(
                    sample.path,
                    provider=provider,
                    backend=backend,
                    device=device,
                    language=lang,
                )
                latency = time.time() - wall_start
                text = _transcript_text(transcript)
                reference = _reference_transcript(sample)
                row.latency_s = round(latency, 3)
                row.n_segments = len(getattr(transcript, "segments", None) or [])
                if reference and text:
                    row.wer = _word_error_rate(reference, text)
                row.estimated_cost_usd = _estimate_cloud_cost_usd(
                    getattr(transcript, "duration", None),
                    provider,
                )
                row.ok = bool(text.strip())
            except Exception as exc:
                row.ok = False
                row.error = str(exc)
                errors.append(f"{provider}:{sample.relative_path}: {exc}")
                logger.exception(
                    "Audio compare failed for %s via %s", sample.relative_path, provider
                )
            results.append(row)

    duration = time.time() - start
    by_provider: dict[str, dict[str, Any]] = {}
    for provider in resolved_providers:
        provider_rows = [row for row in results if row.provider == provider]
        successes = [row for row in provider_rows if row.ok]
        wers = [row.wer for row in successes if row.wer is not None]
        latencies = [row.latency_s for row in successes if row.latency_s is not None]
        costs = [row.estimated_cost_usd for row in successes if row.estimated_cost_usd is not None]
        by_provider[provider] = {
            "n_success": len(successes),
            "n_failed": len(provider_rows) - len(successes),
            "mean_wer": round(sum(wers) / len(wers), 4) if wers else None,
            "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "estimated_cost_usd": round(sum(costs), 6) if costs else None,
        }

    return AudioCompareReport(
        timestamp=datetime.now(UTC).isoformat(),
        providers=resolved_providers,
        packs=active_pack_ids,
        n_files=len(samples),
        n_runs=len(results),
        duration_s=round(duration, 3),
        dry_run=False,
        device=device,
        backend=backend,
        results=results,
        summary={"by_provider": by_provider},
        errors=errors,
    )
