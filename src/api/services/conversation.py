"""Conversation analysis service — light path (default) or full pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...caching import AggregateCache
from ...core.serialization import map_results_to_segment_dicts, texts_from_segments, utc_now_iso
from ...customers import CustomerContext
from ...pipeline import CallAnalysisPipeline
from ...sentiment import analyze_smart
from ..call_persistence import (
    fail_reason_from_exc,
    persist_call_artifact,
    persist_intake_file,
    report_as_dict,
)
from ..call_store import CallStore
from ..helpers import (
    apply_customer_policy,
    apply_llm_ceiling,
    asr_kwargs_from,
    require_usable_transcript,
    transcribe_helper,
)
from ..schemas import (
    AnalyzeConversationRequest,
    AnalyzeConversationResponse,
    SegmentSentiment,
)

logger = logging.getLogger(__name__)


def _sentiment_profile(req: Any, customer: CustomerContext | None = None) -> str:
    policy = apply_customer_policy(
        customer,
        requested_profile=getattr(req, "sentiment_profile", None) or "callcenter",
    )
    return policy.analyzer_profile


def _build_segment_sentiments(
    texts: list[str],
    results: list[Any],
    segments: list[dict[str, Any]],
) -> list[SegmentSentiment]:
    dicts = map_results_to_segment_dicts(texts, results, segments)
    return [SegmentSentiment(**d) for d in dicts]


def _light_analyze(
    req: AnalyzeConversationRequest,
    customer: CustomerContext | None,
    call_store: CallStore | None = None,
) -> AnalyzeConversationResponse:
    tr = transcribe_helper(
        **asr_kwargs_from(req, audio_path=req.audio_path, customer=customer)
    )
    require_usable_transcript(tr)
    segments = tr.get("segments", []) or []
    tr_texts = texts_from_segments(segments)
    results, meta = analyze_smart(
        tr_texts,
        profile=_sentiment_profile(req, customer),
        model_name=req.sentiment_model,
        device=req.device,
        batch_size=16,
        normalize=True,
        return_all_scores=req.return_all_scores,
        max_length=None,
        clean=True,
        lexicon_file=req.lexicon_file,
        lexicon_weight=req.lexicon_weight,
    )
    seg_out = _build_segment_sentiments(tr_texts, results, segments)
    if call_store is not None:
        persisted = persist_call_artifact(
            call_store,
            status="completed",
            transcript=tr,
            report={"mode": "light"},
            customer=customer,
            original_filename=req.original_filename,
            audio_path=req.audio_path,
            route="analyze_conversation",
        )
        meta = dict(meta or {})
        meta["call_id"] = persisted.get("id")
        meta["persisted"] = True
    return AnalyzeConversationResponse(
        transcript=tr,
        segment_sentiments=seg_out,
        meta=meta,
        timestamp=utc_now_iso(),
        pipeline_results=None,
    )


def _full_pipeline_analyze(
    req: AnalyzeConversationRequest,
    cache: AggregateCache | None,
    customer: CustomerContext | None,
    call_store: CallStore | None = None,
) -> AnalyzeConversationResponse:
    policy = apply_customer_policy(
        customer,
        requested_asr_provider=req.provider,
        requested_cloud_fallback=req.cloud_fallback_local,
        requested_profile=req.sentiment_profile,
    )

    def _persist_transcript(transcript: Any) -> None:
        if call_store is None:
            return
        persist_call_artifact(
            call_store,
            status="transcribed",
            transcript=transcript.to_dict() if hasattr(transcript, "to_dict") else dict(transcript),
            customer=customer,
            original_filename=req.original_filename,
            audio_path=req.audio_path,
            route="analyze_conversation",
        )

    pipe = CallAnalysisPipeline(
        sentiment_model=req.sentiment_model or "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=req.device,
        profile=policy.analyzer_profile,
        asr_backend=req.backend,
        asr_model=req.model,
        asr_provider=policy.asr_provider,
        cloud_fallback_local=policy.cloud_fallback_local,
        cache=cache,
        qa_scorecard=policy.qa_scorecard,
        customer_id=policy.customer_id,
        config_fingerprint=policy.config_fingerprint,
        transcript_hook=_persist_transcript if call_store is not None else None,
    )
    apply_llm_ceiling(pipe, policy)
    report = pipe.analyze_audio(
        audio_path=req.audio_path,
        num_speakers=req.num_speakers,
        language=req.language,
        run_diarization=req.diarize,
        hotwords=req.hotwords,
        initial_prompt=req.initial_prompt,
        strict_asr=True,
    )
    segments = report.segments or []
    tr_texts = texts_from_segments(segments)
    transcript = {
        "model": req.model,
        "backend": req.backend,
        "language": req.language,
        "segments": segments,
        "diarization": report.diarization,
    }
    seg_out = _build_segment_sentiments(tr_texts, report.sentiment_results, segments)
    meta: dict[str, Any] = {
        "profile": pipe.profile,
        "model": req.sentiment_model or pipe.sentiment_model,
        "pipeline": True,
        "processing_time_s": report.processing_time_s,
        "qa_scorecard": policy.qa_scorecard,
    }
    if call_store is not None:
        persisted = persist_call_artifact(
            call_store,
            status="completed",
            transcript=transcript,
            report=report_as_dict(report),
            customer=customer,
            original_filename=req.original_filename,
            audio_path=req.audio_path,
            route="analyze_conversation",
        )
        meta["call_id"] = persisted.get("id")
        meta["persisted"] = True
    return AnalyzeConversationResponse(
        transcript=transcript,
        segment_sentiments=seg_out,
        meta=meta,
        timestamp=utc_now_iso(),
        pipeline_results=report.results,
    )


def _persist_run_failure(
    store: CallStore | None,
    customer: CustomerContext | None,
    req: AnalyzeConversationRequest,
    exc: BaseException,
    *,
    route: str,
) -> None:
    if store is None:
        return
    try:
        persist_call_artifact(
            store,
            status="failed",
            customer=customer,
            original_filename=req.original_filename,
            audio_path=req.audio_path,
            route=route,
            fail_reason=fail_reason_from_exc(exc),
        )
    except Exception:
        logger.exception("Failed to persist conversation failure provenance")


def run_analyze_conversation(
    req: AnalyzeConversationRequest,
    *,
    cache: AggregateCache | None = None,
    customer: CustomerContext | None = None,
    call_store: CallStore | None = None,
) -> AnalyzeConversationResponse:
    route = "analyze_conversation"
    try:
        if req.use_full_pipeline:
            return _full_pipeline_analyze(req, cache, customer, call_store)
        return _light_analyze(req, customer, call_store)
    except Exception as exc:
        _persist_run_failure(call_store, customer, req, exc, route=route)
        raise


def _scan_to_conversation_request(req: Any, audio_path: str) -> AnalyzeConversationRequest:
    return AnalyzeConversationRequest(
        audio_path=audio_path,
        model=req.model,
        backend=req.backend,
        device=req.device,
        language=req.language,
        beam_size=req.beam_size,
        vad=req.vad,
        word_timestamps=req.word_timestamps,
        chunk_length_s=req.chunk_length_s,
        revision=req.revision,
        diarize=req.diarize,
        num_speakers=req.num_speakers,
        hotwords=req.hotwords,
        initial_prompt=req.initial_prompt,
        use_full_pipeline=req.use_full_pipeline,
        sentiment_profile=req.sentiment_profile,
        sentiment_model=req.sentiment_model,
        lexicon_file=req.lexicon_file,
        lexicon_weight=req.lexicon_weight,
        return_all_scores=getattr(req, "return_all_scores", True),
        provider=getattr(req, "provider", "local"),
        cloud_fallback_local=getattr(req, "cloud_fallback_local", False),
        original_filename=getattr(req, "original_filename", None) or Path(audio_path).name,
    )


def run_batch_analyze_file(
    req: Any,
    audio_path: str,
    *,
    cache: AggregateCache | None = None,
    customer: CustomerContext | None = None,
    call_store: CallStore | None = None,
) -> tuple[dict[str, Any], list[SegmentSentiment], dict[str, Any], dict[str, Any] | None]:
    """Single-file worker for batch/scan conversation analysis."""
    if getattr(req, "use_full_pipeline", False):
        conv_req = _scan_to_conversation_request(req, audio_path)
        resp = run_analyze_conversation(
            conv_req, cache=cache, customer=customer, call_store=call_store
        )
        return resp.transcript, resp.segment_sentiments, resp.meta, resp.pipeline_results

    try:
        tr = transcribe_helper(
            **asr_kwargs_from(req, audio_path=audio_path, customer=customer)
        )
        require_usable_transcript(tr)
        segments = tr.get("segments", []) or []
        tr_texts = texts_from_segments(segments)
        results, meta = analyze_smart(
            tr_texts,
            profile=_sentiment_profile(req, customer),
            model_name=req.sentiment_model,
            device=req.device,
            batch_size=getattr(req, "sentiment_batch_size", 16),
            normalize=True,
            return_all_scores=True,
            max_length=None,
            clean=True,
            lexicon_file=req.lexicon_file,
            lexicon_weight=req.lexicon_weight,
        )
        seg_out = _build_segment_sentiments(tr_texts, results, segments)
        persisted = persist_intake_file(
            call_store,
            audio_path=audio_path,
            route="batch_analyze_conversation",
            status="completed",
            transcript=tr,
            report={"mode": "light"},
            customer=customer,
        )
        if persisted is not None:
            meta = dict(meta or {})
            meta["call_id"] = persisted.get("id")
            meta["persisted"] = True
        return tr, seg_out, meta, None
    except Exception as exc:
        persist_intake_file(
            call_store,
            audio_path=audio_path,
            route="batch_analyze_conversation",
            status="failed",
            customer=customer,
            error=exc,
        )
        raise
