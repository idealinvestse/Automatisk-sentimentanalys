"""Conversation analysis service — light path (default) or full pipeline."""

from __future__ import annotations

from typing import Any

from ...caching import AggregateCache
from ...core.serialization import map_results_to_segment_dicts, texts_from_segments, utc_now_iso
from ...pipeline import CallAnalysisPipeline
from ...sentiment import analyze_smart
from ..helpers import asr_kwargs_from, transcribe_helper
from ..schemas import (
    AnalyzeConversationRequest,
    AnalyzeConversationResponse,
    SegmentSentiment,
)


def _sentiment_profile(req: Any) -> str:
    return getattr(req, "sentiment_profile", None) or "callcenter"


def _build_segment_sentiments(
    texts: list[str],
    results: list[Any],
    segments: list[dict[str, Any]],
) -> list[SegmentSentiment]:
    dicts = map_results_to_segment_dicts(texts, results, segments)
    return [SegmentSentiment(**d) for d in dicts]


def _light_analyze(req: AnalyzeConversationRequest) -> AnalyzeConversationResponse:
    tr = transcribe_helper(**asr_kwargs_from(req, audio_path=req.audio_path))
    segments = tr.get("segments", []) or []
    tr_texts = texts_from_segments(segments)
    results, meta = analyze_smart(
        tr_texts,
        profile=_sentiment_profile(req),
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
    return AnalyzeConversationResponse(
        transcript=tr,
        segment_sentiments=seg_out,
        meta=meta,
        timestamp=utc_now_iso(),
    )


def _full_pipeline_analyze(
    req: AnalyzeConversationRequest,
    cache: AggregateCache | None,
) -> AnalyzeConversationResponse:
    pipe = CallAnalysisPipeline(
        sentiment_model=req.sentiment_model or "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=req.device,
        profile=_sentiment_profile(req),
        asr_backend=req.backend,
        asr_model=req.model,
        cache=cache,
    )
    report = pipe.analyze_audio(
        audio_path=req.audio_path,
        num_speakers=req.num_speakers,
        language=req.language,
        run_diarization=req.diarize,
        hotwords=req.hotwords,
        initial_prompt=req.initial_prompt,
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
    }
    return AnalyzeConversationResponse(
        transcript=transcript,
        segment_sentiments=seg_out,
        meta=meta,
        timestamp=utc_now_iso(),
        pipeline_results=report.results,
    )


def run_analyze_conversation(
    req: AnalyzeConversationRequest,
    *,
    cache: AggregateCache | None = None,
) -> AnalyzeConversationResponse:
    if req.use_full_pipeline:
        return _full_pipeline_analyze(req, cache)
    return _light_analyze(req)


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
    )


def run_batch_analyze_file(
    req: Any,
    audio_path: str,
    *,
    cache: AggregateCache | None = None,
) -> tuple[dict[str, Any], list[SegmentSentiment], dict[str, Any], dict[str, Any] | None]:
    """Single-file worker for batch/scan conversation analysis."""
    if getattr(req, "use_full_pipeline", False):
        conv_req = _scan_to_conversation_request(req, audio_path)
        resp = run_analyze_conversation(conv_req, cache=cache)
        return resp.transcript, resp.segment_sentiments, resp.meta, resp.pipeline_results

    tr = transcribe_helper(**asr_kwargs_from(req, audio_path=audio_path))
    segments = tr.get("segments", []) or []
    tr_texts = texts_from_segments(segments)
    results, meta = analyze_smart(
        tr_texts,
        profile=_sentiment_profile(req),
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
    return tr, seg_out, meta, None
