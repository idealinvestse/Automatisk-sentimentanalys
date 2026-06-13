"""Fas 2: API service layer tests (pipeline cache, conversation service)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.services.conversation import run_analyze_conversation
from src.api.services.pipeline_cache import resolve_reports, segments_fingerprint
from src.caching import AggregateCache
from src.core.models import CallAnalysisReport
from src.pipeline import CallAnalysisPipeline


@pytest.fixture
def audio_file(tmp_path):
    p = tmp_path / "call.wav"
    p.write_bytes(b"RIFFxxxx")
    return str(p)


@pytest.fixture
def cache(tmp_path):
    return AggregateCache(use_redis=False, cache_dir=str(tmp_path / "agg"))


@pytest.fixture
def pipe(cache):
    return CallAnalysisPipeline(profile="callcenter", device="cpu", cache=cache)


def test_segments_fingerprint_stable():
    segs = [{"text": "hej", "start": 0, "end": 1}]
    assert segments_fingerprint(segs) == segments_fingerprint(segs)
    assert segments_fingerprint(segs) != segments_fingerprint([{"text": "nej"}])


def test_resolve_reports_uses_cache(pipe, cache):
    segs = [[{"text": "hej", "start": 0, "end": 1}]]
    fake = CallAnalysisReport(
        segments=segs[0],
        sentiment_results=[{"label": "positiv", "score": 0.9}],
        results={"agent_performance": {"agent": {"empathy_score": 0.8}}},
    )
    with patch.object(pipe, "analyze_segments", return_value=fake) as analyze:
        r1, hits1 = resolve_reports(pipe, segs)
        r2, hits2 = resolve_reports(pipe, segs)
    assert len(r1) == 1
    assert hits1 == 0
    assert hits2 == 1
    assert analyze.call_count == 1


def test_resolve_reports_reanalyze_bypasses_cache(pipe):
    segs = [[{"text": "hej"}]]
    fake = CallAnalysisReport(segments=segs[0], results={})
    with patch.object(pipe, "analyze_segments", return_value=fake) as analyze:
        resolve_reports(pipe, segs)
        resolve_reports(pipe, segs, reanalyze=True)
    assert analyze.call_count == 2


def test_run_analyze_conversation_light(audio_file):
    from src.api.schemas import AnalyzeConversationRequest

    req = AnalyzeConversationRequest(audio_path=audio_file)
    tr = {"segments": [{"text": "hej", "start": 0, "end": 1}]}
    with (
        patch("src.api.services.conversation.transcribe_helper", return_value=tr),
        patch(
            "src.api.services.conversation.analyze_smart",
            return_value=([{"label": "positiv", "score": 0.9}], {"profile": "call"}),
        ),
    ):
        out = run_analyze_conversation(req)
    assert out.segment_sentiments
    assert out.pipeline_results is None


def test_run_analyze_conversation_full_pipeline(audio_file, cache):
    from src.api.schemas import AnalyzeConversationRequest

    req = AnalyzeConversationRequest(audio_path=audio_file, use_full_pipeline=True)
    fake_report = CallAnalysisReport(
        segments=[{"text": "hej", "start": 0, "end": 1}],
        sentiment_results=[{"label": "positiv", "score": 0.9}],
        results={"qa": {"overall_qa_score": 80}},
    )
    with patch.object(CallAnalysisPipeline, "analyze_audio", return_value=fake_report):
        out = run_analyze_conversation(req, cache=cache)
    assert out.pipeline_results is not None
    assert out.pipeline_results.get("qa")
    assert out.meta.get("pipeline") is True
