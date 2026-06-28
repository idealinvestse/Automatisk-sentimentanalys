"""Offline local inference for Edge AI MVP — no network I/O."""

from __future__ import annotations

import logging
from typing import Any

from .contracts import EdgeAnalysisResult, EdgeSegmentResult

logger = logging.getLogger(__name__)


def analyze_text_offline(
    text: str,
    *,
    profile: str = "callcenter",
) -> EdgeAnalysisResult:
    """Run sentiment + heuristic intent on plain text."""
    from ..intent import IntentClassifier
    from ..sentiment import analyze_smart

    classifier = IntentClassifier(backend="heuristic")
    results, meta = analyze_smart([text], profile=profile)
    sent = results[0] if results else {}
    intent_label, _conf = classifier.classify(text)
    return EdgeAnalysisResult(
        profile=profile,
        segments=[
            EdgeSegmentResult(
                text=text,
                sentiment_label=sent.get("label"),
                sentiment_score=sent.get("score"),
                intent=intent_label,
            )
        ],
        summary=f"Offline analysis ({meta.get('profile', profile)})",
    )


def analyze_segments_offline(
    segments: list[dict[str, Any]],
    *,
    profile: str = "callcenter",
) -> EdgeAnalysisResult:
    """Run offline analysis on pre-transcribed segments."""
    from ..intent import IntentClassifier
    from ..sentiment import analyze_smart
    from ..pipeline_steps import apply_early_pii_redaction
    from ..core.models import Segment

    classifier = IntentClassifier(backend="heuristic")

    typed = [
        Segment(
            start=float(s.get("start", 0) or 0),
            end=float(s.get("end", 0) or 0),
            text=str(s.get("text", "")),
            speaker=s.get("speaker"),
        )
        for s in segments
    ]
    redacted, _pii = apply_early_pii_redaction(typed, profile_name=profile)
    texts = [s.text for s in redacted if s.text]
    sentiments, _meta = analyze_smart(texts, profile=profile) if texts else ([], {})
    out_segments: list[EdgeSegmentResult] = []
    for seg, sent in zip(redacted, sentiments, strict=False):
        out_segments.append(
            EdgeSegmentResult(
                text=seg.text,
                sentiment_label=sent.get("label"),
                sentiment_score=sent.get("score"),
                intent=classifier.classify(seg.text)[0],
            )
        )
    return EdgeAnalysisResult(
        profile=profile,
        segments=out_segments,
        summary=f"Offline segment analysis ({len(out_segments)} segments)",
    )
