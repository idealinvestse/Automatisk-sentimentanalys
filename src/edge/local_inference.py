"""Offline local inference for Edge AI MVP — no network I/O."""

from __future__ import annotations

import logging
from typing import Any

from .contracts import EdgeAnalysisResult, EdgeSegmentResult

logger = logging.getLogger(__name__)


def _run_local_negation_aspect(
    segments: list[Any],
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run negation + aspect analyzers on redacted segments (offline registry path)."""
    from ..analysis.aspect import AspectAnalyzer
    from ..analysis.negation import NegationAnalyzer
    from ..core.models import AnalysisContext

    ctx = AnalysisContext(segments=segments, results=results)
    negation = NegationAnalyzer().analyze(ctx)
    ctx.results["negation"] = negation
    aspects = AspectAnalyzer().analyze(ctx)
    return negation, aspects


def analyze_text_offline(
    text: str,
    *,
    profile: str = "callcenter",
) -> EdgeAnalysisResult:
    """Run sentiment + heuristic intent + negation/aspect on plain text."""
    from ..core.models import Segment
    from ..intent import IntentClassifier
    from ..sentiment import analyze_smart

    classifier = IntentClassifier(backend="heuristic")
    results, meta = analyze_smart([text], profile=profile)
    sent = results[0] if results else {}
    intent_label, _conf = classifier.classify(text)
    seg = Segment(start=0.0, end=0.0, text=text)
    negation, aspects = _run_local_negation_aspect(
        [seg],
        {"sentiment": results},
    )
    neg = negation[0] if negation else {}
    seg_aspects = [a for a in aspects if a.get("start") == seg.start and a.get("end") == seg.end]
    return EdgeAnalysisResult(
        profile=profile,
        segments=[
            EdgeSegmentResult(
                text=text,
                sentiment_label=sent.get("label"),
                sentiment_score=sent.get("score"),
                intent=intent_label,
                has_negation=bool(neg.get("has_negation")),
                aspects=seg_aspects[:5],
            )
        ],
        summary=f"Offline analysis ({meta.get('profile', profile)})",
        limitations=[
            "No LLM",
            "No diarization (pyannote)",
            "No Fas 4 aggregate endpoints",
            "Negation + aspect via local registry analyzers only",
        ],
    )


def analyze_segments_offline(
    segments: list[dict[str, Any]],
    *,
    profile: str = "callcenter",
) -> EdgeAnalysisResult:
    """Run offline analysis on pre-transcribed segments."""
    from ..core.models import Segment
    from ..intent import IntentClassifier
    from ..pipeline_steps import apply_early_pii_redaction
    from ..sentiment import analyze_smart

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
    negation, aspects = _run_local_negation_aspect(redacted, {"sentiment": sentiments})
    out_segments: list[EdgeSegmentResult] = []
    for idx, (seg, sent) in enumerate(zip(redacted, sentiments, strict=False)):
        neg = negation[idx] if idx < len(negation) else {}
        seg_aspects = [
            a for a in aspects if a.get("start") == seg.start and a.get("end") == seg.end
        ]
        out_segments.append(
            EdgeSegmentResult(
                text=seg.text,
                sentiment_label=sent.get("label"),
                sentiment_score=sent.get("score"),
                intent=classifier.classify(seg.text)[0],
                has_negation=bool(neg.get("has_negation")),
                aspects=seg_aspects[:5],
            )
        )
    return EdgeAnalysisResult(
        profile=profile,
        segments=out_segments,
        summary=f"Offline segment analysis ({len(out_segments)} segments)",
        limitations=[
            "No LLM",
            "No diarization (pyannote)",
            "No Fas 4 aggregate endpoints",
            "Negation + aspect via local registry analyzers only",
        ],
    )
