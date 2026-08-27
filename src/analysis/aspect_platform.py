"""Aspect-evidence platform: claim charts and derived call sentiment."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

_SENTIMENT_SCORE = {
    "positive": 1.0,
    "positiv": 1.0,
    "negative": -1.0,
    "negativ": -1.0,
    "neutral": 0.0,
}


def _norm_sentiment(label: str | None) -> str:
    raw = (label or "neutral").strip().lower()
    if raw in {"positive", "positiv", "pos"}:
        return "positive"
    if raw in {"negative", "negativ", "neg"}:
        return "negative"
    return "neutral"


def prefer_aspect_claims(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer LLM refined_aspects (claim charts) over local ABSA sensor hits."""
    llm = results.get("llm") or {}
    refined = llm.get("refined_aspects") if isinstance(llm, dict) else None
    if isinstance(refined, list) and refined:
        claims: list[dict[str, Any]] = []
        for item in refined:
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence") or item.get("evidence_spans") or []
            spans: list[dict[str, Any]] = []
            if isinstance(evidence, list):
                for ev in evidence:
                    if isinstance(ev, dict):
                        spans.append(ev)
                    elif isinstance(ev, str):
                        spans.append({"text": ev})
            claims.append(
                {
                    "aspect": item.get("aspect", "annat"),
                    "sentiment": _norm_sentiment(item.get("sentiment")),
                    "score": float(item.get("score", 0.7)),
                    "evidence_spans": spans,
                    "related_to": item.get("related_to") or [],
                    "source": "llm_refined",
                }
            )
        return claims

    local = results.get("aspect") or []
    if not isinstance(local, list):
        return []
    claims = []
    for item in local:
        if not isinstance(item, dict):
            continue
        spans_raw = item.get("evidence_spans")
        item_spans: list[dict[str, Any]] | None = (
            spans_raw if isinstance(spans_raw, list) else None
        )
        if not item_spans and item.get("evidence"):
            item_spans = [
                {
                    "text": item["evidence"],
                    "speaker_role": item.get("speaker"),
                    "start": item.get("start"),
                    "end": item.get("end"),
                }
            ]
        claims.append(
            {
                "aspect": item.get("aspect", "annat"),
                "sentiment": _norm_sentiment(item.get("sentiment")),
                "score": float(item.get("score", 0.0)),
                "evidence_spans": item_spans or [],
                "related_to": [],
                "source": "local_absa",
            }
        )
    return claims


def derive_call_sentiment_from_aspects(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate aspect claims into a derived call-level sentiment (secondary product unit)."""
    if not claims:
        return {
            "label": "neutral",
            "score": 0.0,
            "aspect_count": 0,
            "by_aspect": {},
            "source": "derived_from_aspects",
        }

    weighted = 0.0
    weight_sum = 0.0
    by_aspect: dict[str, list[float]] = defaultdict(list)

    for c in claims:
        sent = _norm_sentiment(str(c.get("sentiment")))
        conf = abs(float(c.get("score", 0.5))) or 0.5
        signed = _SENTIMENT_SCORE.get(sent, 0.0) * conf
        weighted += signed
        weight_sum += conf
        by_aspect[str(c.get("aspect", "annat"))].append(signed)

    avg = weighted / weight_sum if weight_sum else 0.0
    if avg > 0.15:
        label = "positive"
    elif avg < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": round(avg, 4),
        "aspect_count": len(claims),
        "by_aspect": {k: round(sum(v) / len(v), 4) for k, v in by_aspect.items()},
        "source": "derived_from_aspects",
    }


def attach_aspect_platform(results: dict[str, Any]) -> dict[str, Any]:
    """Write aspect_claims + derived_call_sentiment onto results."""
    claims = prefer_aspect_claims(results)
    results["aspect_claims"] = claims
    results["derived_call_sentiment"] = derive_call_sentiment_from_aspects(claims)
    return results
