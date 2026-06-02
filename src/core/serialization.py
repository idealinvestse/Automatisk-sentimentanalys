"""Shared serialization helpers for sentiment results, timestamps, and segment data.

These utilities are used by both the CLI and API layers so that result
formatting is consistent and not duplicated.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

SENTIMENT_LABELS = frozenset({"negativ", "neutral", "positiv"})

# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def utc_now_iso(trim_microseconds: bool = True) -> str:
    """Return a UTC ISO-8601 timestamp ending with Z.

    Args:
        trim_microseconds: When True (default) drop sub-second precision.

    Returns:
        e.g. ``"2026-06-01T12:00:00Z"``
    """
    dt = datetime.now(UTC)
    if trim_microseconds:
        dt = dt.replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Sentiment score helpers
# ---------------------------------------------------------------------------


def score_dict(entries: Any) -> dict[str, float]:
    """Convert sentiment entries into a safe fixed-label score mapping.

    Handles both a single-dict result *and* a list-of-dicts (return_all_scores)
    format from the HuggingFace pipeline.

    Args:
        entries: A single ``{"label": ..., "score": ...}`` dict, or a list of
            such dicts produced when ``return_all_scores=True``.

    Returns:
        ``{"negativ": float, "neutral": float, "positiv": float}`` – all keys
        always present, missing labels default to 0.0.
    """
    scores: dict[str, float] = {"negativ": 0.0, "neutral": 0.0, "positiv": 0.0}
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return scores
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if label not in SENTIMENT_LABELS:
            continue
        try:
            scores[label] = float(entry.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid sentiment score for label %s: %r", label, entry)
    return scores


def single_label_distribution(result: Any) -> dict[str, float]:
    """Convert a single-label result to a full distribution dict.

    When *result* contains actual probability scores they are kept as-is;
    otherwise the predicted label receives a score of 1.0.

    Args:
        result: A single sentiment ``{"label": str, "score": float}`` dict.

    Returns:
        Full distribution: ``{"negativ": float, "neutral": float, "positiv": float}``.
    """
    scores = score_dict(result)
    if any(scores.values()):
        return scores
    if isinstance(result, dict):
        label = result.get("label")
        if label in SENTIMENT_LABELS:
            scores[label] = 1.0
    return scores


def top_label(scores: dict[str, float]) -> tuple[str, float]:
    """Return the label with the highest score and its value.

    Args:
        scores: Mapping of label → probability/score.

    Returns:
        ``(label, score)`` for the highest-scoring label.
        Falls back to ``("neutral", 0.0)`` for an empty dict.
    """
    if not scores:
        return "neutral", 0.0
    label = max(scores, key=lambda k: scores[k])
    return label, float(scores[label])


# ---------------------------------------------------------------------------
# Segment / transcript helpers
# ---------------------------------------------------------------------------


def segment_time(segment: dict[str, Any], key: str) -> float | None:
    """Extract a float timestamp from a segment dict.

    Args:
        segment: ASR segment dict.
        key: Key to look up (e.g. ``"start"`` or ``"end"``).

    Returns:
        Float timestamp, or *None* if the key is absent or non-numeric.
    """
    value = segment.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def texts_from_segments(segments: list[dict[str, Any]]) -> list[str]:
    """Extract a list of stripped texts from ASR segment dicts.

    When all segments have non-empty text one string per segment is returned.
    If the result would be all-empty the texts are joined into a single entry
    as a fallback.

    Args:
        segments: List of ASR segment dicts, each with at least a ``"text"`` key.

    Returns:
        List of text strings (one per segment, or a single joined fallback).
    """
    texts = [s.get("text", "").strip() for s in segments]
    if texts and any(texts):
        return texts
    joined = " ".join(s.get("text", "").strip() for s in segments if s.get("text")).strip()
    return [joined] if joined else []


def map_results_to_segment_dicts(
    texts: list[str],
    results: list[Any],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map sentiment results back onto transcript segments as plain dicts.

    Each returned dict contains the keys:
    ``index``, ``start``, ``end``, ``text``, ``label``, ``score``,
    ``negativ``, ``neutral``, ``positiv``.

    Args:
        texts: Extracted texts – one per segment.
        results: Sentiment results from the pipeline – one per text.
        segments: Original ASR segment dicts (used for ``start``/``end``).

    Returns:
        List of dicts, one per (text, result) pair.
    """
    if len(texts) != len(results):
        logger.warning(
            "Segment sentiment length mismatch: texts=%d results=%d",
            len(texts),
            len(results),
        )
    out: list[dict[str, Any]] = []
    for idx, (text, result) in enumerate(zip(texts, results, strict=False)):
        scores_map = score_dict(result)
        lbl, score = top_label(scores_map)
        seg = segments[idx] if idx < len(segments) else {}
        out.append(
            {
                "index": idx,
                "start": segment_time(seg, "start"),
                "end": segment_time(seg, "end"),
                "text": text,
                "label": lbl,
                "score": score,
                "negativ": scores_map.get("negativ"),
                "neutral": scores_map.get("neutral"),
                "positiv": scores_map.get("positiv"),
            }
        )
    return out
