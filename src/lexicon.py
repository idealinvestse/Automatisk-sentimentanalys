from __future__ import annotations

import csv
import logging
import os
import re
from collections.abc import Iterable
from functools import lru_cache

_WORD_RE = re.compile(r"[\wäöåÄÖÅ]+", re.UNICODE)
NEGATIONS = {"inte", "ej", "aldrig", "icke", "knappast"}

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def load_lexicon(path: str) -> dict[str, float]:
    """Load a Swedish sentiment lexicon from CSV/TSV.

    Expected columns (case-insensitive): one of (term|word) and one of (polarity|score|sentiment).
    Values should be in [-1, 1]. Lines with parse errors are skipped.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Lexicon not found: {path}")

    _, ext = os.path.splitext(path.lower())
    delimiter = "\t" if ext in {".tsv"} else ","

    lex: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = {h.strip().lower() for h in reader.fieldnames or []}
        term_key = "term" if "term" in headers else ("word" if "word" in headers else None)
        score_key = None
        for k in ("polarity", "score", "sentiment"):
            if k in headers:
                score_key = k
                break
        if not term_key or not score_key:
            raise ValueError("Lexicon must contain columns: term|word and polarity|score|sentiment")
        for row in reader:
            try:
                term = str(row[term_key]).strip().lower()
                if not term:
                    continue
                score = float(row[score_key])
                if score < -1.0:
                    score = -1.0
                if score > 1.0:
                    score = 1.0
                lex[term] = score
            except Exception:
                continue
    return lex


def tokenize(text: str) -> Iterable[str]:
    for m in _WORD_RE.finditer(text.lower()):
        yield m.group(0)


def score_text(text: str, lexicon: dict[str, float]) -> float:
    """Return a scalar sentiment score in [-1, 1] using Swedish negation handling."""
    total = 0.0
    n = 0
    negation_window = 0
    for tok in tokenize(text):
        if tok in NEGATIONS:
            negation_window = 3
            continue
        score = lexicon.get(tok)
        if score is None:
            # Simple Swedish compound-word fallback: match lexicon suffixes such as "kundservice".
            score = next(
                (value for term, value in lexicon.items() if len(term) > 3 and tok.endswith(term)),
                None,
            )
        if score is not None:
            if negation_window > 0:
                score *= -0.8
            total += score
            n += 1
        if negation_window > 0:
            negation_window -= 1
    if n == 0:
        return 0.0
    s = total / n
    if s < -1.0:
        s = -1.0
    if s > 1.0:
        s = 1.0
    return s


def scalar_to_dist(s: float) -> tuple[float, float, float]:
    """Map scalar sentiment [-1, 1] to a 3-class distribution (negativ, neutral, positiv)."""
    s = max(-1.0, min(1.0, s))
    p_pos = max(0.0, s)
    p_neg = max(0.0, -s)
    p_neu = 1.0 - min(1.0, abs(s))
    total = p_pos + p_neg + p_neu
    if total <= 0:
        return 0.33, 0.34, 0.33
    return p_neg / total, p_neu / total, p_pos / total


def blend_distributions(
    model: dict[str, float], lex: tuple[float, float, float], w: float
) -> dict[str, float]:
    """Blend model distribution dict with lexicon 3-tuple using weight w in [0,1]."""
    w = max(0.0, min(1.0, w))
    mn = model.get("negativ", 0.0)
    me = model.get("neutral", 0.0)
    mp = model.get("positiv", 0.0)
    ln, le, lp = lex
    out = {
        "negativ": (1 - w) * mn + w * ln,
        "neutral": (1 - w) * me + w * le,
        "positiv": (1 - w) * mp + w * lp,
    }
    s = out["negativ"] + out["neutral"] + out["positiv"]
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def blend_results_with_lexicon(
    texts: list[str],
    results: list,
    lexicon_file: str | None,
    lexicon_weight: float,
    segment_confidences: list[float | None] | None = None,
) -> list:
    """Blend model results with lexicon.

    When ``segment_confidences`` is provided, low-confidence segments
    (confidence < ~0.60) receive an automatically boosted lexicon weight
    (up to 1.0). This implements the call-center requirement that uncertain
    ASR output should lean more on the trustworthy Swedish lexicon.
    """
    """Blend a list of model sentiment results with lexicon scores.

    This is the high-level helper used by the API and CLI layers.  It handles
    length mismatches gracefully and catches lexicon-loading errors so that the
    caller always receives a valid result list.

    Args:
        texts: Source texts corresponding to ``results``.
        results: Sentiment results from the model pipeline.  Each entry is
            either a single ``{"label": str, "score": float}`` dict
            (``return_all_scores=False``) or a list of such dicts
            (``return_all_scores=True``).
        lexicon_file: Path to a CSV/TSV Swedish lexicon, or *None*/empty to
            skip blending.
        lexicon_weight: Blend weight in ``[0, 1]``.  0 = model only,
            1 = lexicon only.

    Returns:
        New list with the same structure as ``results`` but with scores
        adjusted by the lexicon.  Returns ``results`` unchanged when
        ``lexicon_file`` is falsy or ``lexicon_weight <= 0``.
    """
    if not lexicon_file or lexicon_weight <= 0.0:
        return results
    if len(texts) != len(results):
        logger.warning(
            "Lexicon blending length mismatch: texts=%d results=%d",
            len(texts),
            len(results),
        )
    try:
        lex = load_lexicon(lexicon_file)
        blended: list = []
        full_distribution = bool(results and isinstance(results[0], list))
        n = len(texts)
        confs = segment_confidences or [None] * n

        for idx, (text, result) in enumerate(zip(texts, results, strict=False)):
            if full_distribution:
                scores: dict[str, float] = {
                    entry.get("label"): float(entry.get("score", 0.0) or 0.0)
                    for entry in result
                    if isinstance(entry, dict)
                    and entry.get("label") in {"negativ", "neutral", "positiv"}
                }
            else:
                scores = {"negativ": 0.0, "neutral": 0.0, "positiv": 0.0}
                if isinstance(result, dict):
                    lbl = result.get("label")
                    val = float(result.get("score", 0.0) or 0.0)
                    if lbl in scores:
                        scores[lbl] = val
                    else:
                        # Single-label without probability – give full weight
                        if lbl in {"negativ", "neutral", "positiv"}:
                            scores[lbl] = 1.0
            # Ensure all labels present
            for lbl in ("negativ", "neutral", "positiv"):
                scores.setdefault(lbl, 0.0)

            # Per-segment lexicon weight boost for low ASR confidence (Task 1.2)
            eff_weight = lexicon_weight
            c = confs[idx] if idx < len(confs) else None
            if c is not None and c < 0.60:
                # Boost lexicon trust when the acoustic model is uncertain.
                # The boost is capped so we never go completely lexicon-only unless
                # the caller already asked for lexicon_weight=1.0.
                boost = (0.60 - float(c)) * 1.5
                eff_weight = min(1.0, lexicon_weight + boost)

            lex_dist = scalar_to_dist(score_text(text, lex))
            scores = blend_distributions(scores, lex_dist, eff_weight)

            if full_distribution:
                blended.append([
                    {"label": "negativ", "score": scores["negativ"]},
                    {"label": "neutral", "score": scores["neutral"]},
                    {"label": "positiv", "score": scores["positiv"]},
                ])
            else:
                best = max(scores.items(), key=lambda kv: kv[1])[0]
                blended.append({"label": best, "score": float(scores[best])})
        return blended
    except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
        logger.warning("Lexicon blending failed for %s: %s", lexicon_file, e, exc_info=True)
        return results
