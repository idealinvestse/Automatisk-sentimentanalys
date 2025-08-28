from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, Tuple

_WORD_RE = re.compile(r"[\wäöåÄÖÅ]+", re.UNICODE)


@lru_cache(maxsize=8)
def load_lexicon(path: str) -> Dict[str, float]:
    """Load a Swedish sentiment lexicon from CSV/TSV.

    Expected columns (case-insensitive): one of (term|word) and one of (polarity|score|sentiment).
    Values should be in [-1, 1]. Lines with parse errors are skipped.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Lexicon not found: {path}")

    _, ext = os.path.splitext(path.lower())
    delimiter = "\t" if ext in {".tsv"} else ","

    lex: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
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


def score_text(text: str, lexicon: Dict[str, float]) -> float:
    """Return a scalar sentiment score in [-1, 1] using average of matched terms."""
    total = 0.0
    n = 0
    for tok in tokenize(text):
        if tok in lexicon:
            total += lexicon[tok]
            n += 1
    if n == 0:
        return 0.0
    s = total / n
    if s < -1.0:
        s = -1.0
    if s > 1.0:
        s = 1.0
    return s


def scalar_to_dist(s: float) -> Tuple[float, float, float]:
    """Map scalar sentiment [-1, 1] to a 3-class distribution (negativ, neutral, positiv)."""
    s = max(-1.0, min(1.0, s))
    p_pos = max(0.0, s)
    p_neg = max(0.0, -s)
    p_neu = 1.0 - min(1.0, abs(s))
    total = p_pos + p_neg + p_neu
    if total <= 0:
        return 0.33, 0.34, 0.33
    return p_neg / total, p_neu / total, p_pos / total


def blend_distributions(model: Dict[str, float], lex: Tuple[float, float, float], w: float) -> Dict[str, float]:
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
