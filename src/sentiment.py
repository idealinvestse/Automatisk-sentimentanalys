from __future__ import annotations

from typing import List, Dict
from functools import lru_cache
from transformers import pipeline

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def normalize_label(label: str) -> str:
    l = str(label).strip().lower()
    if l in {"label_0", "negative", "neg"}:
        return "negativ"
    if l in {"label_1", "neutral"}:
        return "neutral"
    if l in {"label_2", "positive", "pos"}:
        return "positiv"
    return label


class SentimentPipeline:
    """Minimal wrapper around Hugging Face sentiment-analysis pipeline.

    Usage:
        sp = SentimentPipeline()
        results = sp.analyze(["Det här är bra!"])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._nlp = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
        )

    def analyze(self, texts: List[str], batch_size: int = 16, normalize: bool = True) -> List[Dict]:
        raw = self._nlp(texts, batch_size=batch_size, truncation=True, max_length=256)
        if normalize:
            for r in raw:
                r["label"] = normalize_label(r.get("label"))
        return raw


def load(model_name: str = DEFAULT_MODEL) -> SentimentPipeline:
    """Load a minimal sentiment pipeline for the given model."""
    return SentimentPipeline(model_name)


@lru_cache(maxsize=2)
def _get_cached(model_name: str = DEFAULT_MODEL) -> SentimentPipeline:
    return SentimentPipeline(model_name)


def analyze(texts: List[str], model_name: str = DEFAULT_MODEL, batch_size: int = 16, normalize: bool = True) -> List[Dict]:
    """Analyze texts in one call using a cached pipeline instance.

    Example:
        from src.sentiment import analyze
        results = analyze(["Det här är bra!"])
    """
    return _get_cached(model_name).analyze(texts, batch_size=batch_size, normalize=normalize)


def analyze_one(text: str, model_name: str = DEFAULT_MODEL, normalize: bool = True) -> Dict:
    """Analyze a single text and return one result dict."""
    return analyze([text], model_name=model_name, batch_size=1, normalize=normalize)[0]


__all__ = [
    "DEFAULT_MODEL",
    "SentimentPipeline",
    "load",
    "analyze",
    "analyze_one",
]
