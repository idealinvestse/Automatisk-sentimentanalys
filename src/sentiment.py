from __future__ import annotations

from functools import lru_cache

import torch
from transformers import pipeline

from .clean import clean_texts
from .core.device import device_arg_from_key, normalize_device_spec
from .core.errors import AnalysisError
from .negation import apply_negation_heuristic, detect_negation, get_intensity_multiplier
from .profiles import resolve_profile

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
POLITE_POSITIVE_TERMS = {"tack", "vänlig", "hjälpsam", "uppskattar"}

_normalize_device_spec = normalize_device_spec
_device_arg_from_key = device_arg_from_key


def normalize_label(label: str | None) -> str:
    if label is None:
        return "neutral"
    lowered = str(label).strip().lower()
    if lowered in {"label_0", "negative", "neg"}:
        return "negativ"
    if lowered in {"label_1", "neutral"}:
        return "neutral"
    if lowered in {"label_2", "positive", "pos"}:
        return "positiv"
    return label


def adjust_distribution_for_callcenter(text: str, dist: dict[str, float]) -> dict[str, float]:
    """Apply lightweight call-center heuristics for negation and polite phrases."""
    adjusted = dict(dist)
    lowered = text.lower()
    if detect_negation(text):
        adjusted["negativ"], adjusted["positiv"] = (
            adjusted.get("positiv", 0.0),
            adjusted.get("negativ", 0.0),
        )
    if any(term in lowered for term in POLITE_POSITIVE_TERMS):
        adjusted["positiv"] = adjusted.get("positiv", 0.0) + 0.05
        adjusted["neutral"] = max(0.0, adjusted.get("neutral", 0.0) - 0.05)
    # New: intensity scaling
    mult = get_intensity_multiplier(text)
    if mult != 1.0:
        for k in ("negativ", "positiv"):
            adjusted[k] = adjusted.get(k, 0.0) * mult
        adjusted["neutral"] = max(0.0, adjusted.get("neutral", 0.0) * (2 - mult))  # damp neutral a bit
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()} if total > 0 else adjusted


class SentimentPipeline:
    """Minimal wrapper around Hugging Face sentiment-analysis pipeline.

    Usage:
        sp = SentimentPipeline()
        results = sp.analyze(["Det här är bra!"])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: int | str | torch.device | None = None,
        return_all_scores: bool = False,
        max_length: int = 256,
    ):
        self.model_name = model_name
        device_arg, device_key = normalize_device_spec(device)
        self.device_key = device_key
        self.return_all_scores = return_all_scores
        self.max_length = max_length
        try:
            self._nlp = pipeline(
                task="sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device_arg,
            )
        except Exception as e:
            raise AnalysisError(f"Failed to initialize sentiment model '{model_name}': {e}") from e

    def analyze(
        self,
        texts: list[str],
        batch_size: int = 16,
        normalize: bool = True,
        return_all_scores: bool | None = None,
        max_length: int | None = None,
    ) -> list[dict]:
        ras = self.return_all_scores if return_all_scores is None else return_all_scores
        ml = self.max_length if max_length is None else max_length
        # Build kwargs to control output shape without deprecated flags
        call_kwargs = dict(batch_size=batch_size, truncation=True, max_length=ml)
        if ras:
            # Full distribution using modern API
            call_kwargs["top_k"] = None
        try:
            raw = self._nlp(texts, **call_kwargs)
        except Exception as e:
            raise AnalysisError(f"Sentiment inference failed for {len(texts)} text(s): {e}") from e

        if not normalize:
            return raw

        # Normalize labels depending on output shape
        if ras:
            # List[List[Dict[label, score]]]
            for inner in raw:
                for entry in inner:
                    entry["label"] = normalize_label(entry.get("label"))
            return raw
        else:
            # List[Dict[label, score]]
            for r in raw:
                r["label"] = normalize_label(r.get("label"))
            return raw


def load(
    model_name: str = DEFAULT_MODEL,
    device: int | str | torch.device | None = "auto",
    return_all_scores: bool = False,
    max_length: int = 256,
) -> SentimentPipeline:
    """Load a minimal sentiment pipeline for the given model."""
    device_arg, _ = normalize_device_spec(device)
    return SentimentPipeline(
        model_name=model_name,
        device=device_arg,
        return_all_scores=return_all_scores,
        max_length=max_length,
    )


@lru_cache(maxsize=4)
def _get_cached(model_name: str, device_key: str) -> SentimentPipeline:
    """Return a shared pipeline via ModelResourcePool (unified with analyzer path)."""
    from .analysis.resources import get_pool

    device_arg = device_arg_from_key(device_key)
    return get_pool().get_sentiment_pipeline(
        model_name=model_name,
        device=device_arg,
    )


def analyze(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    device: int | str | torch.device | None = "auto",
    batch_size: int = 16,
    normalize: bool = True,
    return_all_scores: bool = False,
    max_length: int = 256,
) -> list[dict]:
    """Analyze texts in one call using a cached pipeline instance.

    Example:
        from src.sentiment import analyze
        results = analyze(["Det här är bra!"])
    """
    _, device_key = normalize_device_spec(device)
    return _get_cached(model_name, device_key).analyze(
        texts,
        batch_size=batch_size,
        normalize=normalize,
        return_all_scores=return_all_scores,
        max_length=max_length,
    )


def analyze_smart(
    texts: list[str],
    datatype: str | None = None,
    source: str | None = None,
    profile: str | None = None,
    model_name: str | None = None,
    device: int | str | torch.device | None = "auto",
    batch_size: int = 16,
    normalize: bool = True,
    return_all_scores: bool = False,
    max_length: int | None = None,
    clean: bool = True,
    lexicon_file: str | None = None,
    lexicon_weight: float | None = None,
) -> tuple[list[dict], dict[str, str | int]]:
    """Profile-aware analysis.

    - Resolves a profile from (profile | source | datatype)
    - Picks model and max_length defaults from the profile if not provided
    - Optionally cleans texts according to the profile's cleaning spec
    - Uses cached pipeline by (model, device)
    - If profile (or explicit) specifies lexicon_file/weight, applies blending
      (callers can still override by passing explicit values).

    Returns (results, meta) where meta contains {"profile", "model", "max_length", ...}.
    """
    profile_name, spec = resolve_profile(datatype=datatype, source=source, profile=profile)
    chosen_model = model_name or spec.get("model", DEFAULT_MODEL)
    resolved_max_length = max_length or spec.get("max_length", 256)

    # Default lexicon from profile if not explicitly provided (None means use profile)
    if lexicon_file is None:
        lexicon_file = spec.get("lexicon_file")
    if lexicon_weight is None:
        lexicon_weight = spec.get("lexicon_weight", 0.0)

    proc_texts = clean_texts(texts, spec.get("cleaning", {})) if clean else list(texts)

    # For callcenter profile we always want full distributions internally to apply
    # the domain heuristics, then collapse at the end if caller didn't ask for them.
    internal_return_all = return_all_scores or (profile_name == "callcenter")
    results = analyze(
        proc_texts,
        model_name=chosen_model,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        return_all_scores=internal_return_all,
        max_length=resolved_max_length,
    )
    if profile_name == "callcenter":
        adjusted_results = []
        if len(proc_texts) != len(results):
            raise AnalysisError(
                f"Sentiment result length mismatch: {len(proc_texts)} texts vs {len(results)} results"
            )
        for text, inner in zip(proc_texts, results, strict=True):
            if isinstance(inner, list):
                dist = {
                    entry["label"]: float(entry.get("score", 0.0) or 0.0)
                    for entry in inner
                    if isinstance(entry, dict)
                    and entry.get("label") in {"negativ", "neutral", "positiv"}
                }
                dist = adjust_distribution_for_callcenter(text, dist)
                adjusted_results.append(
                    [{"label": label, "score": score} for label, score in dist.items()]
                )
            else:
                adjusted_results.append(inner)
        results = adjusted_results
        if not return_all_scores:
            # Collapse back to single-label form if caller didn't request full scores
            collapsed = []
            for lst in results:
                if isinstance(lst, list):
                    best = max(lst, key=lambda e: e.get("score", 0))
                    collapsed.append({"label": best["label"], "score": best["score"]})
                else:
                    collapsed.append(lst)
            results = collapsed

    hybrid = False
    # Apply lexicon blending if we have a file and positive weight (from profile default or explicit)
    if lexicon_file and lexicon_weight is not None and float(lexicon_weight) > 0.0:
        from .lexicon import blend_results_with_lexicon
        # Simple hybrid (prop10): boost lexicon weight if model uncertain on any item
        eff_w = float(lexicon_weight)
        try:
            if internal_return_all or profile_name == "callcenter":
                for r in results:
                    if isinstance(r, list):
                        mx = max((e.get("score", 0.0) for e in r), default=0.0)
                        if mx < 0.55:
                            eff_w = min(1.0, eff_w + 0.35)
                            hybrid = True
                            break
        except Exception:
            pass
        results = blend_results_with_lexicon(proc_texts, results, lexicon_file, eff_w)

    meta: dict[str, str | int | float | None] = {
        "profile": profile_name,
        "model": chosen_model,
        "max_length": int(resolved_max_length),
    }
    if lexicon_file:
        meta["lexicon_file"] = lexicon_file
        meta["lexicon_weight"] = float(lexicon_weight) if lexicon_weight is not None else 0.0
    if hybrid:
        meta["hybrid_lexicon_boost"] = True
    return results, meta


def analyze_one(text: str, model_name: str = DEFAULT_MODEL, normalize: bool = True) -> dict:
    """Analyze a single text and return one result dict."""
    results = analyze([text], model_name=model_name, batch_size=1, normalize=normalize)
    if not results:
        raise AnalysisError("Sentiment analysis returned no result for single text")
    return results[0]


__all__ = [
    "DEFAULT_MODEL",
    "SentimentPipeline",
    "load",
    "analyze",
    "analyze_smart",
    "analyze_one",
    "detect_negation",
    "adjust_distribution_for_callcenter",
    "apply_negation_heuristic",
    "clean_texts",
]
