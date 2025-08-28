from __future__ import annotations

from typing import List, Dict, Optional, Union, Tuple
from functools import lru_cache
from transformers import pipeline
import torch
from .profiles import resolve_profile
from .clean import clean_texts

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

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[Union[int, str, torch.device]] = None,
        return_all_scores: bool = False,
        max_length: int = 256,
    ):
        self.model_name = model_name
        device_arg, device_key = _normalize_device_spec(device)
        self.device_key = device_key
        self.return_all_scores = return_all_scores
        self.max_length = max_length
        self._nlp = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device_arg,
        )

    def analyze(
        self,
        texts: List[str],
        batch_size: int = 16,
        normalize: bool = True,
        return_all_scores: Optional[bool] = None,
        max_length: Optional[int] = None,
    ) -> List[Dict]:
        ras = self.return_all_scores if return_all_scores is None else return_all_scores
        ml = self.max_length if max_length is None else max_length
        # Build kwargs to control output shape without deprecated flags
        call_kwargs = dict(batch_size=batch_size, truncation=True, max_length=ml)
        if ras:
            # Full distribution using modern API
            call_kwargs["top_k"] = None
        raw = self._nlp(texts, **call_kwargs)

        if not normalize:
            return raw

        # Normalize labels depending on output shape
        if ras:
            # List[List[Dict[label, score]]]
            for i, inner in enumerate(raw):
                for j, entry in enumerate(inner):
                    entry["label"] = normalize_label(entry.get("label"))
            return raw
        else:
            # List[Dict[label, score]]
            for r in raw:
                r["label"] = normalize_label(r.get("label"))
            return raw


def load(
    model_name: str = DEFAULT_MODEL,
    device: Optional[Union[int, str, torch.device]] = "auto",
    return_all_scores: bool = False,
    max_length: int = 256,
) -> SentimentPipeline:
    """Load a minimal sentiment pipeline for the given model."""
    device_arg, _ = _normalize_device_spec(device)
    return SentimentPipeline(
        model_name=model_name,
        device=device_arg,
        return_all_scores=return_all_scores,
        max_length=max_length,
    )


@lru_cache(maxsize=4)
def _get_cached(model_name: str, device_key: str) -> SentimentPipeline:
    device_arg = _device_arg_from_key(device_key)
    return SentimentPipeline(model_name=model_name, device=device_arg)


def analyze(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    device: Optional[Union[int, str, torch.device]] = "auto",
    batch_size: int = 16,
    normalize: bool = True,
    return_all_scores: bool = False,
    max_length: int = 256,
) -> List[Dict]:
    """Analyze texts in one call using a cached pipeline instance.

    Example:
        from src.sentiment import analyze
        results = analyze(["Det här är bra!"])
    """
    _, device_key = _normalize_device_spec(device)
    return _get_cached(model_name, device_key).analyze(
        texts,
        batch_size=batch_size,
        normalize=normalize,
        return_all_scores=return_all_scores,
        max_length=max_length,
    )


def analyze_smart(
    texts: List[str],
    datatype: Optional[str] = None,
    source: Optional[str] = None,
    profile: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[Union[int, str, torch.device]] = "auto",
    batch_size: int = 16,
    normalize: bool = True,
    return_all_scores: bool = False,
    max_length: Optional[int] = None,
    clean: bool = True,
) -> Tuple[List[Dict], Dict[str, Union[str, int]]]:
    """Profile-aware analysis.

    - Resolves a profile from (profile | source | datatype)
    - Picks model and max_length defaults from the profile if not provided
    - Optionally cleans texts according to the profile's cleaning spec
    - Uses cached pipeline by (model, device)

    Returns (results, meta) where meta contains {"profile", "model", "max_length"}.
    """
    profile_name, spec = resolve_profile(datatype=datatype, source=source, profile=profile)
    chosen_model = model_name or spec.get("model", DEFAULT_MODEL)
    resolved_max_length = max_length or spec.get("max_length", 256)

    proc_texts = clean_texts(texts, spec.get("cleaning", {})) if clean else list(texts)

    results = analyze(
        proc_texts,
        model_name=chosen_model,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        return_all_scores=return_all_scores,
        max_length=resolved_max_length,
    )
    meta: Dict[str, Union[str, int]] = {
        "profile": profile_name,
        "model": chosen_model,
        "max_length": int(resolved_max_length),
    }
    return results, meta


def analyze_one(text: str, model_name: str = DEFAULT_MODEL, normalize: bool = True) -> Dict:
    """Analyze a single text and return one result dict."""
    return analyze([text], model_name=model_name, batch_size=1, normalize=normalize)[0]


def _normalize_device_spec(
    device: Optional[Union[int, str, torch.device]]
) -> Tuple[Union[int, torch.device], str]:
    """Return (device_arg_for_pipeline, device_key_for_cache)."""
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        if torch.cuda.is_available():
            return 0, "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return -1, "cpu"

    if isinstance(device, int):
        if device >= 0 and torch.cuda.is_available():
            return device, f"cuda:{device}"
        return -1, "cpu"

    if isinstance(device, torch.device):
        if device.type == "cuda" and torch.cuda.is_available():
            idx = device.index if device.index is not None else 0
            return idx, f"cuda:{idx}"
        if device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return device, "mps"
        return -1, "cpu"

    if isinstance(device, str):
        d = device.strip().lower()
        if d == "cpu":
            return -1, "cpu"
        if d.startswith("cuda"):
            idx = 0
            if ":" in d:
                try:
                    idx = int(d.split(":", 1)[1])
                except Exception:
                    idx = 0
            if torch.cuda.is_available():
                return idx, f"cuda:{idx}"
            return -1, "cpu"
        if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return -1, "cpu"

    return -1, "cpu"


def _device_arg_from_key(key: str) -> Union[int, torch.device]:
    if key == "cpu":
        return -1
    if key == "mps":
        return torch.device("mps")
    if key.startswith("cuda:"):
        try:
            idx = int(key.split(":", 1)[1])
        except Exception:
            idx = 0
        return idx
    return -1


__all__ = [
    "DEFAULT_MODEL",
    "SentimentPipeline",
    "load",
    "analyze",
    "analyze_smart",
    "analyze_one",
]
