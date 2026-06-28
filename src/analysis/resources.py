"""Shared model resource pool for analyzers (lazy loading, single instance per config)."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from ..core.device import normalize_device_spec
from ..intent import IntentClassifier
from ..sentiment import SentimentPipeline

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "models/intent_classifier")

_pool: ModelResourcePool | None = None
_pool_lock = threading.Lock()


class ModelResourcePool:
    """Process-wide cache of heavy ML resources keyed by configuration."""

    def __init__(self, maxsize: int = 8) -> None:
        self._cache: dict[tuple[str, ...], Any] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def _get_or_create(self, key: tuple[str, ...], factory: Callable[[], T]) -> T:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if len(self._cache) >= self._maxsize:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                logger.debug("Evicted resource pool entry: %s", oldest)
            instance = factory()
            self._cache[key] = instance
            logger.debug("Created resource pool entry: %s", key)
            return instance

    def get_sentiment_pipeline(
        self,
        model_name: str,
        device: str | None = None,
        return_all_scores: bool = False,
    ) -> SentimentPipeline:
        device_arg, device_key = normalize_device_spec(device or "auto")
        key = ("sentiment", model_name, device_key, str(return_all_scores))
        return self._get_or_create(
            key,
            lambda: SentimentPipeline(
                model_name=model_name,
                device=device_arg,
                return_all_scores=return_all_scores,
            ),
        )

    def get_intent_classifier(self, backend: str = "heuristic") -> IntentClassifier:
        model_path = DEFAULT_INTENT_MODEL_PATH if backend == "model" else None
        if backend == "model" and not os.path.isdir(model_path or ""):
            logger.debug("Intent model path missing (%s); using heuristic", model_path)
            backend = "heuristic"
            model_path = None
        key = ("intent", backend, model_path or "")
        return self._get_or_create(
            key,
            lambda: IntentClassifier(backend=backend, model_path=model_path),
        )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def get_pool() -> ModelResourcePool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ModelResourcePool()
    return _pool