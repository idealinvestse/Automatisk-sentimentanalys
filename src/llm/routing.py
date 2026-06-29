"""Cost/quality model routing via OpenRouter catalog."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "fast": "mistralai/mistral-small-3.1-24b-instruct",
    "balanced": "mistralai/mistral-medium-3.5",
    "deep": "mistralai/mistral-large-2512",
}


class RoutingTier(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


def select_model(
    tier: RoutingTier | str = RoutingTier.BALANCED,
    segment_count: int = 0,
    *,
    deep_analysis: bool = False,
    catalog_path: Path | str = Path("data/openrouter_models_catalog.json"),
    budget_usd: float | None = None,
) -> str:
    """Pick an OpenRouter model slug based on tier, call length, and flags."""
    if isinstance(tier, str):
        tier = RoutingTier(tier.lower())

    if deep_analysis or segment_count >= 20:
        tier = RoutingTier.DEEP
    elif segment_count < 6:
        tier = RoutingTier.FAST

    if budget_usd is not None and budget_usd < 0.02:
        tier = RoutingTier.FAST
        logger.debug("Budget %.4f USD → forcing FAST tier", budget_usd)

    default = DEFAULT_MODELS[tier.value]
    path = Path(catalog_path)
    if not path.is_file():
        logger.debug("No model catalog at %s; using default %s", path, default)
        return default

    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        models = catalog.get("models") or catalog.get("data") or []
        ids = {m.get("id") for m in models if isinstance(m, dict)}
        if default in ids:
            return default
        # Fallback: first mistral model in tier-appropriate list
        for candidate in DEFAULT_MODELS.values():
            if candidate in ids:
                return candidate
    except Exception as exc:
        logger.warning("Failed to read model catalog: %s", exc)

    return default
