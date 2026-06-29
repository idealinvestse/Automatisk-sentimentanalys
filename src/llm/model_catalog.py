"""Dynamic OpenRouter model catalog scanner.

Scans https://openrouter.ai/api/v1/models and saves structured catalog with:
- id, name, short description
- pricing (prompt/completion per token and per million)
- context_length, architecture, top_provider

Intended use:
- Launcher: `sentimentanalys scan-openrouter-models` or in setup wizard
- Dashboard: Refresh button in LLM / Model settings panel
- Future: dynamic model picker in profiles + cost estimator in meta

Saves to data/openrouter_models_catalog.json (gitignored recommended for large file)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    from .openrouter_client import get_openrouter_api_key
except Exception:  # fallback if circular

    def get_openrouter_api_key(override=None):
        import os

        return override or os.getenv("OPENROUTER_API_KEY")


logger = logging.getLogger(__name__)


def fetch_openrouter_models_catalog(
    output_path: str | Path = "data/openrouter_models_catalog.json",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch all models from OpenRouter and save enriched catalog."""
    import urllib.error
    import urllib.request

    url = "https://openrouter.ai/api/v1/models"
    headers = {}
    key = api_key or get_openrouter_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    logger.info("[model-catalog] Scanning OpenRouter for all available models...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        models_raw = raw.get("data", [])
        catalog_models: list[dict[str, Any]] = []

        for m in models_raw:
            pricing = m.get("pricing", {}) or {}
            model_entry = {
                "id": m.get("id"),
                "name": m.get("name"),
                "description": (m.get("description") or "")[:700].strip(),
                "context_length": m.get("context_length"),
                "pricing": {
                    "prompt_per_token_usd": float(pricing.get("prompt", 0.0) or 0.0),
                    "completion_per_token_usd": float(pricing.get("completion", 0.0) or 0.0),
                    "prompt_per_million_usd": round(
                        float(pricing.get("prompt", 0.0) or 0.0) * 1_000_000, 4
                    ),
                    "completion_per_million_usd": round(
                        float(pricing.get("completion", 0.0) or 0.0) * 1_000_000, 4
                    ),
                },
                "architecture": m.get("architecture", {}),
                "top_provider": m.get("top_provider", {}),
                "per_request_limits": m.get("per_request_limits"),
            }
            catalog_models.append(model_entry)

        catalog = {
            "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": url,
            "count": len(catalog_models),
            "models": catalog_models,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)

        logger.info(f"[model-catalog] ✅ Saved {len(catalog_models)} models to {out}")
        return catalog

    except urllib.error.HTTPError as e:
        logger.error(f"OpenRouter API error {e.code}: {e.reason}")
        raise
    except Exception:
        logger.exception("Model catalog scan failed")
        raise


def load_catalog(path: str | Path = "data/openrouter_models_catalog.json") -> dict[str, Any] | None:
    """Load saved catalog for UI pickers / cost lookup."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load catalog {p}: {e}")
        return None


if __name__ == "__main__":
    # Quick manual run: python -m src.llm.model_catalog
    fetch_openrouter_models_catalog()
