"""Multi-provider model catalog scanner.

Extends the original OpenRouter-only scanner to Mistral / NVIDIA / Cerebras / Groq.
Each provider catalog is saved under data/model_catalogs/<provider>.json with a unified
index at data/model_catalogs/index.json.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .provider_secrets import get_provider_api_key, load_provider_config

logger = logging.getLogger(__name__)

# Keep backward-compatible import path for get_openrouter_api_key used by older tests
try:
    from .openrouter_client import get_openrouter_api_key
except Exception:  # pragma: no cover

    def get_openrouter_api_key(override=None):  # type: ignore
        import os

        return override or os.getenv("OPENROUTER_API_KEY")


def _catalog_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_provider_config()
    rel = (cfg.get("catalog") or {}).get("dir") or "data/model_catalogs"
    path = Path(rel)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_model_entry(raw: dict[str, Any], *, provider: str) -> dict[str, Any]:
    """Normalize heterogeneous provider model payloads into a common shape."""
    mid = raw.get("id") or raw.get("name") or raw.get("model")
    pricing = raw.get("pricing") or {}
    # OpenRouter uses prompt/completion strings; others often omit pricing
    prompt = pricing.get("prompt") if isinstance(pricing, dict) else None
    completion = pricing.get("completion") if isinstance(pricing, dict) else None
    try:
        prompt_f = float(prompt or 0.0)
    except (TypeError, ValueError):
        prompt_f = 0.0
    try:
        completion_f = float(completion or 0.0)
    except (TypeError, ValueError):
        completion_f = 0.0

    # Free detection heuristics
    is_free = False
    if isinstance(mid, str) and mid.endswith(":free"):
        is_free = True
    if prompt_f == 0.0 and completion_f == 0.0 and provider == "openrouter":
        # OpenRouter free models are $0; paid have non-zero. Native APIs often omit pricing.
        is_free = True
    if raw.get("is_free") is True:
        is_free = True

    context_length = raw.get("context_length") or raw.get("max_model_len")
    if context_length is None and isinstance(raw.get("architecture"), dict):
        context_length = raw["architecture"].get("max_completion_tokens")

    return {
        "id": mid,
        "name": raw.get("name") or mid,
        "description": (raw.get("description") or raw.get("owned_by") or "")[:700].strip(),
        "context_length": context_length,
        "pricing": {
            "prompt_per_token_usd": prompt_f,
            "completion_per_token_usd": completion_f,
            "prompt_per_million_usd": round(prompt_f * 1_000_000, 4),
            "completion_per_million_usd": round(completion_f * 1_000_000, 4),
        },
        "architecture": raw.get("architecture") or {},
        "top_provider": raw.get("top_provider") or {},
        "per_request_limits": raw.get("per_request_limits"),
        "owned_by": raw.get("owned_by"),
        "provider": provider,
        "is_free": is_free,
        "raw_keys": sorted(raw.keys()),
    }


def _http_get_json(url: str, headers: dict[str, str], timeout: float = 60.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))  # type: ignore[no-any-return]


def fetch_provider_models_catalog(
    provider: str,
    *,
    output_path: str | Path | None = None,
    api_key: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch models for one provider and save enriched catalog JSON."""
    cfg = config or load_provider_config()
    providers = cfg.get("providers") or {}
    spec = providers.get(provider)
    if not spec:
        raise ValueError(f"Unknown provider: {provider}")

    base = str(spec.get("base_url") or "").rstrip("/")
    models_path = str(spec.get("models_path") or "/models")
    url = base + models_path

    headers: dict[str, str] = {"Accept": "application/json"}
    key = api_key if api_key is not None else get_provider_api_key(provider, config=cfg)
    # Empty string means explicitly unauthenticated
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for hk, hv in (spec.get("headers_extra") or {}).items():
        headers[str(hk)] = str(hv)

    logger.info("[model-catalog] Scanning %s → %s", provider, url)
    try:
        raw = _http_get_json(url, headers=headers)
    except urllib.error.HTTPError as e:
        logger.error("%s models API error %s: %s", provider, e.code, e.reason)
        # Seed from curated lists when provider forbids model listing (e.g. Cerebras 403)
        curated = list(spec.get("curated_free") or []) + [
            v
            for v in ((spec.get("curated_sv") or {}) if isinstance(spec.get("curated_sv"), dict) else {}).values()
        ]
        if curated and e.code in {401, 403, 404}:
            logger.warning(
                "[model-catalog] %s listing forbidden (%s) — seeding curated catalog (%d models)",
                provider,
                e.code,
                len(curated),
            )
            catalog_models = []
            seen=set()
            for mid in curated:
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                catalog_models.append(
                    {
                        "id": mid,
                        "name": mid,
                        "description": f"Curated seed (live /models unavailable: HTTP {e.code})",
                        "context_length": None,
                        "pricing": {
                            "prompt_per_token_usd": 0.0,
                            "completion_per_token_usd": 0.0,
                            "prompt_per_million_usd": 0.0,
                            "completion_per_million_usd": 0.0,
                        },
                        "architecture": {},
                        "top_provider": {},
                        "per_request_limits": None,
                        "owned_by": provider,
                        "provider": provider,
                        "is_free": mid in set(spec.get("curated_free") or []),
                        "raw_keys": [],
                        "seeded": True,
                    }
                )
            catalog = {
                "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "provider": provider,
                "source": url,
                "count": len(catalog_models),
                "free_count": sum(1 for m in catalog_models if m.get("is_free")),
                "models": catalog_models,
                "seeded_from_curated": True,
                "http_error": e.code,
            }
            out = Path(output_path) if output_path else _catalog_dir(cfg) / f"{provider}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
            return catalog
        raise
    except Exception:
        logger.exception("Model catalog scan failed for %s", provider)
        raise

    # Normalize list extraction
    if isinstance(raw, dict):
        models_raw = raw.get("data") or raw.get("models") or raw.get("items") or []
    elif isinstance(raw, list):
        models_raw = raw
    else:
        models_raw = []

    catalog_models = [
        _normalize_model_entry(m, provider=provider)
        for m in models_raw
        if isinstance(m, dict) and (m.get("id") or m.get("name"))
    ]

    # Apply curated free flags from config when pricing missing
    curated_free = set(spec.get("curated_free") or [])
    for m in catalog_models:
        if m.get("id") in curated_free:
            m["is_free"] = True

    catalog = {
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider,
        "source": url,
        "count": len(catalog_models),
        "free_count": sum(1 for m in catalog_models if m.get("is_free")),
        "models": catalog_models,
    }

    out = Path(output_path) if output_path else _catalog_dir(cfg) / f"{provider}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[model-catalog] Saved %d models for %s → %s", len(catalog_models), provider, out)
    return catalog


def fetch_openrouter_models_catalog(
    output_path: str | Path = "data/openrouter_models_catalog.json",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible OpenRouter scanner (also writes multi-catalog copy)."""
    # When api_key is explicitly None, still allow unauthenticated public list —
    # but do not inject env key if caller patched get_openrouter_api_key to None.
    resolved = api_key if api_key is not None else get_openrouter_api_key()
    catalog = fetch_provider_models_catalog(
        "openrouter",
        output_path=output_path,
        api_key=resolved if resolved else "",
    )
    # Mirror into multi-catalog dir
    multi = _catalog_dir() / "openrouter.json"
    with contextlib.suppress(OSError):
        multi.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return catalog


def fetch_all_provider_catalogs(
    providers: list[str] | None = None,
    *,
    config: dict[str, Any] | None = None,
    skip_missing_keys: bool = True,
) -> dict[str, Any]:
    """Scan all (or selected) providers; update index.json."""
    cfg = config or load_provider_config()
    all_providers = list((cfg.get("providers") or {}).keys())
    targets = providers or all_providers

    results: dict[str, Any] = {"scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "providers": {}}
    for name in targets:
        spec = (cfg.get("providers") or {}).get(name) or {}
        if spec.get("enabled", True) is False:
            results["providers"][name] = {"skipped": True, "reason": "disabled"}
            continue
        key = get_provider_api_key(name, config=cfg)
        if not key and skip_missing_keys and name != "openrouter":
            # OpenRouter models list often works without auth
            results["providers"][name] = {"skipped": True, "reason": "missing_api_key"}
            logger.warning("[model-catalog] skip %s — no API key", name)
            continue
        try:
            cat = fetch_provider_models_catalog(name, api_key=key, config=cfg)
            results["providers"][name] = {
                "ok": True,
                "count": cat.get("count"),
                "free_count": cat.get("free_count"),
                "path": str(_catalog_dir(cfg) / f"{name}.json"),
            }
        except Exception as exc:
            logger.exception("scan failed for %s", name)
            results["providers"][name] = {"ok": False, "error": str(exc)}

    index_path = Path((cfg.get("catalog") or {}).get("index_file") or "data/model_catalogs/index.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    # Legacy openrouter path convenience
    openrouter_multi = _catalog_dir(cfg) / "openrouter.json"
    legacy = Path("data/openrouter_models_catalog.json")
    if openrouter_multi.is_file() and not legacy.is_file():
        with contextlib.suppress(OSError):
            legacy.write_text(openrouter_multi.read_text(encoding="utf-8"), encoding="utf-8")
    return results


def load_catalog(path: str | Path = "data/openrouter_models_catalog.json") -> dict[str, Any] | None:
    """Load saved catalog for UI pickers / cost lookup."""
    p = Path(path)
    default_legacy = Path("data/openrouter_models_catalog.json")
    if not p.exists() and p == default_legacy:
        # Only fall back for the canonical legacy path
        alt = _catalog_dir() / "openrouter.json"
        p = alt if alt.exists() else p
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("Failed to load catalog %s: %s", p, e)
        return None


def load_provider_catalog(provider: str, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cfg = config or load_provider_config()
    path = _catalog_dir(cfg) / f"{provider}.json"
    if provider == "openrouter" and not path.exists():
        return load_catalog("data/openrouter_models_catalog.json")
    return load_catalog(path)


def list_free_models(provider: str, config: dict[str, Any] | None = None) -> list[str]:
    """Return free model ids for provider (catalog + curated)."""
    cfg = config or load_provider_config()
    cat = load_provider_catalog(provider, cfg)
    ids: list[str] = []
    if cat:
        for m in cat.get("models") or []:
            if isinstance(m, dict) and m.get("is_free") and m.get("id"):
                ids.append(str(m["id"]))
    spec = (cfg.get("providers") or {}).get(provider) or {}
    for mid in spec.get("curated_free") or []:
        if mid not in ids:
            ids.append(mid)
    return ids


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        for p in args:
            fetch_provider_models_catalog(p)
    else:
        print(json.dumps(fetch_all_provider_catalogs(), indent=2))
