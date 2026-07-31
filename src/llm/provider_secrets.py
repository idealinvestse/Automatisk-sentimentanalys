"""Load/store multi-provider LLM API keys (env + gitignored key files).

Never commit real keys. Resolution order per provider:
  1. Explicit override argument
  2. First non-empty env var from provider env_keys
  3. First readable configs/*.key (or AppData Sentimentanalys/secrets/)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fallback defaults when configs/llm_providers.yaml is unavailable
_PROVIDER_KEY_SPEC: dict[str, dict[str, Any]] = {
    "openrouter": {
        "env_keys": ["OPENROUTER_API_KEY"],
        "key_files": ["configs/openrouter.key", "openrouter.key", "OPENROUTER_API_KEY.txt"],
    },
    "mistral": {
        "env_keys": ["MISTRAL_API_KEY"],
        "key_files": ["configs/mistral.key"],
    },
    "nvidia": {
        "env_keys": ["NVIDIA_API_KEY", "NGC_API_KEY", "NVIDIA_NIM_API_KEY"],
        "key_files": ["configs/nvidia.key"],
    },
    "cerebras": {
        "env_keys": ["CEREBRAS_API_KEY"],
        "key_files": ["configs/cerebras.key"],
    },
    "groq": {
        "env_keys": ["GROQ_API_KEY"],
        "key_files": ["configs/groq.key"],
    },
}


def _appdata_secrets_dir() -> Path | None:
    appdata = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if not appdata:
        # Windows default
        home = Path.home()
        candidate = home / "AppData" / "Roaming" / "Sentimentanalys" / "secrets"
        return candidate if candidate.parent.exists() or True else None
    return Path(appdata) / "Sentimentanalys" / "secrets"


def _read_key_file(path: Path) -> str | None:
    try:
        if path.is_file():
            raw = path.read_text(encoding="utf-8").strip().lstrip("\ufeff").strip()
            if raw and not raw.lower().startswith("your-") and "example" not in raw.lower():
                return raw
    except OSError as exc:
        logger.debug("Could not read key file %s: %s", path, exc)
    return None


def load_provider_config(path: str | Path = "configs/llm_providers.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"providers": _PROVIDER_KEY_SPEC}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {"providers": _PROVIDER_KEY_SPEC}
        return data
    except Exception as exc:
        logger.warning("Failed to load %s: %s", p, exc)
        return {"providers": _PROVIDER_KEY_SPEC}


def get_provider_api_key(
    provider: str,
    override: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Resolve API key for a named provider."""
    if override and override.strip():
        return override.strip()

    cfg = config or load_provider_config()
    providers = cfg.get("providers") or _PROVIDER_KEY_SPEC
    spec = providers.get(provider) or _PROVIDER_KEY_SPEC.get(provider) or {}
    for env_name in spec.get("env_keys") or []:
        val = os.getenv(env_name)
        if val and val.strip():
            return val.strip()

    key_files = list(spec.get("key_files") or [])
    # Always also check AppData secrets/<basename>
    secrets_dir = _appdata_secrets_dir()
    candidates: list[Path] = []
    for rel in key_files:
        candidates.append(Path(rel))
        if secrets_dir is not None:
            candidates.append(secrets_dir / Path(rel).name)

    for path in candidates:
        key = _read_key_file(path)
        if key:
            # Mirror into env for child libs (openai SDK etc.)
            env_keys = spec.get("env_keys") or []
            if env_keys and not os.getenv(env_keys[0]):
                os.environ[env_keys[0]] = key
            return key
    return None


def list_configured_providers(config: dict[str, Any] | None = None) -> dict[str, bool]:
    """Return {provider: has_key} for enabled providers."""
    cfg = config or load_provider_config()
    providers = cfg.get("providers") or {}
    out: dict[str, bool] = {}
    for name, spec in providers.items():
        if isinstance(spec, dict) and spec.get("enabled", True) is False:
            out[name] = False
            continue
        out[name] = bool(get_provider_api_key(name, config=cfg))
    return out


def save_provider_key(provider: str, key: str, *, also_env: bool = True) -> Path:
    """Persist key to configs/<provider>.key (gitignored). Returns path written."""
    key = key.strip()
    if not key:
        raise ValueError("empty key")
    path = Path("configs") / f"{provider}.key"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8")
    secrets_dir = _appdata_secrets_dir()
    if secrets_dir is not None:
        try:
            secrets_dir.mkdir(parents=True, exist_ok=True)
            (secrets_dir / path.name).write_text(key + "\n", encoding="utf-8")
        except OSError:
            pass
    if also_env:
        cfg = load_provider_config()
        spec = (cfg.get("providers") or {}).get(provider) or _PROVIDER_KEY_SPEC.get(provider) or {}
        env_keys = spec.get("env_keys") or [f"{provider.upper()}_API_KEY"]
        os.environ[env_keys[0]] = key
    logger.info("Saved API key for provider=%s → %s (len=%d)", provider, path, len(key))
    return path
