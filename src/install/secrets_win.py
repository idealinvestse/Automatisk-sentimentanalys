"""Windows Credential Manager integration for API keys."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

SERVICE_NAME = "Sentimentanalys"
SECRET_NAMES = {
    "openrouter": "OPENROUTER_API_KEY",
    "huggingface": "HF_TOKEN",
    "groq": "GROQ_API_KEY",
}

SecretKind = Literal["openrouter", "huggingface", "groq"]

_OPENROUTER_FILE_CANDIDATES = (
    Path("configs/openrouter.key"),
    Path("OPENROUTER_API_KEY.txt"),
)


def _keyring_available() -> bool:
    try:
        import keyring  # noqa: F401

        return True
    except ImportError:
        return False


def _has_keyring_secret(env_name: str) -> bool:
    if sys.platform != "win32" or not _keyring_available():
        return False
    import keyring

    stored = keyring.get_password(SERVICE_NAME, env_name)
    return bool(stored and stored.strip())


def _has_file_openrouter_key(app_root: Path | None = None) -> bool:
    root = app_root or Path.cwd()
    for rel in _OPENROUTER_FILE_CANDIDATES:
        path = root / rel
        if not path.is_file():
            continue
        line = path.read_text(encoding="utf-8").strip().splitlines()
        if line and not line[0].startswith("sk-or-REPLACE"):
            return True
    return False


def _resolve_secret_source(kind: SecretKind, env_name: str, val: str | None) -> str:
    if os.environ.get(env_name, "").strip():
        return "env"
    if _has_keyring_secret(env_name):
        return "keyring"
    if val and kind == "openrouter" and _has_file_openrouter_key():
        return "file"
    if val:
        return "unknown"
    return "none"


def get_secret(kind: SecretKind) -> str | None:
    """Resolve secret: env var first, then Credential Manager, then dev files."""
    env_name = SECRET_NAMES[kind]
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return env_val

    if sys.platform == "win32" and _keyring_available():
        import keyring

        stored = keyring.get_password(SERVICE_NAME, env_name)
        if stored and stored.strip():
            return stored.strip()

    if kind == "openrouter":
        from ..llm.openrouter_client import get_openrouter_api_key

        return get_openrouter_api_key()

    return os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip() or None


def set_secret(kind: SecretKind, value: str, *, use_credential_manager: bool = True) -> None:
    value = value.strip()
    env_name = SECRET_NAMES[kind]
    if not value:
        delete_secret(kind)
        return

    if use_credential_manager and sys.platform == "win32" and _keyring_available():
        import keyring

        keyring.set_password(SERVICE_NAME, env_name, value)
        logger.info("Stored %s in Windows Credential Manager", env_name)
        return

    os.environ[env_name] = value
    if kind == "huggingface":
        os.environ["HUGGINGFACE_HUB_TOKEN"] = value


def delete_secret(kind: SecretKind) -> None:
    env_name = SECRET_NAMES[kind]
    os.environ.pop(env_name, None)
    if kind == "huggingface":
        os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    if sys.platform == "win32" and _keyring_available():
        import keyring

        with contextlib.suppress(keyring.errors.PasswordDeleteError):
            keyring.delete_password(SERVICE_NAME, env_name)


def secret_status(app_root: Path | None = None) -> dict[str, dict[str, bool | str]]:
    """Summary for doctor / setup hub (never returns secret values)."""
    out: dict[str, dict[str, bool | str]] = {}
    for kind, env_name in SECRET_NAMES.items():
        val = get_secret(kind)  # type: ignore[arg-type]
        out[kind] = {
            "configured": bool(val),
            "source": _resolve_secret_source(kind, env_name, val),  # type: ignore[arg-type]
        }
    return out


def apply_secrets_to_env() -> dict[str, str]:
    """Inject resolved secrets into os.environ for subprocess spawning."""
    applied: dict[str, str] = {}
    for kind, env_name in SECRET_NAMES.items():
        val = get_secret(kind)  # type: ignore[arg-type]
        if val:
            os.environ[env_name] = val
            applied[env_name] = val
            if kind == "huggingface":
                os.environ["HUGGINGFACE_HUB_TOKEN"] = val
    return applied
