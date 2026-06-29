"""Windows Credential Manager + persistent user_data file storage for API keys."""

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

SECRET_FILE_NAMES = {
    "openrouter": "openrouter.key",
    "huggingface": "hf_token.key",
    "groq": "groq.key",
}

SecretKind = Literal["openrouter", "huggingface", "groq"]

_OPENROUTER_FILE_CANDIDATES = (
    Path("configs/openrouter.key"),
    Path("OPENROUTER_API_KEY.txt"),
)


class SecretStoreError(RuntimeError):
    """Raised when a secret could not be persisted."""


def _keyring_available() -> bool:
    try:
        import keyring  # noqa: F401

        return True
    except ImportError:
        return False


def user_secrets_dir(app_root: Path | None = None) -> Path:
    """Persistent secrets directory under user data (roaming or portable)."""
    from .user_config import load_user_config

    cfg = load_user_config(app_root)
    return cfg.resolved_user_data_dir() / "secrets"


def user_secret_file(kind: SecretKind, app_root: Path | None = None) -> Path:
    return user_secrets_dir(app_root) / SECRET_FILE_NAMES[kind]


def _read_user_secret_file(kind: SecretKind, app_root: Path | None = None) -> str | None:
    path = user_secret_file(kind, app_root)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            return text
    return None


def _write_user_secret_file(kind: SecretKind, value: str, app_root: Path | None = None) -> Path:
    path = user_secret_file(kind, app_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")
    return path


def _delete_user_secret_file(kind: SecretKind, app_root: Path | None = None) -> None:
    path = user_secret_file(kind, app_root)
    if path.is_file():
        path.unlink()


def _has_keyring_secret(env_name: str) -> bool:
    if sys.platform != "win32" or not _keyring_available():
        return False
    import keyring

    stored = keyring.get_password(SERVICE_NAME, env_name)
    return bool(stored and stored.strip())


def _has_repo_openrouter_key(app_root: Path | None = None) -> bool:
    root = app_root or Path.cwd()
    for rel in _OPENROUTER_FILE_CANDIDATES:
        path = root / rel
        if not path.is_file():
            continue
        line = path.read_text(encoding="utf-8").strip().splitlines()
        if line and not line[0].startswith("sk-or-REPLACE"):
            return True
    return False


def _resolve_secret_source(kind: SecretKind, env_name: str, app_root: Path | None = None) -> str:
    if os.environ.get(env_name, "").strip():
        return "env"
    if _has_keyring_secret(env_name):
        return "keyring"
    if _read_user_secret_file(kind, app_root):
        return "user_file"
    if kind == "openrouter" and _has_repo_openrouter_key(app_root):
        return "repo_file"
    return "none"


def get_secret(kind: SecretKind, app_root: Path | None = None) -> str | None:
    """Resolve secret: env → keyring → user_data file → repo dev files (openrouter)."""
    env_name = SECRET_NAMES[kind]
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return env_val

    if sys.platform == "win32" and _keyring_available():
        import keyring

        stored = keyring.get_password(SERVICE_NAME, env_name)
        if stored and stored.strip():
            return stored.strip()

    file_val = _read_user_secret_file(kind, app_root)
    if file_val:
        return file_val

    if kind == "openrouter":
        from ..llm.openrouter_client import get_openrouter_api_key

        return get_openrouter_api_key()

    if kind == "huggingface":
        return os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip() or None

    return None


def set_secret(
    kind: SecretKind,
    value: str,
    *,
    use_credential_manager: bool = True,
    persist_user_file: bool = True,
    app_root: Path | None = None,
) -> list[str]:
    """Persist secret; returns list of storage backends used (keyring, user_file, env)."""
    value = value.strip()
    env_name = SECRET_NAMES[kind]
    if not value:
        delete_secret(kind, app_root=app_root)
        return []

    stored: list[str] = []
    keyring_error: str | None = None

    if use_credential_manager and sys.platform == "win32" and _keyring_available():
        import keyring

        try:
            keyring.set_password(SERVICE_NAME, env_name, value)
            stored.append("keyring")
            logger.info("Stored %s in Windows Credential Manager", env_name)
        except Exception as exc:
            keyring_error = str(exc)
            logger.warning("Credential Manager store failed for %s: %s", env_name, exc)

    if persist_user_file:
        path = _write_user_secret_file(kind, value, app_root)
        stored.append("user_file")
        logger.info("Stored %s in %s", env_name, path)

    os.environ[env_name] = value
    if kind == "huggingface":
        os.environ["HUGGINGFACE_HUB_TOKEN"] = value
    if "env" not in stored:
        stored.append("env")

    if not stored or (stored == ["env"] and keyring_error and not persist_user_file):
        raise SecretStoreError(
            f"Kunde inte spara {kind}-nyckel permanent."
            + (f" Credential Manager: {keyring_error}" if keyring_error else "")
        )
    return stored


def delete_secret(kind: SecretKind, app_root: Path | None = None) -> None:
    env_name = SECRET_NAMES[kind]
    os.environ.pop(env_name, None)
    if kind == "huggingface":
        os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    _delete_user_secret_file(kind, app_root)
    if sys.platform == "win32" and _keyring_available():
        import keyring

        with contextlib.suppress(keyring.errors.PasswordDeleteError):
            keyring.delete_password(SERVICE_NAME, env_name)


def secret_status(app_root: Path | None = None) -> dict[str, dict[str, bool | str]]:
    """Summary for UI (never returns secret values)."""
    out: dict[str, dict[str, bool | str]] = {}
    for kind, env_name in SECRET_NAMES.items():
        val = get_secret(kind, app_root)  # type: ignore[arg-type]
        out[kind] = {
            "configured": bool(val),
            "source": _resolve_secret_source(kind, env_name, app_root),  # type: ignore[arg-type]
        }
    return out


def apply_secrets_to_env(app_root: Path | None = None) -> dict[str, str]:
    """Inject resolved secrets into os.environ for subprocess spawning."""
    applied: dict[str, str] = {}
    for kind, env_name in SECRET_NAMES.items():
        val = get_secret(kind, app_root)  # type: ignore[arg-type]
        if val:
            os.environ[env_name] = val
            applied[env_name] = val
            if kind == "huggingface":
                os.environ["HUGGINGFACE_HUB_TOKEN"] = val
    return applied
