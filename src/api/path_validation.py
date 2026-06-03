"""Server-side path validation for API requests (Fas 2 — API_MEDIA_ROOT sandbox)."""

from __future__ import annotations

import os

from .settings import get_api_settings


def _resolve_under_media_root(path: str) -> str:
    resolved = os.path.realpath(path)
    settings = get_api_settings()
    if not settings.media_root:
        return resolved
    root = os.path.realpath(settings.media_root)
    try:
        common = os.path.commonpath([resolved, root])
    except ValueError:
        common = ""
    if common != root:
        raise ValueError(f"Path must be under API_MEDIA_ROOT ({root!r}): {path!r}")
    return resolved


def validate_audio_path(path: str) -> str:
    """Ensure audio file exists and is under optional media root."""
    resolved = _resolve_under_media_root(path)
    if not os.path.isfile(resolved):
        raise ValueError(f"Audio file not found on server: {path!r}")
    return resolved


def validate_directory_path(path: str) -> str:
    """Ensure directory exists and is under optional media root."""
    resolved = _resolve_under_media_root(path)
    if not os.path.isdir(resolved):
        raise ValueError(f"Directory not found on server: {path!r}")
    return resolved
