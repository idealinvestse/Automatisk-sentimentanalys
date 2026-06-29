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


def _resolve_under_allowed_roots(path: str, roots: list[str], *, label: str) -> str:
    """Ensure *path* resolves under at least one allowed root."""
    resolved = os.path.realpath(path)
    for root in roots:
        if not root:
            continue
        root_resolved = os.path.realpath(root)
        try:
            common = os.path.commonpath([resolved, root_resolved])
        except ValueError:
            continue
        if common == root_resolved:
            return resolved
    roots_display = ", ".join(repr(r) for r in roots if r)
    raise ValueError(f"Path must be under an allowed {label} directory ({roots_display}): {path!r}")


def _allowed_data_roots() -> list[str]:
    settings = get_api_settings()
    roots: list[str] = []
    if settings.media_root:
        roots.append(settings.media_root)
    roots.append(settings.cache_dir)
    roots.append(settings.state_dir)
    return roots


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


def validate_batch_audio_input(path: str) -> str:
    """Validate a batch path input (file, directory, or glob pattern)."""
    if any(ch in path for ch in ("*", "?", "[")):
        prefix = path.split("*")[0].split("?")[0].split("[")[0]
        parent = os.path.dirname(os.path.abspath(prefix)) or os.getcwd()
        _resolve_under_media_root(parent)
        return path
    if os.path.isfile(path):
        validate_audio_path(path)
        return path
    if os.path.isdir(path):
        validate_directory_path(path)
        return path
    _resolve_under_media_root(path)
    return path


def validate_resolved_audio_paths(paths: list[str]) -> list[str]:
    """Re-validate resolved audio files (catches symlink escapes)."""
    return [validate_audio_path(p) for p in paths]


def resolve_and_validate_audio_paths(
    *,
    audio_paths: list[str] | None = None,
    directory: str | None = None,
    pattern: str | None = None,
    recursive: bool = True,
    limit: int | None = None,
) -> list[str]:
    """Resolve audio files and validate each path under the media sandbox."""
    from ..core.audio import resolve_audio_paths

    resolved = resolve_audio_paths(
        audio_paths=audio_paths,
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        limit=limit,
    )
    return validate_resolved_audio_paths(resolved)


def validate_lexicon_path(path: str) -> str:
    """Ensure lexicon file exists and is under allowed data roots."""
    resolved = _resolve_under_allowed_roots(path, _allowed_data_roots(), label="data")
    if not os.path.isfile(resolved):
        raise ValueError(f"Lexicon file not found on server: {path!r}")
    return resolved


def validate_state_file_path(path: str) -> str:
    """Ensure state file path is under the configured state directory."""
    settings = get_api_settings()
    resolved = _resolve_under_allowed_roots(path, [settings.state_dir], label="state")
    return resolved
