"""Shared audio file resolution utilities.

Consolidates path-resolution logic from the CLI and API layers into a single
implementation that handles explicit paths, directories, and glob patterns.
"""

from __future__ import annotations

import glob as _glob
import logging
import os

from .config import AUDIO_EXTS

logger = logging.getLogger(__name__)


def resolve_audio_paths(
    audio_paths: list[str] | None = None,
    directory: str | None = None,
    pattern: str | None = None,
    recursive: bool = True,
    limit: int | None = None,
) -> list[str]:
    """Resolve audio file paths from various input sources.

    Handles multiple specification methods:

    1. ``audio_paths`` – explicit file paths, directory paths, or glob patterns
       (mixed is fine).
    2. ``directory`` + optional ``pattern`` – scan a directory for audio files,
       optionally filtered by a glob pattern such as ``**/*.wav``.

    Args:
        audio_paths: List of file paths, directory paths, or glob patterns.
        directory: Base directory to scan (used with or without ``pattern``).
        pattern: Glob pattern relative to ``directory``, e.g. ``**/*.wav``.
            When omitted the entire directory is scanned for supported
            audio extensions.
        recursive: Whether to recurse into subdirectories (default ``True``).
        limit: Optional maximum number of files returned after sorting.

    Returns:
        Sorted, deduplicated list of absolute paths to valid audio files.
    """
    files: list[str] = []

    # --- Process explicit paths / globs ---
    for p in audio_paths or []:
        if any(ch in p for ch in ("*", "?", "[")):
            # Treat as glob pattern
            for m in _glob.glob(p, recursive=recursive):
                if os.path.isfile(m) and os.path.splitext(m)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.abspath(m))
        elif os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in AUDIO_EXTS:
                files.append(os.path.abspath(p))
            else:
                logger.warning("Skipping file with unsupported extension: %s", p)
        elif os.path.isdir(p):
            for root, _dirs, fnames in os.walk(p):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(os.path.join(root, fn)))
                if not recursive:
                    break
        else:
            # Last resort: try the string as a glob
            matched = _glob.glob(p, recursive=recursive)
            if matched:
                for m in matched:
                    if os.path.isfile(m) and os.path.splitext(m)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(m))
            else:
                logger.warning("Input path could not be resolved: %s", p)

    # --- Process directory + optional pattern ---
    if directory:
        if not os.path.isdir(directory):
            logger.warning("Directory not found: %s", directory)
        elif pattern:
            glob_pattern = os.path.join(directory, pattern)
            for m in _glob.glob(glob_pattern, recursive=recursive):
                if os.path.isfile(m) and os.path.splitext(m)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.abspath(m))
        else:
            for root, _dirs, fnames in os.walk(directory):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(os.path.join(root, fn)))
                if not recursive:
                    break

    # Deduplicate (preserve first-seen order) then sort for determinism
    seen: set[str] = set()
    deduped: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    deduped.sort()

    if limit is not None:
        deduped = deduped[:limit]

    return deduped
