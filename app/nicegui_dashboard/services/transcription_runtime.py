"""Shared transcription helpers – validation, hotwords, kwargs, retries."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HOTWORDS_FILE = Path("configs/callcenter_hotwords.txt")
MAX_AUDIO_BYTES = 500 * 1024 * 1024  # 500 MB
MIN_AUDIO_BYTES = 128
DEFAULT_LOCAL_TIMEOUT_S = 900.0
DEFAULT_API_RETRIES = 2


def validate_audio_file(path: Path, *, max_bytes: int = MAX_AUDIO_BYTES) -> None:
    """Raise ValueError if the audio file is missing or unsuitable for ASR."""
    if not path.is_file():
        raise ValueError(f"Ljudfilen finns inte: {path}")
    size = path.stat().st_size
    if size < MIN_AUDIO_BYTES:
        raise ValueError(f"Ljudfilen är för liten ({size} B): {path.name}")
    if size > max_bytes:
        raise ValueError(
            f"Ljudfilen överskrider maxstorlek ({max_bytes // (1024 * 1024)} MB): {path.name}"
        )


def validate_upload_bytes(content: bytes, filename: str, *, max_bytes: int = MAX_AUDIO_BYTES) -> None:
    if not content:
        raise ValueError("Uppladdad fil är tom")
    if len(content) > max_bytes:
        raise ValueError(
            f"Filen {filename} är för stor (max {max_bytes // (1024 * 1024)} MB)"
        )


def load_hotwords_from_file(
    path: Path | str | None = None,
    *,
    max_words: int = 80,
) -> list[str]:
    """Load callcenter hotwords from the configured text file."""
    p = Path(path) if path else HOTWORDS_FILE
    if not p.is_file():
        return []
    words: list[str] = []
    try:
        with p.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                words.append(text)
                if len(words) >= max_words:
                    break
    except OSError as err:
        logger.warning("Could not read hotwords file %s: %s", p, err)
    return words


def resolve_hotwords(settings: dict[str, Any]) -> list[str] | None:
    """Merge inline hotwords with optional file-based callcenter terms."""
    merged: list[str] = []
    seen: set[str] = set()

    if settings.get("use_hotwords_file", True):
        file_path = settings.get("hotwords_file") or str(HOTWORDS_FILE)
        for word in load_hotwords_from_file(file_path):
            key = word.lower()
            if key not in seen:
                seen.add(key)
                merged.append(word)

    raw = settings.get("hotwords") or ""
    if raw:
        if isinstance(raw, list):
            parts = [str(w).strip() for w in raw if str(w).strip()]
        else:
            parts = [w.strip() for w in str(raw).replace(";", ",").split(",") if w.strip()]
        for word in parts:
            key = word.lower()
            if key not in seen:
                seen.add(key)
                merged.append(word)

    return merged or None


def build_local_transcribe_kwargs(
    settings: dict[str, Any],
    audio_path: str | Path,
    *,
    on_chunk_progress: Callable[[int, int], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build kwargs for transcriber.transcribe() from dashboard settings."""
    path = Path(audio_path)
    validate_audio_file(path)

    backend = settings.get("backend", "faster")
    kwargs: dict[str, Any] = {
        "audio_path": str(path),
        "language": settings.get("language", "sv"),
        "preprocess": bool(settings.get("preprocess", True)),
        "diarize": bool(settings.get("diarize", False)),
        "beam_size": int(settings.get("beam_size", 5)),
        "vad": bool(settings.get("vad", True)),
        "chunk_length_s": int(settings.get("chunk_length_s", 30)),
        "revision": settings.get("revision"),
        "num_speakers": settings.get("num_speakers"),
        "word_timestamps": bool(settings.get("word_timestamps", True)),
    }
    hotwords = resolve_hotwords(settings)
    if hotwords:
        kwargs["hotwords"] = hotwords
    if settings.get("initial_prompt"):
        kwargs["initial_prompt"] = settings.get("initial_prompt")
    if backend == "faster" and on_chunk_progress is not None:
        kwargs["on_chunk_progress"] = on_chunk_progress
    return backend, kwargs


def local_timeout_seconds(settings: dict[str, Any]) -> float:
    return float(settings.get("local_timeout_s", DEFAULT_LOCAL_TIMEOUT_S))


def api_retry_count(settings: dict[str, Any]) -> int:
    return max(1, int(settings.get("api_retries", DEFAULT_API_RETRIES)))