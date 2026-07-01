"""Tests for transcription_runtime helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.archive.nicegui_dashboard.services.transcription_runtime import (
    DEFAULT_API_RETRIES,
    DEFAULT_LOCAL_TIMEOUT_S,
    api_retry_count,
    build_local_transcribe_kwargs,
    load_hotwords_from_file,
    local_timeout_seconds,
    resolve_hotwords,
    validate_audio_file,
    validate_upload_bytes,
)


def test_validate_upload_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="tom"):
        validate_upload_bytes(b"", "test.wav")


def test_validate_upload_bytes_rejects_oversized() -> None:
    with pytest.raises(ValueError, match="för stor"):
        validate_upload_bytes(b"x" * (500 * 1024 * 1024 + 1), "big.wav")


def test_validate_audio_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finns inte"):
        validate_audio_file(tmp_path / "missing.wav")


def test_validate_audio_file_too_small(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.wav"
    tiny.write_bytes(b"x" * 10)
    with pytest.raises(ValueError, match="för liten"):
        validate_audio_file(tiny)


def test_validate_audio_file_ok(tmp_path: Path) -> None:
    ok = tmp_path / "ok.wav"
    ok.write_bytes(b"x" * 256)
    validate_audio_file(ok)


def test_load_hotwords_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "hotwords.txt"
    path.write_text("# comment\n\nHej\n\nVälkommen\n", encoding="utf-8")
    words = load_hotwords_from_file(path)
    assert words == ["Hej", "Välkommen"]


def test_resolve_hotwords_merges_inline_and_file(tmp_path: Path, monkeypatch) -> None:
    hotwords_file = tmp_path / "hw.txt"
    hotwords_file.write_text("Agent\n", encoding="utf-8")
    settings = {
        "use_hotwords_file": True,
        "hotwords_file": str(hotwords_file),
        "hotwords": "Kund, agent",
    }
    merged = resolve_hotwords(settings)
    assert merged is not None
    assert "Agent" in merged
    assert "Kund" in merged
    assert len(merged) == 2


def test_build_local_transcribe_kwargs_includes_hotwords(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x" * 256)
    settings = {
        "backend": "faster",
        "language": "sv",
        "preprocess": True,
        "use_hotwords_file": False,
        "hotwords": "Testord",
    }
    backend, kwargs = build_local_transcribe_kwargs(settings, audio)
    assert backend == "faster"
    assert kwargs["audio_path"] == str(audio)
    assert kwargs["hotwords"] == ["Testord"]


def test_timeout_and_retry_defaults() -> None:
    assert local_timeout_seconds({}) == DEFAULT_LOCAL_TIMEOUT_S
    assert api_retry_count({}) == DEFAULT_API_RETRIES
    assert api_retry_count({"api_retries": 0}) == 1
