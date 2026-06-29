"""Unit tests for API path validation sandbox helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.path_validation import (
    validate_audio_path,
    validate_batch_audio_input,
    validate_directory_path,
    validate_lexicon_path,
    validate_resolved_audio_paths,
    validate_state_file_path,
)
from src.api.settings import get_api_settings


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    monkeypatch.delenv("API_STATE_DIR", raising=False)
    monkeypatch.delenv("API_CACHE_DIR", raising=False)
    get_api_settings.cache_clear()


def test_validate_audio_path_existing_file(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    audio = media / "call.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    resolved = validate_audio_path(str(audio))
    assert resolved == str(audio.resolve())


def test_validate_audio_path_rejects_outside_media_root(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    with pytest.raises(ValueError, match="API_MEDIA_ROOT"):
        validate_audio_path(str(outside))


def test_validate_audio_path_missing_file(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    missing = media / "missing.wav"
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    with pytest.raises(ValueError, match="not found"):
        validate_audio_path(str(missing))


def test_validate_directory_path_existing(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    sub = media / "calls"
    sub.mkdir(parents=True)
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    resolved = validate_directory_path(str(sub))
    assert resolved == str(sub.resolve())


def test_validate_directory_path_missing(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    missing = media / "nope"
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    with pytest.raises(ValueError, match="Directory not found"):
        validate_directory_path(str(missing))


def test_validate_batch_audio_input_glob_pattern(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    sub = media / "calls"
    sub.mkdir(parents=True)
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    pattern = str(sub / "*.wav")
    assert validate_batch_audio_input(pattern) == pattern


def test_validate_batch_audio_input_rejects_glob_outside_root(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    with pytest.raises(ValueError, match="API_MEDIA_ROOT"):
        validate_batch_audio_input(str(tmp_path / ".." / "escape" / "*.wav"))


def test_validate_resolved_audio_paths(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    a = media / "a.wav"
    b = media / "b.wav"
    a.write_bytes(b"RIFF")
    b.write_bytes(b"RIFF")
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    resolved = validate_resolved_audio_paths([str(a), str(b)])
    assert len(resolved) == 2


def test_validate_lexicon_path_under_cache_dir(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    lex = cache_dir / "lexicon.yaml"
    lex.write_text("words: []", encoding="utf-8")
    monkeypatch.setenv("API_CACHE_DIR", str(cache_dir))
    get_api_settings.cache_clear()

    resolved = validate_lexicon_path(str(lex))
    assert resolved == str(lex.resolve())


def test_validate_lexicon_path_rejects_outside_roots(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("words: []", encoding="utf-8")
    monkeypatch.setenv("API_CACHE_DIR", str(cache_dir))
    get_api_settings.cache_clear()

    with pytest.raises(ValueError, match="allowed data"):
        validate_lexicon_path(str(outside))


def test_validate_state_file_path_under_state_dir(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_file = state_dir / "scan.json"
    monkeypatch.setenv("API_STATE_DIR", str(state_dir))
    get_api_settings.cache_clear()

    resolved = validate_state_file_path(str(state_file))
    assert resolved == str(state_file.resolve())


def test_validate_state_file_path_rejects_escape(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    outside = tmp_path / "escape.json"
    monkeypatch.setenv("API_STATE_DIR", str(state_dir))
    get_api_settings.cache_clear()

    with pytest.raises(ValueError, match="allowed state"):
        validate_state_file_path(str(outside))


def test_resolve_and_validate_audio_paths_delegates(tmp_path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    audio = media / "x.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()

    from src.api.path_validation import resolve_and_validate_audio_paths

    with patch(
        "src.core.audio.resolve_audio_paths",
        return_value=[str(audio)],
    ):
        result = resolve_and_validate_audio_paths(audio_paths=[str(audio)])
    assert result == [str(audio.resolve())]
