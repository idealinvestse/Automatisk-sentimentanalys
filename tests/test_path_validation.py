"""Tests for API path validation sandbox (Fas 2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.api.path_validation import (
    resolve_and_validate_audio_paths,
    validate_audio_path,
    validate_batch_audio_input,
    validate_directory_path,
    validate_lexicon_path,
    validate_resolved_audio_paths,
    validate_state_file_path,
)
from src.api.settings import get_api_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_api_settings.cache_clear()
    yield
    get_api_settings.cache_clear()


@pytest.fixture
def media_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    media = tmp_path / "media"
    media.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    monkeypatch.setenv("API_CACHE_DIR", str(cache))
    monkeypatch.setenv("API_STATE_DIR", str(state))
    get_api_settings.cache_clear()
    return media


class TestValidateAudioPath:
    def test_accepts_file_under_media_root(self, media_sandbox: Path) -> None:
        audio = media_sandbox / "call.wav"
        audio.write_bytes(b"RIFF")
        resolved = validate_audio_path(str(audio))
        assert resolved == str(audio.resolve())

    def test_rejects_missing_file(self, media_sandbox: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            validate_audio_path(str(media_sandbox / "missing.wav"))

    def test_rejects_path_outside_media_root(self, media_sandbox: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"RIFF")
        with pytest.raises(ValueError, match="API_MEDIA_ROOT"):
            validate_audio_path(str(outside))


class TestValidateDirectoryPath:
    def test_accepts_directory_under_media_root(self, media_sandbox: Path) -> None:
        sub = media_sandbox / "batch"
        sub.mkdir()
        resolved = validate_directory_path(str(sub))
        assert resolved == str(sub.resolve())

    def test_rejects_missing_directory(self, media_sandbox: Path) -> None:
        with pytest.raises(ValueError, match="Directory not found"):
            validate_directory_path(str(media_sandbox / "missing"))


class TestValidateBatchAudioInput:
    def test_glob_pattern_parent_must_be_in_sandbox(self, media_sandbox: Path) -> None:
        sub = media_sandbox / "calls"
        sub.mkdir()
        pattern = str(sub / "*.wav")
        assert validate_batch_audio_input(pattern) == pattern

    def test_glob_outside_sandbox_rejected(self, tmp_path: Path, media_sandbox: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        with pytest.raises(ValueError, match="API_MEDIA_ROOT"):
            validate_batch_audio_input(str(outside / "*.wav"))

    def test_file_input_delegates_to_audio_validator(self, media_sandbox: Path) -> None:
        audio = media_sandbox / "one.wav"
        audio.write_bytes(b"RIFF")
        assert validate_batch_audio_input(str(audio)) == str(audio)

    def test_directory_input_delegates_to_directory_validator(self, media_sandbox: Path) -> None:
        sub = media_sandbox / "dir"
        sub.mkdir()
        assert validate_batch_audio_input(str(sub)) == str(sub)

    def test_unknown_path_under_sandbox_returns_as_is(self, media_sandbox: Path) -> None:
        ghost = str(media_sandbox / "ghost.wav")
        assert validate_batch_audio_input(ghost) == ghost


class TestValidateResolvedAudioPaths:
    def test_validates_each_path(self, media_sandbox: Path) -> None:
        a = media_sandbox / "a.wav"
        b = media_sandbox / "b.wav"
        a.write_bytes(b"A")
        b.write_bytes(b"B")
        out = validate_resolved_audio_paths([str(a), str(b)])
        assert len(out) == 2


class TestValidateLexiconPath:
    def test_accepts_lexicon_under_allowed_roots(
        self, media_sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = Path(get_api_settings().cache_dir)
        lex = cache / "lexicon.yaml"
        lex.write_text("words: []\n", encoding="utf-8")
        resolved = validate_lexicon_path(str(lex))
        assert resolved == str(lex.resolve())

    def test_rejects_lexicon_outside_roots(self, tmp_path: Path) -> None:
        outside = tmp_path / "lexicon.yaml"
        outside.write_text("words: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="allowed data"):
            validate_lexicon_path(str(outside))


class TestValidateStateFilePath:
    def test_accepts_path_under_state_dir(self, media_sandbox: Path) -> None:
        state = Path(get_api_settings().state_dir)
        target = state / "scan_state.json"
        resolved = validate_state_file_path(str(target))
        assert resolved == str(target.resolve())

    def test_rejects_state_path_outside_state_dir(self, tmp_path: Path) -> None:
        outside = tmp_path / "state.json"
        with pytest.raises(ValueError, match="allowed state"):
            validate_state_file_path(str(outside))


class TestNoMediaRoot:
    def test_without_media_root_allows_any_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
        get_api_settings.cache_clear()
        audio = tmp_path / "free.wav"
        audio.write_bytes(b"RIFF")
        resolved = validate_audio_path(str(audio))
        assert resolved == str(audio.resolve())


class TestResolveAndValidateAudioPaths:
    def test_resolves_and_validates_under_sandbox(self, media_sandbox: Path) -> None:
        audio = media_sandbox / "resolved.wav"
        audio.write_bytes(b"RIFF")
        with patch("src.core.audio.resolve_audio_paths", return_value=[str(audio)]):
            out = resolve_and_validate_audio_paths(directory=str(media_sandbox))
        assert out == [str(audio.resolve())]
