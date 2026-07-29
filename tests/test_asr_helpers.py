"""Tests for API ASR helpers routed through AsrRouter."""

from __future__ import annotations

from unittest.mock import patch

from src.api.helpers import asr_kwargs_from, transcribe_helper
from src.api.schemas import TranscribeRequest
from src.core.models import Transcript
from src.transcription.base import resolve_model_name


def test_transcribe_helper_uses_router():
    fake = Transcript("m", "faster", "sv", 1.0, 0.1, provider="local")
    with patch("src.api.helpers.AsrRouter") as R:
        R.return_value.transcribe.return_value = fake
        d = transcribe_helper("a.wav", provider="local")
    R.return_value.transcribe.assert_called()
    assert d["provider"] == "local"


def test_asr_kwargs_from_includes_provider_defaults(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF")
    req = TranscribeRequest(audio_path=str(audio))
    kwargs = asr_kwargs_from(req, audio_path=str(audio))
    assert kwargs["provider"] == "local"
    assert kwargs["cloud_fallback_local"] is False


def test_transcribe_helper_default_provider_local():
    fake = Transcript("m", "faster", "sv", 1.0, 0.1, provider="local")
    with patch("src.api.helpers.AsrRouter") as R:
        R.return_value.transcribe.return_value = fake
        transcribe_helper("a.wav")
    call_kwargs = R.return_value.transcribe.call_args.kwargs
    assert call_kwargs["provider"] == "local"


def test_kb_whisper_medium_alias():
    assert resolve_model_name("kb-whisper-medium") == "KBLab/kb-whisper-medium"
