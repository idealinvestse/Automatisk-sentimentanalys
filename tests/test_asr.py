"""Tests for the transcription module (unit tests with mocked model loading)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.config import KB_REVISIONS
from src.core.device import normalize_device_for_asr
from src.transcription.base import resolve_model_name
from src.transcription.factory import get_transcriber


class TestModelResolution:
    def test_alias_kb_whisper(self):
        assert resolve_model_name("kb-whisper-large") == "KBLab/kb-whisper-large"

    def test_alias_large_v3(self):
        assert resolve_model_name("large-v3") == "openai/whisper-large-v3"

    def test_full_name_passthrough(self):
        assert resolve_model_name("KBLab/kb-whisper-large") == "KBLab/kb-whisper-large"

    def test_empty_string(self):
        assert resolve_model_name("") == "KBLab/kb-whisper-large"

    def test_whitespace(self):
        assert resolve_model_name("  kb-whisper-large  ") == "KBLab/kb-whisper-large"


class TestDeviceNormalization:
    def test_auto(self):
        device, idx = normalize_device_for_asr("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_cpu(self):
        device, idx = normalize_device_for_asr("cpu")
        assert device == "cpu"
        assert idx is None

    def test_cuda_with_index(self):
        with (
            patch("src.core.device.torch.cuda.is_available", return_value=True),
            patch("src.core.device.torch.cuda.device_count", return_value=2),
        ):
            device, idx = normalize_device_for_asr("cuda:1")
        assert device == "cuda"
        assert idx == 1

    def test_cuda_invalid_index_fallback(self):
        with (
            patch("src.core.device.torch.cuda.is_available", return_value=True),
            patch("src.core.device.torch.cuda.device_count", return_value=1),
        ):
            device, idx = normalize_device_for_asr("cuda:5")
        assert device == "cuda"
        assert idx == 0

    def test_mps(self):
        device, idx = normalize_device_for_asr("mps")
        assert device == "mps"
        assert idx is None

    def test_none(self):
        device, idx = normalize_device_for_asr(None)
        assert device in ("cpu", "cuda", "mps")


class TestKBRevisions:
    def test_known_revisions(self):
        assert "standard" in KB_REVISIONS
        assert "strict" in KB_REVISIONS
        assert "subtitle" in KB_REVISIONS
        assert len(KB_REVISIONS) == 3


class TestTranscribeMocked:
    """Test the transcribe function with mocked backends."""

    def test_mocked_faster_backend(self):
        """Test faster-whisper backend with mocked model."""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.duration = 10.0

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "Det här är ett test."
        mock_segment.words = []

        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with (
            patch("src.transcription.faster_whisper.WhisperModel", return_value=mock_model),
            patch("src.transcription.faster_whisper._HAS_FASTER", True),
        ):
            transcriber = get_transcriber(
                backend="faster", model_name="kb-whisper-large", device="cpu"
            )
            result = transcriber.transcribe(
                audio_path="test.wav",
                language="sv",
                beam_size=5,
                vad=False,
                word_timestamps=False,
                revision="strict",
            )

        assert result.model == "KBLab/kb-whisper-large"
        assert result.backend == "faster"
        assert result.revision == "strict"
        assert result.duration == 10.0
        assert len(result.segments) == 1
        assert result.segments[0].text == "Det här är ett test."

    def test_faster_backend_not_available(self):
        """Test that missing faster-whisper raises ImportError."""
        with (
            patch("src.transcription.faster_whisper._HAS_FASTER", False),
            pytest.raises(ImportError, match="faster-whisper is not installed"),
        ):
            get_transcriber(backend="faster", device="cpu")

    def test_transformers_backend_mocked(self):
        """Test transformers backend with mocked pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "chunks": [
                {
                    "timestamp": (0.0, 5.0),
                    "text": "Hej världen",
                    "timestamps": [],
                }
            ]
        }

        with patch("src.transcription.transformers.pipeline", return_value=mock_pipeline):
            transcriber = get_transcriber(
                backend="transformers", model_name="kb-whisper-large", device="cpu"
            )
            result = transcriber.transcribe(
                audio_path="test.wav",
                language="sv",
                word_timestamps=False,
            )

        assert result.backend == "transformers"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hej världen"

    def test_unknown_revision_warns(self):
        """Test that unknown revision doesn't break things."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {"text": "test"}

        with patch("src.transcription.transformers.pipeline", return_value=mock_pipeline):
            transcriber = get_transcriber(backend="transformers", device="cpu")
            result = transcriber.transcribe(
                audio_path="test.wav",
                revision="unknown_revision",
            )

        # Should still work, revision just ignored
        assert result.revision is None
