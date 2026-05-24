"""Tests for ASR module (unit tests with mocked model loading)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.asr import (
    KB_REVISIONS,
    _normalize_device,
    _resolve_model_name,
    _to_transcript,
    transcribe,
)


class TestModelResolution:
    def test_alias_kb_whisper(self):
        assert _resolve_model_name("kb-whisper-large") == "KBLab/kb-whisper-large"

    def test_alias_large_v3(self):
        assert _resolve_model_name("large-v3") == "openai/whisper-large-v3"

    def test_full_name_passthrough(self):
        assert _resolve_model_name("KBLab/kb-whisper-large") == "KBLab/kb-whisper-large"

    def test_empty_string(self):
        assert _resolve_model_name("") == "KBLab/kb-whisper-large"

    def test_whitespace(self):
        assert _resolve_model_name("  kb-whisper-large  ") == "KBLab/kb-whisper-large"


class TestDeviceNormalization:
    def test_auto(self):
        device, idx = _normalize_device("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_cpu(self):
        device, idx = _normalize_device("cpu")
        assert device == "cpu"
        assert idx is None

    def test_cuda_with_index(self):
        device, idx = _normalize_device("cuda:1")
        assert device == "cuda"
        assert idx == 1

    def test_mps(self):
        device, idx = _normalize_device("mps")
        assert device == "mps"
        assert idx is None

    def test_none(self):
        device, idx = _normalize_device(None)
        assert device in ("cpu", "cuda", "mps")


class TestToTranscript:
    def test_basic(self):
        result = _to_transcript(
            segments=[{"text": "hello"}],
            model_name="test-model",
            backend="faster",
            language="sv",
            duration=10.0,
            proc_time=1.5,
            revision="strict",
        )
        assert result["model"] == "test-model"
        assert result["backend"] == "faster"
        assert result["language"] == "sv"
        assert result["duration"] == 10.0
        assert result["processing_time"] == 1.5
        assert result["revision"] == "strict"
        assert result["segments"] == [{"text": "hello"}]

    def test_no_revision(self):
        result = _to_transcript(
            segments=[],
            model_name="m",
            backend="b",
            language="sv",
            duration=None,
            proc_time=0.0,
        )
        assert "revision" not in result


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
            patch("src.asr.WhisperModel", return_value=mock_model),
            patch("src.asr._HAS_FASTER", True),
        ):
            result = transcribe(
                audio_path="test.wav",
                model="kb-whisper-large",
                backend="faster",
                device="cpu",
                language="sv",
                beam_size=5,
                vad=False,
                word_timestamps=False,
                revision="strict",
            )

        assert result["model"] == "KBLab/kb-whisper-large"
        assert result["backend"] == "faster"
        assert result["revision"] == "strict"
        assert result["duration"] == 10.0
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Det här är ett test."

    def test_faster_backend_not_available(self):
        """Test that missing faster-whisper raises RuntimeError."""
        with (
            patch("src.asr._HAS_FASTER", False),
            pytest.raises(RuntimeError, match="faster-whisper not installed"),
        ):
            transcribe(
                audio_path="test.wav",
                backend="faster",
                device="cpu",
            )

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

        with patch("src.asr.pipeline", return_value=mock_pipeline):
            result = transcribe(
                audio_path="test.wav",
                model="kb-whisper-large",
                backend="transformers",
                device="cpu",
                language="sv",
                word_timestamps=False,
            )

        assert result["backend"] == "transformers"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hej världen"

    def test_unknown_revision_warns(self):
        """Test that unknown revision doesn't break things."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {"text": "test"}

        with patch("src.asr.pipeline", return_value=mock_pipeline):
            result = transcribe(
                audio_path="test.wav",
                backend="transformers",
                device="cpu",
                revision="unknown_revision",
            )

        # Should still work, revision just ignored
        assert "revision" not in result
