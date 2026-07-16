"""Task 5: condition_on_previous_text + chunk retry/warnings."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from src.transcription.faster_whisper import FasterWhisperTranscriber


def _mock_segment(text: str = "Hej") -> MagicMock:
    seg = MagicMock()
    seg.start = 0.0
    seg.end = 5.0
    seg.text = text
    seg.words = []
    return seg


def _make_transcriber(mock_model: MagicMock) -> FasterWhisperTranscriber:
    with (
        patch("src.transcription.faster_whisper.WhisperModel", return_value=mock_model),
        patch("src.transcription.faster_whisper._HAS_FASTER", True),
    ):
        t = FasterWhisperTranscriber(model_name="kb-whisper-large", device="cpu")
    t._model = mock_model
    return t


def _patch_decode_audio(audio: np.ndarray):
    """Inject a fake faster_whisper.audio module (package may be absent in CI)."""
    mock_audio_mod = MagicMock()
    mock_audio_mod.decode_audio = MagicMock(return_value=audio)
    mock_fw = MagicMock()
    mock_fw.audio = mock_audio_mod
    return patch.dict(
        sys.modules,
        {"faster_whisper": mock_fw, "faster_whisper.audio": mock_audio_mod},
    )


def test_transcribe_passes_condition_on_previous_text_false():
    mock_model = MagicMock()
    mock_info = MagicMock(duration=5.0)
    mock_model.transcribe.return_value = ([_mock_segment()], mock_info)

    transcriber = _make_transcriber(mock_model)
    transcriber.transcribe("test.wav", chunk_length_s=0, vad=False, word_timestamps=False)

    _, kwargs = mock_model.transcribe.call_args
    assert kwargs.get("condition_on_previous_text") is False


def test_transcribe_condition_on_previous_text_overridable():
    mock_model = MagicMock()
    mock_info = MagicMock(duration=5.0)
    mock_model.transcribe.return_value = ([_mock_segment()], mock_info)

    transcriber = _make_transcriber(mock_model)
    transcriber.transcribe(
        "test.wav",
        chunk_length_s=0,
        vad=False,
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    _, kwargs = mock_model.transcribe.call_args
    assert kwargs.get("condition_on_previous_text") is True


def test_chunk_failure_retries_then_warns():
    """Chunk 1 fails all 3 attempts; chunk 2 succeeds; warning attached."""
    mock_model = MagicMock()
    ndarray_calls = [0]

    def transcribe_side_effect(audio, **kwargs):
        if isinstance(audio, str):
            return ([_mock_segment("full")], MagicMock(duration=12.0))
        ndarray_calls[0] += 1
        if ndarray_calls[0] <= 3:
            raise RuntimeError("chunk decode failed")
        return ([_mock_segment("chunk two ok")], MagicMock())

    mock_model.transcribe.side_effect = transcribe_side_effect

    # 10s chunks, 5s overlap -> step 5s; 170k samples (~10.6s) yields 2 chunks
    audio = np.zeros(170_000, dtype=np.float32)

    transcriber = _make_transcriber(mock_model)
    with _patch_decode_audio(audio):
        result = transcriber.transcribe(
            "test.wav",
            chunk_length_s=10,
            vad=False,
            word_timestamps=False,
        )

    assert any("chunk_failed:1" in w for w in result.warnings)
    assert mock_model.transcribe.call_count >= 4
    assert any(s.text == "chunk two ok" for s in result.segments)


def test_chunk_retry_succeeds_on_third_attempt():
    """Chunk 1 fails twice then succeeds on 3rd attempt — no chunk_failed warning."""
    mock_model = MagicMock()
    ndarray_calls = [0]

    def transcribe_side_effect(audio, **kwargs):
        if isinstance(audio, str):
            return ([_mock_segment("full")], MagicMock(duration=12.0))
        ndarray_calls[0] += 1
        if ndarray_calls[0] <= 2:
            raise RuntimeError("transient chunk error")
        return ([_mock_segment("recovered")], MagicMock())

    mock_model.transcribe.side_effect = transcribe_side_effect

    audio = np.zeros(170_000, dtype=np.float32)
    transcriber = _make_transcriber(mock_model)
    with _patch_decode_audio(audio):
        result = transcriber.transcribe(
            "test.wav",
            chunk_length_s=10,
            vad=False,
            word_timestamps=False,
        )

    assert not any("chunk_failed" in w for w in result.warnings)
    assert any(s.text == "recovered" for s in result.segments)
