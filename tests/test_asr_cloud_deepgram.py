"""Unit tests for Deepgram cloud ASR adapter (no network in default CI)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.core.errors import TranscriptionError
from src.transcription.cloud_deepgram import DeepgramTranscriber, map_deepgram_response
from src.transcription.error_codes import AsrErrorCode
from src.transcription.router import AsrRouter

_FIXTURE = Path("tests/fixtures/deepgram_listen_response.json")


@pytest.fixture
def audio_wav(tmp_path: Path) -> str:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF")
    return str(path)


def test_map_deepgram_response_segments():
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    segs = map_deepgram_response(data)
    assert segs[0].text
    assert segs[0].start >= 0
    assert len(segs[0].words) == 2
    assert segs[0].words[0].word == "Hej"


def test_map_deepgram_response_low_confidence():
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for w in data["results"]["channels"][0]["alternatives"][0]["words"]:
        w["confidence"] = 0.4
    segs = map_deepgram_response(data)
    assert segs[0].low_confidence is True
    assert segs[0].avg_confidence is not None
    assert segs[0].avg_confidence < 0.60


def test_missing_api_key_raises_cloud_auth(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CLOUD_STT_API_KEY", raising=False)
    with pytest.raises(TranscriptionError) as ei:
        DeepgramTranscriber().transcribe("x.wav")
    assert ei.value.error_code == AsrErrorCode.CLOUD_AUTH


def test_cloud_stt_api_key_alias(monkeypatch, audio_wav):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("CLOUD_STT_API_KEY", "test-key")
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    with patch("src.transcription.cloud_deepgram.httpx.post", return_value=mock_resp) as post:
        out = DeepgramTranscriber().transcribe(audio_wav)
    assert out.backend == "deepgram"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Token test-key"


def test_transcribe_maps_response(monkeypatch, audio_wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    with patch("src.transcription.cloud_deepgram.httpx.post", return_value=mock_resp):
        out = DeepgramTranscriber().transcribe(audio_wav, language="sv")
    assert out.backend == "deepgram"
    assert out.model == "nova-2"
    assert out.language == "sv"
    assert out.duration == 1.2
    assert out.segments[0].text == "Hej jag behöver hjälp"


def test_429_raises_cloud_quota(monkeypatch, audio_wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    with (
        patch("src.transcription.cloud_deepgram.httpx.post", return_value=mock_resp),
        patch("src.transcription.cloud_deepgram.time.sleep"),
    ):
        with pytest.raises(TranscriptionError) as ei:
            DeepgramTranscriber().transcribe(audio_wav)
    assert ei.value.error_code == AsrErrorCode.CLOUD_QUOTA


def test_timeout_raises_cloud_timeout(monkeypatch, audio_wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    with (
        patch(
            "src.transcription.cloud_deepgram.httpx.post",
            side_effect=httpx.TimeoutException("timed out"),
        ),
        patch("src.transcription.cloud_deepgram.time.sleep"),
    ):
        with pytest.raises(TranscriptionError) as ei:
            DeepgramTranscriber().transcribe(audio_wav)
    assert ei.value.error_code == AsrErrorCode.CLOUD_TIMEOUT


def test_router_cloud_without_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(TranscriptionError) as ei:
            AsrRouter().transcribe("x.wav", provider="cloud")
    assert ei.value.error_code == AsrErrorCode.CLOUD_AUTH


def test_router_cloud_success(monkeypatch, audio_wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    with patch("src.transcription.cloud_deepgram.httpx.post", return_value=mock_resp):
        out = AsrRouter().transcribe(audio_wav, provider="cloud")
    assert out.provider == "cloud"
    assert out.backend == "deepgram"
    assert out.segments


def test_default_no_cloud_fallback_on_timeout(monkeypatch, audio_wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    with (
        patch(
            "src.transcription.cloud_deepgram.httpx.post",
            side_effect=httpx.TimeoutException("timed out"),
        ),
        patch("src.transcription.cloud_deepgram.time.sleep"),
        patch("src.transcription.router.get_transcriber") as mock_factory,
    ):
        with pytest.raises(TranscriptionError) as ei:
            AsrRouter().transcribe(audio_wav, provider="cloud", cloud_fallback_local=False)
    assert ei.value.error_code == AsrErrorCode.CLOUD_TIMEOUT
    mock_factory.assert_not_called()


def test_cloud_fallback_on_timeout_when_enabled(monkeypatch, audio_wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    from src.core.models import Segment, Transcript

    fake = Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        segments=[Segment(0, 1, "lokal transkript")],
    )
    mock_t = MagicMock()
    mock_t.transcribe.return_value = fake
    with (
        patch(
            "src.transcription.cloud_deepgram.httpx.post",
            side_effect=httpx.TimeoutException("timed out"),
        ),
        patch("src.transcription.cloud_deepgram.time.sleep"),
        patch("src.transcription.router.get_transcriber", return_value=mock_t),
    ):
        out = AsrRouter().transcribe(
            audio_wav,
            provider="cloud",
            cloud_fallback_local=True,
        )
    assert out.provider == "local"
    assert out.segments[0].text == "lokal transkript"
    mock_t.transcribe.assert_called_once()
