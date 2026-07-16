"""Tests for ASR Prometheus metrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.metrics import (
    ASR_CLOUD_EGRESS_TOTAL,
    ASR_TRANSCRIPTIONS_TOTAL,
    record_asr_transcription,
)
from src.core.models import Segment, Transcript
from src.transcription.router import AsrRouter


@pytest.fixture
def audio_wav(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF")
    return str(path)


def test_record_asr_increments_when_prometheus_available():
    if ASR_TRANSCRIPTIONS_TOTAL is None:
        pytest.skip("prometheus_client not installed")
    before = ASR_TRANSCRIPTIONS_TOTAL.labels(
        provider="local", backend="faster", outcome="success"
    )._value.get()
    record_asr_transcription("local", "faster", "success", 1.2)
    after = ASR_TRANSCRIPTIONS_TOTAL.labels(
        provider="local", backend="faster", outcome="success"
    )._value.get()
    assert after == before + 1


def test_router_local_success_records_metrics():
    if ASR_TRANSCRIPTIONS_TOTAL is None:
        pytest.skip("prometheus_client not installed")
    fake = Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        segments=[Segment(0, 1, "hej")],
    )
    mock_t = MagicMock()
    mock_t.transcribe.return_value = fake
    before = ASR_TRANSCRIPTIONS_TOTAL.labels(
        provider="local", backend="faster", outcome="success"
    )._value.get()
    with patch("src.transcription.router.get_transcriber", return_value=mock_t):
        AsrRouter().transcribe("x.wav", provider="local", backend="faster")
    after = ASR_TRANSCRIPTIONS_TOTAL.labels(
        provider="local", backend="faster", outcome="success"
    )._value.get()
    assert after == before + 1


def test_router_cloud_increments_egress_counter(monkeypatch, audio_wav):
    if ASR_CLOUD_EGRESS_TOTAL is None:
        pytest.skip("prometheus_client not installed")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    data = {"results": {"channels": [{"alternatives": [{"transcript": "Hej", "words": []}]}]}}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    before = ASR_CLOUD_EGRESS_TOTAL._value.get()
    with patch("src.transcription.cloud_deepgram.httpx.post", return_value=mock_resp):
        AsrRouter().transcribe(audio_wav, provider="cloud")
    after = ASR_CLOUD_EGRESS_TOTAL._value.get()
    assert after == before + 1
