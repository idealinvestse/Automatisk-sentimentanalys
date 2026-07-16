from unittest.mock import MagicMock, patch

import pytest

from src.core.models import Segment, Transcript
from src.transcription.router import AsrRouter, resolve_asr_provider


def test_default_provider_is_local():
    assert resolve_asr_provider(None) == "local"
    assert resolve_asr_provider("") == "local"
    assert resolve_asr_provider("LOCAL") == "local"


def test_cloud_requires_explicit():
    assert resolve_asr_provider("cloud") == "cloud"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        resolve_asr_provider("azure")


def test_router_local_calls_factory_and_filters():
    fake = Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        segments=[Segment(0, 1, "Thanks for watching")],
    )
    mock_t = MagicMock()
    mock_t.transcribe.return_value = fake
    with patch("src.transcription.router.get_transcriber", return_value=mock_t):
        out = AsrRouter().transcribe("x.wav", provider="local")
    assert out.segments == []
    assert out.provider == "local"
    mock_t.transcribe.assert_called_once()


def test_router_cloud_without_adapter_raises():
    with pytest.raises(Exception) as ei:
        AsrRouter().transcribe("x.wav", provider="cloud")
    assert (
        "CLOUD" in str(ei.value).upper()
        or "deepgram" in str(ei.value).lower()
        or "not configured" in str(ei.value).lower()
    )
