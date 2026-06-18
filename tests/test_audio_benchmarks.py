"""Slow integration tests for audio benchmarks (real ASR when enabled)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.audio_runner import run_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPO_ROOT / "samples" / "audio"


pytestmark = [pytest.mark.audio, pytest.mark.slow]


def _skip_if_disabled() -> None:
    if os.environ.get("SENTIMENT_SKIP_AUDIO") == "1":
        pytest.skip("SENTIMENT_SKIP_AUDIO=1")
    if not (AUDIO_ROOT / "manifest.yaml").is_file():
        pytest.skip("samples/audio not available")


def test_catalog_scenario_dry_run():
    _skip_if_disabled()
    report = run_scenario("catalog", audio_root=str(AUDIO_ROOT), dry_run=True)
    assert report.n_files == 1440
    assert report.summary.get("catalog_only") is True


def test_smoke_dry_run_selects_three_files():
    _skip_if_disabled()
    report = run_scenario("smoke", audio_root=str(AUDIO_ROOT), dry_run=True)
    assert report.n_files == 3
    assert report.dry_run is True


@patch("src.transcription.get_transcriber")
def test_smoke_with_mocked_asr(mock_get_transcriber):
    _skip_if_disabled()
    mock_transcriber = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Kids are talking by the door"
    mock_transcriber.transcribe.return_value = MagicMock(segments=[mock_segment], text="")
    mock_get_transcriber.return_value = mock_transcriber

    report = run_scenario("smoke", audio_root=str(AUDIO_ROOT), device="cpu")

    assert report.n_files == 3
    assert report.summary.get("asr_success_rate") == 1.0
    assert all(f.ok for f in report.files)


def test_list_command_via_evaluate():
    _skip_if_disabled()
    from typer.testing import CliRunner

    from src.evaluate import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["audio", "list", "--audio-root", str(AUDIO_ROOT), "--pack", "ravdess_en", "--limit", "2"],
    )
    assert result.exit_code == 0
    assert "ravdess_en" in result.output
