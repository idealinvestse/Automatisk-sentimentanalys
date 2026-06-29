"""Slow integration tests for audio benchmarks (real ASR when enabled)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.audio_runner import run_scenario
from tests.fixtures.ravdess_catalog import (
    REPO_AUDIO_ROOT,
    build_mini_ravdess_catalog,
    full_ravdess_available,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPO_AUDIO_ROOT


pytestmark = [pytest.mark.audio, pytest.mark.slow]


def _audio_root(tmp_path) -> tuple[str, int]:
    if os.environ.get("SENTIMENT_SKIP_AUDIO") == "1":
        pytest.skip("SENTIMENT_SKIP_AUDIO=1")
    if not (AUDIO_ROOT / "manifest.yaml").is_file():
        pytest.skip("samples/audio not available")
    if full_ravdess_available():
        return str(AUDIO_ROOT), 1440
    mini = build_mini_ravdess_catalog(tmp_path / "audio_mini")
    return str(mini), 8


def test_catalog_scenario_dry_run(tmp_path):
    audio_root, expected = _audio_root(tmp_path)
    report = run_scenario("catalog", audio_root=audio_root, dry_run=True)
    assert report.n_files == expected
    assert report.summary.get("catalog_only") is True


def test_smoke_dry_run_selects_three_files(tmp_path):
    audio_root, _ = _audio_root(tmp_path)
    report = run_scenario("smoke", audio_root=audio_root, dry_run=True)
    assert report.n_files == 3
    assert report.dry_run is True


@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.transcription.get_transcriber")
def test_smoke_with_mocked_asr(mock_get_transcriber, _mock_requires_ml, tmp_path):
    audio_root, _ = _audio_root(tmp_path)
    mock_transcriber = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Kids are talking by the door"
    mock_transcriber.transcribe.return_value = MagicMock(segments=[mock_segment], text="")
    mock_get_transcriber.return_value = mock_transcriber

    report = run_scenario("smoke", audio_root=audio_root, device="cpu")

    assert report.n_files == 3
    assert report.summary.get("asr_success_rate") == 1.0
    assert all(f.ok for f in report.files)


def test_list_command_via_evaluate(tmp_path):
    audio_root, _ = _audio_root(tmp_path)
    from typer.testing import CliRunner

    from src.evaluate import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["audio", "list", "--audio-root", audio_root, "--pack", "ravdess_en", "--limit", "2"],
    )
    assert result.exit_code == 0
    assert "ravdess_en" in result.output
