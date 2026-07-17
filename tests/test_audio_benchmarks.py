"""Slow integration tests for audio benchmarks (real ASR when enabled)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

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
@patch("src.transcription.router.AsrRouter.transcribe")
def test_smoke_oom_fallback_uses_medium(mock_transcribe, _mock_requires_ml, tmp_path):
    from src.core.models import Segment, Transcript

    audio_root, _ = _audio_root(tmp_path)
    calls: list[str] = []

    def _side_effect(*args, **kwargs):
        model = kwargs.get("model_name", "kb-whisper-large")
        calls.append(model)
        if model == "kb-whisper-large":
            raise RuntimeError("CUDA out of memory")
        return Transcript(
            model=model,
            backend="faster",
            language="sv",
            duration=1.0,
            processing_time=0.1,
            segments=[Segment(start=0.0, end=1.0, text="hej")],
        )

    mock_transcribe.side_effect = _side_effect
    report = run_scenario(
        "smoke",
        audio_root=audio_root,
        device="cpu",
        model_name="kb-whisper-large",
        oom_fallback=True,
    )
    assert report.summary.get("oom_fallbacks", 0) >= 1
    assert "kb-whisper-medium" in calls


@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.transcription.router.AsrRouter.transcribe")
def test_smoke_with_mocked_asr(mock_transcribe, _mock_requires_ml, tmp_path):
    audio_root, _ = _audio_root(tmp_path)
    from src.core.models import Segment, Transcript

    mock_transcribe.return_value = Transcript(
        model="test",
        backend="faster",
        language="en",
        duration=1.0,
        processing_time=0.1,
        segments=[Segment(start=0.0, end=1.0, text="Kids are talking by the door")],
    )

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


def test_compare_dry_run_local_provider(tmp_path, monkeypatch):
    audio_root, _ = _audio_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from src.evaluate import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "audio",
            "compare",
            "--dry-run",
            "--providers",
            "local",
            "--limit",
            "2",
            "--audio-root",
            audio_root,
        ],
    )
    assert result.exit_code == 0
    assert "local" in result.output.lower()

    reports = list((tmp_path / "reports").glob("audio_compare_*.json"))
    assert reports
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert "local" in payload["providers"]
    assert any(row["provider"] == "local" for row in payload["results"])
