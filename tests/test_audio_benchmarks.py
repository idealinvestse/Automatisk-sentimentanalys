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


@patch("torch.cuda.is_available", return_value=False)
def test_cuda_unavailable_raises_on_cuda_device(_mock_cuda_available):
    from src.benchmarks.audio_runner import _run_asr_on_sample

    with pytest.raises(
        RuntimeError,
        match=r"device=cuda requested but torch\.cuda\.is_available\(\) is False",
    ):
        _run_asr_on_sample(
            "dummy.wav",
            backend="faster",
            device="cuda",
            language="sv",
        )


@patch("torch.cuda.is_available", return_value=False)
def test_cuda_unavailable_raises_on_pipeline_cuda_device(_mock_cuda_available):
    from src.benchmarks.audio_runner import _run_pipeline_on_sample

    with pytest.raises(
        RuntimeError,
        match=r"device=cuda requested but torch\.cuda\.is_available\(\) is False",
    ):
        _run_pipeline_on_sample(
            "dummy.wav",
            backend="faster",
            device="cuda",
            language="sv",
        )


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


@patch("src.benchmarks.audio_runner._run_sentiment_on_text", return_value="positive")
@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.pipeline.CallAnalysisPipeline")
def test_pipeline_oom_fallback_uses_medium(
    mock_pipeline_cls, _mock_requires_ml, _mock_sentiment, tmp_path
):
    from unittest.mock import MagicMock

    audio_root, _ = _audio_root(tmp_path)
    models_used: list[str] = []

    def _pipeline_factory(**kwargs):
        model = kwargs.get("asr_model", "kb-whisper-large")
        models_used.append(model)
        instance = MagicMock()
        if model == "kb-whisper-large":
            instance.analyze_audio.side_effect = RuntimeError("CUDA out of memory")
        else:
            report = MagicMock()
            report.segments = [{"text": "hej"}]
            report.diarization = None
            instance.analyze_audio.return_value = report
        return instance

    mock_pipeline_cls.side_effect = _pipeline_factory
    report = run_scenario(
        "pipeline",
        audio_root=audio_root,
        device="cpu",
        model_name="kb-whisper-large",
        oom_fallback=True,
    )
    assert report.summary.get("oom_fallbacks", 0) >= 1
    assert "kb-whisper-medium" in models_used
    assert all(f.metadata.get("model_used") for f in report.files if f.ok)


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


def _sidecar_audio_root(tmp_path: Path, phrases: list[str]) -> str:
    root = tmp_path / "sidecar_audio"
    pack = root / "sv" / "callcenter"
    pack.mkdir(parents=True)
    (pack / "demo.wav").write_bytes(b"RIFF\x00\x00\x00\x00")
    phrase_yaml = "\n".join(f'  - "{p}"' for p in phrases)
    (pack / "demo.meta.yaml").write_text(
        "schema: audio_smoke_v1\n"
        "language: sv\n"
        "expected_sentiment: neutral\n"
        "expected_transcript_contains:\n"
        f"{phrase_yaml}\n",
        encoding="utf-8",
    )
    (root / "manifest.yaml").write_text(
        """
version: 1
packs:
  sv_callcenter:
    label: test
    language: sv
    root: sv/callcenter
    glob: "**/*.wav"
    parser: sidecar
    default_asr_language: sv
    tags: [swedish]
    enabled: true
""",
        encoding="utf-8",
    )
    return str(root)


@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.transcription.router.AsrRouter.transcribe")
def test_smoke_fails_on_empty_transcript(mock_transcribe, _mock_requires_ml, tmp_path):
    from src.core.models import Transcript

    audio_root = _sidecar_audio_root(tmp_path, ["hej"])
    mock_transcribe.return_value = Transcript(
        model="test",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        segments=[],
    )
    report = run_scenario(
        "smoke", audio_root=audio_root, pack_ids=["sv_callcenter"], device="cpu"
    )
    assert report.summary.get("n_failed") == 1
    assert "empty transcript" in (report.files[0].error or "")


@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.transcription.router.AsrRouter.transcribe")
def test_smoke_fails_on_missing_expected_phrases(mock_transcribe, _mock_requires_ml, tmp_path):
    from src.core.models import Segment, Transcript

    audio_root = _sidecar_audio_root(tmp_path, ["faktura"])
    mock_transcribe.return_value = Transcript(
        model="test",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        segments=[Segment(start=0.0, end=1.0, text="Välkommen till växeln")],
    )
    report = run_scenario(
        "smoke", audio_root=audio_root, pack_ids=["sv_callcenter"], device="cpu"
    )
    assert report.files[0].ok is False
    assert "missing expected phrases" in (report.files[0].error or "")


@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.transcription.router.AsrRouter.transcribe")
def test_smoke_passes_expected_phrases_casefold(mock_transcribe, _mock_requires_ml, tmp_path):
    from src.core.models import Segment, Transcript

    audio_root = _sidecar_audio_root(tmp_path, ["Välkommen till växeln"])
    mock_transcribe.return_value = Transcript(
        model="test",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        segments=[Segment(start=0.0, end=1.0, text="välkommen   till växeln idag")],
    )
    report = run_scenario(
        "smoke", audio_root=audio_root, pack_ids=["sv_callcenter"], device="cpu"
    )
    assert report.files[0].ok is True


@patch("src.benchmarks.audio_runner._run_sentiment_on_text", return_value="neutral")
@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.pipeline.CallAnalysisPipeline")
def test_pipeline_uses_callcenter_profile_and_preprocess(
    mock_pipeline_cls, _mock_requires_ml, _mock_sentiment, tmp_path
):
    from unittest.mock import MagicMock

    audio_root = _sidecar_audio_root(tmp_path, ["hej"])
    instance = MagicMock()
    report = MagicMock()
    report.segments = [{"text": "hej där"}]
    report.diarization = None
    instance.analyze_audio.return_value = report
    mock_pipeline_cls.return_value = instance

    result = run_scenario(
        "pipeline", audio_root=audio_root, pack_ids=["sv_callcenter"], device="cpu"
    )
    assert result.files[0].ok is True
    assert mock_pipeline_cls.call_args.kwargs.get("profile") == "callcenter"
    kwargs = instance.analyze_audio.call_args.kwargs
    assert kwargs.get("preprocess_mode") == "callcenter"
    assert kwargs.get("strict_asr") is True
    assert kwargs.get("run_diarization") is False


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
