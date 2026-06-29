"""Fas 1 end-to-end smoke tests for CLI and API (mocked ASR/ML)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from src.api import app as api_app
from src.api.settings import get_api_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
client = TestClient(api_app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_api_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


@pytest.fixture
def scan_directory(tmp_path):
    d = tmp_path / "audio_dir"
    d.mkdir()
    (d / "a.wav").write_bytes(b"RIFF")
    (d / "b.wav").write_bytes(b"RIFF")
    return str(d)


def _mock_sentiment(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_analyze(self, texts, **kwargs):
        return [{"label": "negativ", "score": 0.7} for _ in texts]

    monkeypatch.setattr(
        "src.analysis.sentiment.SentimentPipeline.analyze",
        mock_analyze,
    )


class TestAnalyzeCallE2E:
    def test_analyze_call_with_mistral_flag_mocked_asr(self, monkeypatch, tmp_path):
        _mock_sentiment(monkeypatch)
        audio = tmp_path / "call.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        fake_report = MagicMock()
        fake_report.to_dict.return_value = {
            "segments": [{"text": "Hej", "speaker": "A"}],
            "results": {
                "agent_performance": {"agent": {"empathy_score": 0.6}},
                "pii_redaction": {"total_redacted": 0},
            },
            "llm": {"fallback": True},
        }

        with patch("src.cli.CallAnalysisPipeline") as mock_pipe:
            inst = mock_pipe.return_value
            inst.analyze_audio.return_value = fake_report
            from src.cli import app as cli_app

            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "analyze-call",
                    str(audio),
                    "--use-mistral-llm",
                    "--backend",
                    "faster",
                    "--device",
                    "cpu",
                ],
            )
        assert result.exit_code == 0
        assert mock_pipe.call_args.kwargs.get("use_mistral_llm") is True


class TestAPIE2ESmoke:
    def test_analyze_pipeline_deep_analysis_mocked(self):
        fake_report = MagicMock()
        fake_report.sentiment_results = []
        fake_report.intent_results = []
        fake_report.summary = {}
        fake_report.topics = {}
        fake_report.insights = {}
        fake_report.risks = {}
        fake_report.processing_time_s = 0.1
        fake_report.llm = {"fallback": True}
        fake_report.results = {
            "agent_performance": {"agent": {"empathy_score": 0.5}},
            "qa": {"overall_qa_score": 80, "passed": True},
        }

        with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
            inst = mock_pipe.return_value
            inst.analyze_segments.return_value = fake_report
            r = client.post(
                "/analyze_pipeline",
                json={
                    "segments": [{"text": "Faktura fel", "start": 0, "end": 2}],
                    "deep_analysis": True,
                    "use_mistral_llm": True,
                },
            )
        assert r.status_code == 200
        assert "sentiment_results" in r.json()

    def test_scan_process_analyze_with_pipeline_mock(self, scan_directory):
        with (
            patch(
                "src.api.routers.scan.resolve_and_validate_audio_paths",
                return_value=[f"{scan_directory}/a.wav"],
            ),
            patch(
                "src.api.services.conversation.transcribe_helper",
                return_value={"segments": [{"text": "Hej", "start": 0, "end": 1}]},
            ),
            patch(
                "src.api.services.conversation.analyze_smart",
                return_value=([{"label": "positiv", "score": 0.8}], {"profile": "call"}),
            ),
        ):
            r = client.post(
                "/scan_process",
                json={"directory": scan_directory, "operation": "analyze_conversation"},
            )
        assert r.status_code == 200
        assert r.json()["items"][0]["data"]["segment_sentiments"]


class TestFas4ValidationCommand:
    def test_fas4_validation_generates_report(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from src.evaluate import app as eval_app

        monkeypatch.setattr(
            "src.llm.mistral_analyzer.ConversationMistralAnalyzer.analyze_full_conversation",
            lambda self, **kwargs: {"fallback": True, "meta": {"llm_used": False}},
        )
        out_md = tmp_path / "validation.md"
        out_json = tmp_path / "validation.json"
        runner = CliRunner()
        result = runner.invoke(
            eval_app,
            [
                "fas4-validation",
                "--output",
                str(out_md),
                "--json-output",
                str(out_json),
                "--testset",
                str(REPO_ROOT / "data" / "test_swedish.csv"),
            ],
        )
        assert result.exit_code == 0
        assert out_md.is_file()
        assert out_json.is_file()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert "pipeline" in payload
        assert payload["pipeline"]["n_samples"] >= 2
