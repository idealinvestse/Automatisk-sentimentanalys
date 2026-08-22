"""Gated integration tests for the real sv/callcenter WAV pack."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.benchmarks.audio_catalog import parse_sample_metadata
from src.benchmarks.audio_models import SamplePack
from src.benchmarks.audio_runner import run_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
CALLCENTER = REPO_ROOT / "samples" / "audio" / "sv" / "callcenter"
AUDIO_ROOT = REPO_ROOT / "samples" / "audio"
REAL_STEMS = ("090932", "091922", "100545", "103330", "104207")


def _real_wavs() -> list[Path]:
    return [CALLCENTER / f"{stem}.wav" for stem in REAL_STEMS]


def test_committed_sidecars_parse_audio_smoke_v1():
    pack = SamplePack(id="sv_callcenter", label="Svenska callcenter-samtal", parser="sidecar")
    for stem in REAL_STEMS:
        parsed = parse_sample_metadata(CALLCENTER / f"{stem}.wav", pack)
        assert parsed.language == "sv"
        assert parsed.schema_name == "audio_smoke_v1"
        assert parsed.expected_transcript_contains
        assert parsed.skip_ml is False


def _require_real_pack() -> None:
    if os.environ.get("SENTIMENT_SKIP_AUDIO") == "1":
        pytest.skip("SENTIMENT_SKIP_AUDIO=1")
    missing = [path.name for path in _real_wavs() if not path.is_file()]
    if missing:
        pytest.skip(f"real sv/callcenter WAV files missing: {missing}")
    if not (AUDIO_ROOT / "manifest.yaml").is_file():
        pytest.skip("samples/audio/manifest.yaml not present")


@pytest.mark.audio
@pytest.mark.slow
def test_sv_callcenter_real_smoke_asserts_sidecar_phrases():
    _require_real_pack()
    report = run_scenario(
        "smoke",
        audio_root=str(AUDIO_ROOT),
        pack_ids=["sv_callcenter"],
        device=os.environ.get("SENTIMENT_AUDIO_DEVICE", "cpu"),
    )
    assert report.n_files == 5
    assert report.summary.get("n_failed", 1) == 0
    assert all(row.ok for row in report.files)
    seen = {Path(row.path).stem for row in report.files}
    assert seen == set(REAL_STEMS)


@pytest.mark.audio
@pytest.mark.slow
def test_sv_callcenter_real_pipeline_uses_callcenter_profile():
    _require_real_pack()
    report = run_scenario(
        "pipeline",
        audio_root=str(AUDIO_ROOT),
        pack_ids=["sv_callcenter"],
        device=os.environ.get("SENTIMENT_AUDIO_DEVICE", "cpu"),
    )
    assert report.n_files == 5
    assert report.summary.get("n_failed", 1) == 0
    assert all(row.ok and row.pipeline_ok for row in report.files)
    for row in report.files:
        assert row.expected_sentiment
        assert row.transcript_preview
