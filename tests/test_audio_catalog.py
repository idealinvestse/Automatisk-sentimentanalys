"""Fast structural tests for the audio sample catalog (no ML)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmarks.audio_catalog import (
    AudioCatalog,
    default_audio_root,
    load_catalog,
    parse_ravdess_filename,
)
from src.benchmarks.audio_models import SampleFilter
from src.benchmarks.audio_scenarios import resolve_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPO_ROOT / "samples" / "audio"


@pytest.fixture
def catalog() -> AudioCatalog:
    if not (AUDIO_ROOT / "manifest.yaml").is_file():
        pytest.skip("samples/audio/manifest.yaml not present")
    return load_catalog(AUDIO_ROOT)


def test_default_audio_root_finds_manifest():
    root = default_audio_root(REPO_ROOT)
    assert (root / "manifest.yaml").is_file()


def test_manifest_loads(catalog: AudioCatalog):
    assert "ravdess_en" in catalog.manifest.packs
    assert catalog.manifest.packs["ravdess_en"].parser == "ravdess_speech"


def test_ravdess_file_count(catalog: AudioCatalog):
    samples = catalog.discover(SampleFilter(pack_ids=["ravdess_en"]))
    assert len(samples) == 1440


def test_ravdess_parse_known_filename(catalog: AudioCatalog):
    pack = catalog.manifest.packs["ravdess_en"]
    meta = parse_ravdess_filename("03-01-03-01-01-01-01.wav", pack)
    assert meta is not None
    assert meta.emotion == "happy"
    assert meta.intensity == "normal"
    assert meta.statement_id == "01"
    assert meta.actor == "01"
    assert meta.expected_sentiment == "positiv"


def test_emotion_coverage_subset_has_eight_emotions(catalog: AudioCatalog):
    samples = resolve_samples(catalog, "asr", pack_ids=["ravdess_en"])
    emotions = {s.metadata.emotion for s in samples}
    assert len(samples) == 8
    assert emotions == {
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    }


def test_smoke_subset_is_deterministic(catalog: AudioCatalog):
    first = resolve_samples(catalog, "smoke", pack_ids=["ravdess_en"])
    second = resolve_samples(catalog, "smoke", pack_ids=["ravdess_en"])
    assert [s.relative_path for s in first] == [s.relative_path for s in second]
    assert len(first) == 3


def test_sv_callcenter_disabled_without_files(catalog: AudioCatalog):
    active = catalog.active_packs()
    sv_dir = AUDIO_ROOT / "sv" / "callcenter"
    has_audio = sv_dir.is_dir() and any(
        p.suffix.lower() in {".wav", ".mp3", ".flac"} for p in sv_dir.rglob("*") if p.is_file()
    )
    if not has_audio:
        assert "sv_callcenter" not in active


def test_validate_ravdess_pack(catalog: AudioCatalog):
    report = catalog.validate()
    ravdess = report.packs["ravdess_en"]
    assert ravdess["active"] is True
    assert ravdess["file_count"] == 1440
    assert ravdess["parse_failures"] == 0
    assert report.ok is True
