"""Tests for sv_callcenter pack discovery and smoke_subset selection."""

from __future__ import annotations

from pathlib import Path

from src.benchmarks.audio_catalog import AudioCatalog, parse_sample_metadata
from src.benchmarks.audio_models import SampleFilter, SamplePack
from src.benchmarks.audio_scenarios import resolve_samples

_MINIMAL_WAV = b"RIFF\x00\x00\x00\x00"

_SIDECAR_MANIFEST = """
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
"""


def _write_pack(root: Path, n_files: int = 5, *, skip_last: bool = False) -> Path:
    pack = root / "sv" / "callcenter"
    pack.mkdir(parents=True)
    for i in range(n_files):
        (pack / f"demo_{i}.wav").write_bytes(_MINIMAL_WAV)
        sidecar = [
            "schema: audio_smoke_v1",
            "language: sv",
            "expected_sentiment: neutral",
            "scenario: routing",
            "speakers: 2",
            "expected_transcript_contains:",
            '  - "hej"',
        ]
        if skip_last and i == n_files - 1:
            sidecar = [
                "schema: audio_smoke_v1",
                "language: sv",
                "skip_ml: true",
                "notes: synthetic tone",
            ]
        (pack / f"demo_{i}.meta.yaml").write_text("\n".join(sidecar) + "\n", encoding="utf-8")
    (root / "manifest.yaml").write_text(_SIDECAR_MANIFEST, encoding="utf-8")
    return pack


def test_smoke_subset_returns_sidecar_pack_files(tmp_path: Path):
    root = tmp_path / "audio"
    _write_pack(root, 5)
    catalog = AudioCatalog(root)
    samples = catalog.discover(
        SampleFilter(pack_ids=["sv_callcenter"], subset="smoke_subset", limit=5)
    )
    assert len(samples) == 5


def test_parse_sidecar_audio_smoke_v1(tmp_path: Path):
    wav = tmp_path / "demo.wav"
    wav.write_bytes(_MINIMAL_WAV)
    (tmp_path / "demo.meta.yaml").write_text(
        """
schema: audio_smoke_v1
language: sv
expected_sentiment: negativ
scenario: billing_complaint
speakers: 2
expected_transcript_contains:
  - "faktura"
  - "Återbetalning"
skip_ml: false
notes: fixture
""",
        encoding="utf-8",
    )
    pack = SamplePack(id="sv_callcenter", label="test", parser="sidecar")
    meta = parse_sample_metadata(wav, pack)
    assert meta.language == "sv"
    assert meta.expected_transcript_contains == ["faktura", "Återbetalning"]
    assert meta.skip_ml is False
    assert meta.schema_name == "audio_smoke_v1"
    assert "expected_transcript_contains" not in meta.extra
    assert meta.expected_sentiment == "negativ"


def test_sidecar_smoke_runs_all_files_without_limit(tmp_path: Path):
    root = tmp_path / "audio"
    _write_pack(root, 5)
    catalog = AudioCatalog(root)
    samples = resolve_samples(catalog, "smoke", pack_ids=["sv_callcenter"])
    assert len(samples) == 5


def test_sidecar_pipeline_runs_all_files_without_limit(tmp_path: Path):
    root = tmp_path / "audio"
    _write_pack(root, 5)
    catalog = AudioCatalog(root)
    samples = resolve_samples(catalog, "pipeline", pack_ids=["sv_callcenter"])
    assert len(samples) == 5


def test_skip_ml_excluded_from_ml_scenarios(tmp_path: Path):
    root = tmp_path / "audio"
    _write_pack(root, 5, skip_last=True)
    catalog = AudioCatalog(root)
    smoke = resolve_samples(catalog, "smoke", pack_ids=["sv_callcenter"])
    catalog_rows = resolve_samples(catalog, "catalog", pack_ids=["sv_callcenter"])
    assert len(smoke) == 4
    assert len(catalog_rows) == 5
    assert all(not sample.metadata.skip_ml for sample in smoke)


def test_skip_ml_fallback_when_only_fixtures(tmp_path: Path):
    root = tmp_path / "audio"
    _write_pack(root, 1, skip_last=True)
    catalog = AudioCatalog(root)
    smoke = resolve_samples(catalog, "smoke", pack_ids=["sv_callcenter"])
    assert len(smoke) == 1
    assert smoke[0].metadata.skip_ml is True


def test_sidecar_validate_requires_expected_phrases(tmp_path: Path):
    root = tmp_path / "audio"
    pack = root / "sv" / "callcenter"
    pack.mkdir(parents=True)
    (pack / "bare.wav").write_bytes(_MINIMAL_WAV)
    (root / "manifest.yaml").write_text(_SIDECAR_MANIFEST, encoding="utf-8")
    report = AudioCatalog(root).validate()
    assert report.ok is False
    assert any("expected_transcript_contains" in err for err in report.errors)


def test_brace_glob_discovers_wav_mp3_flac(tmp_path: Path):
    """Manifest uses ``**/*.{wav,mp3,flac}``; stdlib glob alone finds nothing."""
    root = tmp_path / "audio"
    pack = root / "sv" / "callcenter"
    pack.mkdir(parents=True)
    (pack / "a.wav").write_bytes(_MINIMAL_WAV)
    (pack / "b.mp3").write_bytes(b"ID3")
    (pack / "c.flac").write_bytes(b"fLaC")
    (pack / "d.txt").write_text("skip", encoding="utf-8")
    (root / "manifest.yaml").write_text(
        """
version: 1
packs:
  sv_callcenter:
    label: test
    language: sv
    root: sv/callcenter
    glob: "**/*.{wav,mp3,flac}"
    parser: sidecar
    default_asr_language: sv
    tags: [swedish]
    enabled: true
""",
        encoding="utf-8",
    )
    samples = AudioCatalog(root).discover(SampleFilter(pack_ids=["sv_callcenter"]))
    assert sorted(Path(s.path).name for s in samples) == ["a.wav", "b.mp3", "c.flac"]
