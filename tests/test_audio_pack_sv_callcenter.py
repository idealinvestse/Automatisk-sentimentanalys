"""Tests for sv_callcenter pack discovery and smoke_subset selection."""

from __future__ import annotations

from pathlib import Path

from src.benchmarks.audio_catalog import AudioCatalog
from src.benchmarks.audio_models import SampleFilter

_MINIMAL_WAV = b"RIFF\x00\x00\x00\x00"


def test_smoke_subset_returns_sidecar_pack_files(tmp_path: Path):
    root = tmp_path / "audio"
    pack = root / "sv" / "callcenter"
    pack.mkdir(parents=True)
    for i in range(5):
        (pack / f"demo_{i}.wav").write_bytes(_MINIMAL_WAV)
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
    catalog = AudioCatalog(root)
    samples = catalog.discover(
        SampleFilter(pack_ids=["sv_callcenter"], subset="smoke_subset", limit=5)
    )
    assert len(samples) == 5
