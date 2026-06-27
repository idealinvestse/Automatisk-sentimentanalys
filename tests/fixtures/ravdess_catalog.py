"""Minimal RAVDESS audio catalog for tests when the full dataset is not installed."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.benchmarks.audio_catalog import AudioCatalog, load_catalog
from src.benchmarks.audio_models import SampleFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_AUDIO_ROOT = REPO_ROOT / "samples" / "audio"
FULL_RAVDESS_COUNT = 1440

_EMOTION_CODES = ("01", "02", "03", "04", "05", "06", "07", "08")


def ravdess_file_count(audio_root: Path | None = None) -> int:
    root = audio_root or REPO_AUDIO_ROOT
    if not (root / "manifest.yaml").is_file():
        return 0
    catalog = load_catalog(root)
    return len(catalog.discover(SampleFilter(pack_ids=["ravdess_en"])))


def full_ravdess_available() -> bool:
    return ravdess_file_count() >= FULL_RAVDESS_COUNT


def build_mini_ravdess_catalog(target: Path) -> Path:
    """Create a tiny RAVDESS pack (8 emotions, actor 01) for structural tests."""
    target.mkdir(parents=True, exist_ok=True)
    manifest = REPO_AUDIO_ROOT / "manifest.yaml"
    if manifest.is_file():
        shutil.copy(manifest, target / "manifest.yaml")
    else:
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    actor_dir = target / "Actor_01"
    actor_dir.mkdir(parents=True, exist_ok=True)
    for code in _EMOTION_CODES:
        filename = f"03-01-{code}-01-01-01-01.wav"
        (actor_dir / filename).write_bytes(b"RIFF\x00\x00\x00\x00")
    return target


def load_test_catalog(tmp_path: Path) -> tuple[AudioCatalog, int]:
    """Load repo catalog when full RAVDESS is present, otherwise a mini fixture."""
    if full_ravdess_available():
        return load_catalog(REPO_AUDIO_ROOT), FULL_RAVDESS_COUNT
    mini_root = tmp_path / "audio_mini"
    build_mini_ravdess_catalog(mini_root)
    return load_catalog(mini_root), len(_EMOTION_CODES)