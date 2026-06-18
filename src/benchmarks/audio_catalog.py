"""Load and query the audio sample catalog from samples/audio/manifest.yaml."""

from __future__ import annotations

import glob as _glob
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .audio_models import (
    RAVDESS_EMOTIONS,
    RAVDESS_INTENSITIES,
    AudioManifest,
    AudioSample,
    ParsedMetadata,
    SampleFilter,
    SamplePack,
    ValidationReport,
)

logger = logging.getLogger(__name__)

RAVDESS_FILENAME_RE = re.compile(
    r"^03-01-(?P<emotion>\d{2})-(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-(?P<actor>\d{2})\.wav$",
    re.IGNORECASE,
)


def default_audio_root(start: Path | None = None) -> Path:
    """Resolve samples/audio relative to repo root or cwd."""
    if start is None:
        start = Path.cwd()
    candidates = [
        start / "samples" / "audio",
        start.parent / "samples" / "audio",
    ]
    for candidate in candidates:
        if (candidate / "manifest.yaml").is_file():
            return candidate.resolve()
    return (start / "samples" / "audio").resolve()


def load_manifest(manifest_path: Path) -> AudioManifest:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    packs: dict[str, SamplePack] = {}
    for pack_id, cfg in (raw.get("packs") or {}).items():
        packs[pack_id] = SamplePack(id=pack_id, **(cfg or {}))
    return AudioManifest(version=int(raw.get("version", 1)), packs=packs)


def _pack_is_active(pack: SamplePack, pack_root: Path) -> bool:
    if pack.enabled:
        return True
    if not pack_root.exists():
        return False
    for _root, _dirs, files in os.walk(pack_root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                return True
    return False


def _discover_pack_files(pack: SamplePack, audio_root: Path) -> list[Path]:
    pack_root = (audio_root / pack.root).resolve()
    if not pack_root.exists():
        return []
    pattern = str(pack_root / pack.glob)
    files = [
        Path(p).resolve()
        for p in _glob.glob(pattern, recursive=True)
        if Path(p).is_file() and Path(p).suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ]
    return sorted(set(files))


def parse_ravdess_filename(
    filename: str,
    pack: SamplePack,
) -> ParsedMetadata | None:
    match = RAVDESS_FILENAME_RE.match(filename)
    if not match:
        return None
    groups = match.groupdict()
    emotion_code = groups["emotion"]
    emotion = RAVDESS_EMOTIONS.get(emotion_code)
    if emotion is None:
        return None
    statement_id = groups["statement"]
    return ParsedMetadata(
        parser="ravdess_speech",
        emotion=emotion,
        intensity=RAVDESS_INTENSITIES.get(groups["intensity"]),
        statement_id=statement_id,
        statement_text=pack.statements.get(statement_id),
        repetition=groups["repetition"],
        actor=groups["actor"],
        expected_sentiment=pack.emotion_to_sentiment.get(emotion),
    )


def _load_sidecar_meta(audio_path: Path) -> dict[str, Any]:
    sidecar = audio_path.with_suffix(audio_path.suffix + ".meta.yaml")
    if not sidecar.is_file():
        alt = audio_path.with_name(audio_path.stem + ".meta.yaml")
        if not alt.is_file():
            return {}
        sidecar = alt
    try:
        return yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to read sidecar %s: %s", sidecar, exc)
        return {}


def parse_sample_metadata(audio_path: Path, pack: SamplePack) -> ParsedMetadata:
    if pack.parser == "ravdess_speech":
        parsed = parse_ravdess_filename(audio_path.name, pack)
        if parsed is not None:
            return parsed
    if pack.parser == "sidecar":
        meta = _load_sidecar_meta(audio_path)
        return ParsedMetadata(
            parser="sidecar",
            expected_sentiment=meta.get("expected_sentiment"),
            scenario=meta.get("scenario"),
            speakers=meta.get("speakers"),
            notes=meta.get("notes"),
            extra={k: v for k, v in meta.items() if k not in {"expected_sentiment", "scenario", "speakers", "notes"}},
        )
    return ParsedMetadata(parser=pack.parser)


class AudioCatalog:
    def __init__(self, audio_root: Path | None = None) -> None:
        self.audio_root = (audio_root or default_audio_root()).resolve()
        self.manifest_path = self.audio_root / "manifest.yaml"
        self.manifest = load_manifest(self.manifest_path)

    def active_packs(self) -> dict[str, SamplePack]:
        active: dict[str, SamplePack] = {}
        for pack_id, pack in self.manifest.packs.items():
            pack_root = (self.audio_root / pack.root).resolve()
            if _pack_is_active(pack, pack_root):
                active[pack_id] = pack
        return active

    def discover(self, filters: SampleFilter | None = None) -> list[AudioSample]:
        filters = filters or SampleFilter()
        samples: list[AudioSample] = []
        active = self.active_packs()

        pack_ids = filters.pack_ids or list(active.keys())
        for pack_id in pack_ids:
            pack = active.get(pack_id)
            if pack is None:
                continue
            if filters.tags and not set(filters.tags).issubset(set(pack.tags)):
                continue

            for path in _discover_pack_files(pack, self.audio_root):
                metadata = parse_sample_metadata(path, pack)
                if filters.emotions and metadata.emotion not in filters.emotions:
                    continue
                if filters.actors and metadata.actor not in filters.actors:
                    continue

                rel = str(path.relative_to(self.audio_root)).replace("\\", "/")
                expected = metadata.expected_sentiment
                samples.append(
                    AudioSample(
                        pack_id=pack_id,
                        path=str(path),
                        relative_path=rel,
                        language=pack.language,
                        metadata=metadata,
                        expected_sentiment=expected,
                    )
                )

        samples.sort(key=lambda s: s.relative_path)
        if filters.subset:
            samples = _apply_subset(samples, filters.subset, active)
        if filters.limit is not None:
            samples = samples[: filters.limit]
        return samples

    def validate(self) -> ValidationReport:
        errors: list[str] = []
        pack_reports: dict[str, dict[str, Any]] = {}

        if not self.manifest_path.is_file():
            return ValidationReport(ok=False, errors=[f"Manifest missing: {self.manifest_path}"])

        for pack_id, pack in self.manifest.packs.items():
            pack_root = (self.audio_root / pack.root).resolve()
            active = _pack_is_active(pack, pack_root)
            files = _discover_pack_files(pack, self.audio_root) if active else []
            parse_failures = 0
            if pack.parser == "ravdess_speech" and files:
                for path in files:
                    if parse_ravdess_filename(path.name, pack) is None:
                        parse_failures += 1
                        if parse_failures <= 3:
                            errors.append(f"{pack_id}: unparseable RAVDESS filename: {path.name}")

            pack_reports[pack_id] = {
                "active": active,
                "enabled": pack.enabled,
                "root": str(pack_root),
                "file_count": len(files),
                "parse_failures": parse_failures,
                "tags": pack.tags,
            }
            if pack.enabled and not pack_root.exists():
                errors.append(f"{pack_id}: enabled but root missing: {pack_root}")

        ok = not errors
        return ValidationReport(ok=ok, packs=pack_reports, errors=errors)


def load_catalog(audio_root: Path | str | None = None) -> AudioCatalog:
    root = Path(audio_root).resolve() if audio_root else None
    return AudioCatalog(root)


def _apply_subset(
    samples: list[AudioSample],
    subset: str,
    active_packs: dict[str, SamplePack],
) -> list[AudioSample]:
    if subset == "smoke_subset":
        desired = [
            ("happy", "03"),
            ("angry", "05"),
            ("neutral", "01"),
        ]
        chosen: list[AudioSample] = []
        ravdess = [s for s in samples if s.pack_id == "ravdess_en"]
        for _name, code in desired:
            for sample in ravdess:
                if (
                    sample.metadata.emotion == RAVDESS_EMOTIONS[code]
                    and sample.metadata.actor == "01"
                    and sample.metadata.statement_id == "01"
                ):
                    chosen.append(sample)
                    break
        return chosen or samples[:3]

    if subset == "emotion_coverage":
        chosen = []
        seen: set[str] = set()
        ravdess = [s for s in samples if s.pack_id == "ravdess_en"]
        for sample in ravdess:
            if sample.metadata.actor != "01":
                continue
            if sample.metadata.statement_id != "01":
                continue
            if sample.metadata.intensity != "normal":
                continue
            emotion = sample.metadata.emotion
            if emotion and emotion not in seen:
                seen.add(emotion)
                chosen.append(sample)
        return chosen

    if subset == "one_per_pack":
        chosen = []
        seen_packs: set[str] = set()
        for sample in samples:
            if sample.pack_id not in seen_packs:
                seen_packs.add(sample.pack_id)
                chosen.append(sample)
        return chosen

    return samples
