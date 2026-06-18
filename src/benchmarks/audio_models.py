"""Pydantic models for the audio sample catalog and benchmark reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ScenarioId = Literal[
    "catalog",
    "smoke",
    "asr",
    "pipeline",
    "sentiment_chain",
    "language_sanity",
]

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

RAVDESS_INTENSITIES = {
    "01": "normal",
    "02": "strong",
}


class SamplePack(BaseModel):
    """Declarative definition of a sample pack from manifest.yaml."""

    id: str
    label: str
    language: str = "sv"
    root: str = "."
    glob: str = "**/*.wav"
    parser: Literal["ravdess_speech", "sidecar", "none"] = "none"
    default_asr_language: str = "sv"
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    emotion_to_sentiment: dict[str, str] = Field(default_factory=dict)
    statements: dict[str, str] = Field(default_factory=dict)


class AudioManifest(BaseModel):
    version: int = 1
    packs: dict[str, SamplePack] = Field(default_factory=dict)


class ParsedMetadata(BaseModel):
    parser: str
    emotion: str | None = None
    intensity: str | None = None
    statement_id: str | None = None
    statement_text: str | None = None
    repetition: str | None = None
    actor: str | None = None
    expected_sentiment: str | None = None
    scenario: str | None = None
    speakers: int | None = None
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AudioSample(BaseModel):
    pack_id: str
    path: str
    relative_path: str
    language: str
    metadata: ParsedMetadata
    expected_sentiment: str | None = None


class SampleFilter(BaseModel):
    pack_ids: list[str] | None = None
    tags: list[str] | None = None
    emotions: list[str] | None = None
    actors: list[str] | None = None
    limit: int | None = None
    subset: str | None = None


class FileResult(BaseModel):
    path: str
    relative_path: str
    pack_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ok: bool = False
    error: str | None = None
    transcript_preview: str | None = None
    sentiment_pred: str | None = None
    expected_sentiment: str | None = None
    latency_s: float | None = None
    pipeline_ok: bool | None = None
    language_used: str | None = None


class AudioRunReport(BaseModel):
    timestamp: str
    scenario: str
    packs: list[str]
    n_files: int
    duration_s: float
    dry_run: bool = False
    device: str | None = None
    backend: str | None = None
    files: list[FileResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    ok: bool
    packs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
