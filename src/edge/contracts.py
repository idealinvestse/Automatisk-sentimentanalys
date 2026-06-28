"""Pydantic contracts for Edge AI offline analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EdgeSegmentResult(BaseModel):
    text: str
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    intent: str | None = None


class EdgeAnalysisResult(BaseModel):
    profile: str
    offline: bool = True
    llm_used: bool = False
    segments: list[EdgeSegmentResult] = Field(default_factory=list)
    summary: str = ""
    limitations: list[str] = Field(
        default_factory=lambda: [
            "No LLM",
            "No diarization (pyannote)",
            "No Fas 4 aggregate endpoints",
        ]
    )
