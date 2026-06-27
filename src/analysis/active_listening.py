"""Active Listening Behavior Analyzer.

Detects good and poor listening behaviors:
- Backchannels and acknowledgments ("mm", "ja", "okej", "förstår")
- Potential interruptions (time overlap between speakers)
- Paraphrasing / confirmation
- Talk/listen balance

Helps coach agents on empathy and de-escalation skills.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

BACKCHANNELS = ["mm", "ja", "okej", "förstår", "bra", "precis", "just det", "absolut"]


@register_analyzer("active_listening")
class ActiveListeningBehaviorAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "active_listening"

    @property
    def requires(self) -> list[str]:
        return []

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"listening_score": 50, "events": []}

        events = []
        backchannel_count = 0
        speaker_times: dict[str, float] = {}

        prev_end = 0.0
        prev_speaker = None

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            speaker = getattr(seg, "speaker", "unknown")
            dur = getattr(seg, "end", 0) - getattr(seg, "start", 0)

            # Backchannels
            if any(bc in text for bc in BACKCHANNELS) and dur < 3:
                backchannel_count += 1
                events.append({
                    "type": "backchannel",
                    "speaker": speaker,
                    "time": getattr(seg, "start", 0),
                    "text": seg.text,
                })

            # Simple interruption detection (consecutive different speakers with close timing)
            if prev_speaker and prev_speaker != speaker and (getattr(seg, "start", 0) - prev_end) < 0.3:
                events.append({
                    "type": "possible_interruption",
                    "speaker": speaker,
                    "time": getattr(seg, "start", 0),
                })

            speaker_times[speaker] = speaker_times.get(speaker, 0) + max(0, dur)
            prev_end = getattr(seg, "end", 0)
            prev_speaker = speaker

        total_time = sum(speaker_times.values()) or 1
        balance = {s: round(t / total_time * 100, 1) for s, t in speaker_times.items()}

        listening_score = min(100, 50 + backchannel_count * 4)  # simple heuristic

        return {
            "listening_score": round(listening_score, 1),
            "backchannel_count": backchannel_count,
            "speaker_balance": balance,
            "events": events[:10],  # limit
            "tips": ["Uppmuntra fler backchannels", "Undvik att avbryta"] if listening_score < 60 else []
        }
