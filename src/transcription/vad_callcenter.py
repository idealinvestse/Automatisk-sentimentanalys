"""Call-center tuned Voice Activity Detection (VAD) parameters.

faster-whisper uses Silero VAD with defaults tuned for general speech. Swedish
call-center recordings (narrowband telephony, quiet agents, short interjections
like "ja"/"nej") benefit from lower thresholds and adjusted silence windows.

Used when ``preprocess_mode="callcenter"`` (Transcription v2, Task A-1).
"""

from __future__ import annotations

from typing import Any

# Tuned for 8 kHz-equivalent narrowband after bandpass preprocessing.
# Values are conservative: prefer slightly more speech than missed words.
CALLCENTER_VAD_PARAMETERS: dict[str, float | int] = {
    "threshold": 0.35,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 350,
    "speech_pad_ms": 400,
}


def get_callcenter_vad_parameters() -> dict[str, Any]:
    """Return a copy of VAD kwargs for ``faster_whisper`` / ``VadOptions``."""
    return dict(CALLCENTER_VAD_PARAMETERS)


def vad_options_for_mode(mode: str, *, vad_enabled: bool = True) -> dict[str, Any] | None:
    """Resolve VAD parameter dict for a preprocess mode, or None for defaults."""
    if not vad_enabled:
        return None
    if str(mode).strip().lower() == "callcenter":
        return get_callcenter_vad_parameters()
    return None