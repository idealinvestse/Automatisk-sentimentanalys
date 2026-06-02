"""Base definitions and protocols for ASR transcribers."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Protocol

from ..core.models import Segment, Transcript

logger = logging.getLogger(__name__)

_MODEL_ALIASES = {
    "kb-whisper-large": "KBLab/kb-whisper-large",
    "large-v3": "openai/whisper-large-v3",
}


def resolve_model_name(name: str) -> str:
    """Resolve a model alias or return the name as-is."""
    if not name:
        return _MODEL_ALIASES["kb-whisper-large"]
    key = name.strip()
    return _MODEL_ALIASES.get(key, key)


def add_diarization(
    transcript: Transcript,
    audio_path: str,
    diarize: bool,
    num_speakers: int | None,
) -> Transcript:
    """Add speaker diarization to a Transcript object if requested.

    Returns a new Transcript; the input object is not mutated."""
    if not diarize:
        return transcript

    try:
        from ..diarization import DiarizationPipeline

        dp = DiarizationPipeline(backend="heuristic")
        diar_result = dp.diarize(audio_path, num_speakers=num_speakers)

        # Convert transcript segments to dicts as expected by assign_speakers_to_segments
        segments_dict = [s.to_dict() for s in transcript.segments]
        assigned_dicts = dp.assign_speakers_to_segments(segments_dict, diar_result)

        # Re-convert assigned segment dicts to Segment objects
        updated_segments = [Segment.from_dict(s_dict) for s_dict in assigned_dicts]

        logger.info(
            "Diarization added | speakers=%d | segments=%d",
            diar_result.num_speakers,
            len(updated_segments),
        )
        return replace(
            transcript,
            segments=updated_segments,
            diarization=diar_result.to_dict(),
        )
    except Exception as e:
        logger.warning("Diarization failed: %s. Continuing without speaker labels.", e)
        return replace(
            transcript,
            diarization={"error": str(e), "backend": "failed"},
        )


class Transcriber(Protocol):
    """Protocol defining the interface for transcription backends."""

    def transcribe(
        self,
        audio_path: str,
        language: str = "sv",
        beam_size: int = 5,
        vad: bool = True,
        word_timestamps: bool = True,
        chunk_length_s: int = 30,
        revision: str | None = None,
        diarize: bool = False,
        num_speakers: int | None = 2,
        hotwords: list[str] | None = None,
        initial_prompt: str | None = None,
        preprocess: bool = False,
    ) -> Transcript:
        """Transcribe an audio file and return a Transcript object.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g. 'sv').
            beam_size: Beam size for decoding.
            vad: Enable Voice Activity Detection (VAD) filter.
            word_timestamps: Return word-level timestamps if available.
            chunk_length_s: Chunk length in seconds.
            revision: Model revision or release variant.
            diarize: Enable speaker diarization.
            num_speakers: Expected number of speakers (None for auto).
            hotwords: Optional list of domain-specific words to boost (e.g. company terms,
                      "fakturering", "återbetalning"). Passed to Whisper for better WER on
                      callcenter vocabulary.
            initial_prompt: Optional text prompt to condition the decoder (helps with
                            style, names, expected terminology at the start of the call).
            preprocess: If True, run optional audio preprocessing (high-pass + noise
                        reduction if noisereduce installed) before ASR. Improves WER
                        on noisy call center recordings.
        """
        ...
