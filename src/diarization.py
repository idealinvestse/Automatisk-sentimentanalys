"""Speaker diarization for Swedish call center conversations.

Supports two backends:
    - 'pyannote': pyannote.audio (requires HF token, best accuracy)
    - 'heuristic': simple energy/VAD-based speaker change detection (no dependencies)

Usage:
    from src.diarization import DiarizationPipeline
    dp = DiarizationPipeline()
    segments = dp.diarize("audio.wav", num_speakers=2)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional backends
_HAS_PYANNOTE = False
try:
    from pyannote.audio import Pipeline as PyannotePipeline

    _HAS_PYANNOTE = True
except ImportError:
    PyannotePipeline = None  # type: ignore

_HAS_WHISPERX = False
try:
    import whisperx  # type: ignore # noqa: F401

    _HAS_WHISPERX = True
except ImportError:
    pass


@dataclass
class SpeakerSegment:
    """A diarized segment with speaker label and timing."""

    start: float
    end: float
    speaker: str
    confidence: float = 1.0


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""

    segments: list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0
    speakers: list[str] = field(default_factory=list)
    backend: str = "heuristic"
    processing_time_s: float = 0.0
    audio_duration_s: float | None = None

    def speaker_timeline(self, speaker: str) -> list[tuple[float, float]]:
        """Get time ranges where a specific speaker is active."""
        return [(s.start, s.end) for s in self.segments if s.speaker == speaker]

    def speaker_ratio(self, speaker: str) -> float:
        """Fraction of total audio spoken by a given speaker."""
        if not self.segments or self.audio_duration_s is None or self.audio_duration_s <= 0:
            return 0.0
        total = sum(s.end - s.start for s in self.segments if s.speaker == speaker)
        return total / self.audio_duration_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker, "confidence": s.confidence}
                for s in self.segments
            ],
            "num_speakers": self.num_speakers,
            "speakers": self.speakers,
            "backend": self.backend,
            "processing_time_s": self.processing_time_s,
        }


class DiarizationPipeline:
    """Speaker diarization pipeline with multiple backends.

    Args:
        backend: 'pyannote' (requires HF token) or 'heuristic' (no dependencies).
        hf_token: HuggingFace token for pyannote models.
        device: 'cpu', 'cuda', or 'auto'.
    """

    def __init__(
        self,
        backend: str = "heuristic",
        hf_token: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.backend = backend
        self.device = device
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self._pipeline: Any = None

        if backend == "pyannote":
            self._init_pyannote()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def diarize(
        self,
        audio_path: str,
        num_speakers: int | None = 2,
        min_segment_length: float = 1.0,
    ) -> DiarizationResult:
        """Run speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: Expected number of speakers (None = auto-detect).
            min_segment_length: Minimum segment duration in seconds.

        Returns:
            DiarizationResult with speaker-labelled segments.
        """
        t0 = time.time()

        if self.backend == "pyannote" and self._pipeline is not None:
            result = self._diarize_pyannote(audio_path, num_speakers, min_segment_length)
        else:
            result = self._diarize_heuristic(audio_path, num_speakers, min_segment_length)

        result.processing_time_s = round(time.time() - t0, 2)
        return result

    def assign_speakers_to_segments(
        self,
        asr_segments: list[dict[str, Any]],
        diarization: DiarizationResult,
    ) -> list[dict[str, Any]]:
        """Assign speaker labels to ASR transcript segments.

        For each ASR segment, finds the dominant speaker based on temporal overlap
        with diarization segments.

        Args:
            asr_segments: List of ASR segments with 'start' and 'end' keys.
            diarization: DiarizationResult from diarize().

        Returns:
            ASR segments with added 'speaker' and 'speaker_confidence' keys.
        """
        if not diarization.segments:
            for seg in asr_segments:
                seg["speaker"] = "UNKNOWN"
                seg["speaker_confidence"] = 0.0
            return asr_segments

        for asr_seg in asr_segments:
            a_start = float(asr_seg.get("start", 0) or 0)
            a_end = float(asr_seg.get("end", a_start + 1) or a_start + 1)

            # Find overlapping diarization segments
            speaker_votes: dict[str, float] = {}
            for ds in diarization.segments:
                overlap = min(a_end, ds.end) - max(a_start, ds.start)
                if overlap > 0:
                    speaker_votes[ds.speaker] = speaker_votes.get(ds.speaker, 0) + overlap

            if speaker_votes:
                dominant = max(speaker_votes.items(), key=lambda kv: kv[1])
                total = sum(speaker_votes.values())
                asr_seg["speaker"] = dominant[0]
                asr_seg["speaker_confidence"] = round(dominant[1] / total, 3) if total > 0 else 0.0
            else:
                asr_seg["speaker"] = "UNKNOWN"
                asr_seg["speaker_confidence"] = 0.0

        return asr_segments

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _init_pyannote(self) -> None:
        """Initialize pyannote pipeline."""
        if not _HAS_PYANNOTE:
            logger.warning("pyannote.audio not installed. Falling back to heuristic.")
            self.backend = "heuristic"
            return

        if not self.hf_token:
            logger.warning(
                "No HF_TOKEN provided. pyannote requires authentication. Falling back to heuristic."
            )
            self.backend = "heuristic"
            return

        try:
            self._pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            if self.device != "cpu":
                import torch

                if torch.cuda.is_available():
                    self._pipeline.to(torch.device(self.device))
            logger.info("pyannote diarization pipeline initialized")
        except Exception as e:
            logger.warning("Failed to initialize pyannote: %s. Falling back to heuristic.", e)
            self.backend = "heuristic"

    def _diarize_pyannote(
        self,
        audio_path: str,
        num_speakers: int | None,
        min_segment_length: float,
    ) -> DiarizationResult:
        """Run pyannote diarization."""
        diarization = self._pipeline(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=1 if num_speakers is None else num_speakers,
            max_speakers=num_speakers or 10,
        )

        segments: list[SpeakerSegment] = []
        speakers_set: set[str] = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration < min_segment_length:
                continue
            segments.append(
                SpeakerSegment(
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                    speaker=f"SPEAKER_{speaker}",
                    confidence=1.0,
                )
            )
            speakers_set.add(f"SPEAKER_{speaker}")

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers_set),
            speakers=sorted(speakers_set),
            backend="pyannote",
        )

    def _diarize_heuristic(
        self,
        audio_path: str,
        num_speakers: int | None,
        min_segment_length: float,
    ) -> DiarizationResult:
        """Simple energy/VAD-based diarization fallback.

        Splits audio into alternating speaker turns based on VAD segments.
        Assumes a 2-person conversation (agent + customer) by default.
        """
        try:
            audio_dur = self._get_audio_duration(audio_path)
        except Exception:
            audio_dur = None

        n_speakers = num_speakers or 2
        speakers = [f"SPEAKER_{i}" for i in range(n_speakers)]

        # Try to use VAD segments if faster-whisper is available
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            audio_np = self._load_audio_numpy(audio_path)
            if audio_np is not None and len(audio_np) > 0:
                timestamps = get_speech_timestamps(
                    audio_np,
                    VadOptions(
                        threshold=0.5,
                        min_speech_duration_ms=int(min_segment_length * 1000),
                        min_silence_duration_ms=500,
                    ),
                )
                segments: list[SpeakerSegment] = []
                for i, ts in enumerate(timestamps):
                    start = ts["start"] / 16000.0
                    end = ts["end"] / 16000.0
                    if end - start >= min_segment_length:
                        speaker = speakers[i % n_speakers]
                        segments.append(
                            SpeakerSegment(
                                start=round(start, 3), end=round(end, 3), speaker=speaker
                            )
                        )
                return DiarizationResult(
                    segments=segments,
                    num_speakers=n_speakers,
                    speakers=speakers,
                    backend="heuristic",
                    audio_duration_s=audio_dur,
                )
        except Exception as e:
            logger.debug("VAD-based diarization failed: %s. Using uniform split.", e)

        # Ultimate fallback: uniform split into alternating segments
        if audio_dur is None:
            audio_dur = 60.0
        segment_dur = max(min_segment_length, 5.0)
        segments = []
        pos = 0.0
        speaker_idx = 0
        while pos < audio_dur:
            end = min(pos + segment_dur, audio_dur)
            segments.append(
                SpeakerSegment(
                    start=round(pos, 3),
                    end=round(end, 3),
                    speaker=speakers[speaker_idx % n_speakers],
                )
            )
            pos = end
            speaker_idx += 1

        return DiarizationResult(
            segments=segments,
            num_speakers=n_speakers,
            speakers=speakers,
            backend="heuristic",
            audio_duration_s=audio_dur,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_audio_duration(path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning("Failed to get audio duration for %s: %s", path, e)
            return 0.0

    @staticmethod
    def _load_audio_numpy(path: str) -> np.ndarray | None:
        """Load audio as numpy array (16kHz mono)."""
        try:
            import subprocess
            import tempfile
            import wave

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # noqa: SIM115
            tmp_path = tmp.name
            tmp.close()  # Close immediately to avoid Windows lock issues

            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True,
                timeout=30,
                check=True,
            )
            with wave.open(tmp_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            with suppress(OSError):
                os.unlink(tmp_path)  # Windows may still hold the file briefly
            return audio
        except Exception as e:
            logger.warning("Failed to load audio as numpy from %s: %s", path, e)
            return None


__all__ = [
    "DiarizationPipeline",
    "DiarizationResult",
    "SpeakerSegment",
]
