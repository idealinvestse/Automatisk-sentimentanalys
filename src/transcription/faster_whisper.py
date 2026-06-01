"""Faster-Whisper transcription backend."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.config import KB_REVISIONS
from ..core.device import normalize_device_for_asr
from ..core.errors import TranscriptionError
from ..core.models import Segment, Transcript, Word
from .base import add_diarization, resolve_model_name

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel  # type: ignore

    _HAS_FASTER = True
except Exception:
    WhisperModel = None
    _HAS_FASTER = False


class FasterWhisperTranscriber:
    """ASR transcriber powered by faster-whisper (ctranslate2)."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if not _HAS_FASTER:
            raise ImportError(
                "faster-whisper is not installed. Install it with pip or choose another backend."
            )
        self.model_name = resolve_model_name(model_name)
        self.device = device
        self._model: WhisperModel | None = None

    def _get_model(self, revision: str | None = None) -> WhisperModel:
        """Lazy-load and cache the WhisperModel instance."""
        if self._model is not None:
            return self._model

        dev_kind, cuda_idx = normalize_device_for_asr(self.device)
        compute_type = (
            "float16" if dev_kind == "cuda" else ("int8" if dev_kind == "cpu" else "float32")
        )

        logger.debug(
            "Loading faster-whisper model '%s' | compute_type=%s | revision=%s",
            self.model_name,
            compute_type,
            revision or "default",
        )

        model_kwargs: dict[str, Any] = {
            "device": dev_kind,
            "device_index": cuda_idx,
            "compute_type": compute_type,
        }
        if revision:
            model_kwargs["revision"] = revision

        try:
            self._model = WhisperModel(self.model_name, **model_kwargs)
            logger.debug("Model loaded successfully: %s", self.model_name)
            return self._model
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load faster-whisper model '{self.model_name}': {e}"
            ) from e

    def transcribe(
        self,
        audio_path: str,
        language: str = "sv",
        beam_size: int = 5,
        vad: bool = True,
        word_timestamps: bool = True,
        chunk_length_s: int = 30,  # Unused in faster-whisper
        revision: str | None = None,
        diarize: bool = False,
        num_speakers: int | None = 2,
    ) -> Transcript:
        """Transcribe audio file using faster-whisper."""
        t0 = time.time()

        # Validate revision
        if revision and revision not in KB_REVISIONS:
            logger.warning(
                "Unknown revision '%s', using default. Valid: %s",
                revision,
                sorted(KB_REVISIONS),
            )
            revision = None

        logger.info(
            "ASR (faster-whisper) start | path=%s | model=%s | revision=%s | device=%s | lang=%s",
            audio_path,
            self.model_name,
            revision or "default",
            self.device,
            language,
        )

        try:
            wmodel = self._get_model(revision=revision)
            segments_iter, info = wmodel.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad,
                word_timestamps=word_timestamps,
            )

            segs: list[Segment] = []
            dur = getattr(info, "duration", None)

            for seg_count, s in enumerate(segments_iter, start=1):
                words: list[Word] = []
                if word_timestamps and getattr(s, "words", None):
                    for w in s.words:
                        words.append(
                            Word(
                                start=float(getattr(w, "start", 0.0) or 0.0),
                                end=float(getattr(w, "end", 0.0) or 0.0),
                                word=getattr(w, "word", ""),
                                prob=float(getattr(w, "probability", 0.0) or 0.0),
                            )
                        )

                avg_conf = None
                if words:
                    avg_conf = float(sum(w.prob for w in words) / max(1, len(words)))

                segs.append(
                    Segment(
                        start=float(getattr(s, "start", 0.0) or 0.0),
                        end=float(getattr(s, "end", 0.0) or 0.0),
                        text=getattr(s, "text", "").strip(),
                        words=words,
                        avg_confidence=avg_conf,
                    )
                )

                if logger.isEnabledFor(logging.DEBUG):
                    preview = getattr(s, "text", "").strip()[:80].replace("\n", " ")
                    logger.debug(
                        "Segment %d | %.2f -> %.2f | '%s'",
                        seg_count,
                        getattr(s, "start", 0.0),
                        getattr(s, "end", 0.0),
                        preview,
                    )

            proc = time.time() - t0
            logger.info(
                "ASR done (faster-whisper) | segs=%d | audio_dur=%s | proc=%.2fs",
                len(segs),
                dur,
                proc,
            )

            transcript = Transcript(
                model=self.model_name,
                backend="faster",
                language=language,
                duration=dur,
                processing_time=proc,
                segments=segs,
                revision=revision,
            )

            return add_diarization(transcript, audio_path, diarize, num_speakers)

        except Exception as e:
            if not isinstance(e, TranscriptionError):
                raise TranscriptionError(f"Transcription failed under faster-whisper: {e}") from e
            raise
