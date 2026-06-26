"""Hugging Face Transformers transcription backend."""

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
    from transformers import pipeline  # type: ignore

    _HAS_TRANSFORMERS = True
except Exception:
    pipeline = None
    _HAS_TRANSFORMERS = False


class TransformersTranscriber:
    """ASR transcriber powered by Hugging Face Transformers pipeline."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is not installed. Install it with pip or choose another backend."
            )
        self.model_name = resolve_model_name(model_name)
        self.device = device
        self._pipeline: Any = None

    def _get_pipeline(self, revision: str | None = None) -> Any:
        """Lazy-load and cache the Transformers ASR pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        dev_kind, cuda_idx = normalize_device_for_asr(self.device)
        logger.debug(
            "Creating transformers ASR pipeline '%s' | device=%s | revision=%s",
            self.model_name,
            f"cuda:{cuda_idx}" if dev_kind == "cuda" else dev_kind,
            revision or "default",
        )

        if dev_kind == "cuda":
            pipeline_device = cuda_idx
        elif dev_kind == "mps":
            pipeline_device = "mps"
        else:
            pipeline_device = -1

        pipeline_kwargs: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self.model_name,
            "device": pipeline_device,
        }
        if revision:
            pipeline_kwargs["revision"] = revision

        try:
            self._pipeline = pipeline(**pipeline_kwargs)
            logger.debug("Transformers ASR pipeline created successfully")
            return self._pipeline
        except Exception as e:
            raise TranscriptionError(
                f"Failed to create transformers ASR pipeline for model '{self.model_name}': {e}"
            ) from e

    def transcribe(
        self,
        audio_path: str,
        language: str = "sv",
        beam_size: int = 5,  # Unused directly in pipeline standard kwargs, but can be passed in generate_kwargs
        vad: bool = True,  # Unused in transformers standard pipeline
        word_timestamps: bool = True,
        chunk_length_s: int = 30,
        revision: str | None = None,
        diarize: bool = False,
        num_speakers: int | None = 2,
        hotwords: list[str] | None = None,
        initial_prompt: str | None = None,
        preprocess: bool = False,
        preprocess_mode: str | None = None,
    ) -> Transcript:
        """Transcribe audio file using Hugging Face Transformers."""
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
            "ASR (transformers) start | path=%s | model=%s | revision=%s | device=%s | lang=%s | chunk=%ds | hotwords=%s | prompt=%s | preprocess=%s | preprocess_mode=%s",
            audio_path,
            self.model_name,
            revision or "default",
            self.device,
            language,
            chunk_length_s,
            bool(hotwords),
            bool(initial_prompt),
            preprocess,
            preprocess_mode,
        )

        from .preprocess import prepare_asr_audio

        prep, _resolved_mode = prepare_asr_audio(
            audio_path,
            preprocess=preprocess,
            preprocess_mode=preprocess_mode,
        )

        asr_audio_path = prep.path
        try:
            asr_pipeline = self._get_pipeline(revision=revision)

            generate_kwargs: dict[str, Any] = {
                "language": language,
                "task": "transcribe",
                "return_timestamps": "word" if word_timestamps else True,
                "chunk_length_s": chunk_length_s,
            }
            # Add beam size to generation kwargs if custom
            if beam_size != 5:
                generate_kwargs["num_beams"] = beam_size

            # Hotwords and initial_prompt are primarily for faster-whisper / WhisperX.
            # We forward them for completeness; the HF pipeline may ignore or partially support
            # via generate config (best-effort for transformers backend).
            if initial_prompt:
                generate_kwargs["initial_prompt"] = initial_prompt
            if hotwords:
                generate_kwargs["hotwords"] = hotwords  # may be used by custom processors

            t_call = time.time()
            out = asr_pipeline(asr_audio_path, **generate_kwargs)
            logger.info(
                "ASR backend call finished (transformers) | call_time=%.2fs", time.time() - t_call
            )

            segs: list[Segment] = []
            dur = None

            if isinstance(out, dict) and "chunks" in out:
                for ch in out["chunks"]:
                    start = float(ch.get("timestamp", [0.0, 0.0])[0] or 0.0)
                    end = float(ch.get("timestamp", [0.0, 0.0])[1] or 0.0)
                    text = ch.get("text", "").strip()

                    words: list[Word] = []
                    if word_timestamps and "timestamps" in ch:
                        for w in ch["timestamps"]:
                            words.append(
                                Word(
                                    start=float(w.get("timestamp", [0.0, 0.0])[0] or 0.0),
                                    end=float(w.get("timestamp", [0.0, 0.0])[1] or 0.0),
                                    word=w.get("text", ""),
                                    prob=float(w.get("avg_logprob", 0.0) or 0.0),
                                )
                            )

                    avg_conf = None
                    if words:
                        avg_conf = float(sum(w.prob for w in words) / max(1, len(words)))

                    low_conf = avg_conf is not None and avg_conf < 0.60
                    segs.append(
                        Segment(
                            start=start,
                            end=end,
                            text=text,
                            words=words,
                            avg_confidence=avg_conf,
                            confidence=avg_conf,
                            low_confidence=low_conf,
                        )
                    )
            else:
                # Single segment fallback
                text_val = (out.get("text") if isinstance(out, dict) else str(out)).strip()
                segs.append(
                    Segment(
                        start=0.0,
                        end=0.0,  # Or unknown
                        text=text_val,
                        words=[],
                        avg_confidence=None,
                        confidence=None,
                        low_confidence=False,
                    )
                )

            proc = time.time() - t0
            logger.info("ASR done (transformers) | segs=%d | proc=%.2fs", len(segs), proc)

            transcript = Transcript(
                model=self.model_name,
                backend="transformers",
                language=language,
                duration=dur,
                processing_time=proc,
                segments=segs,
                revision=revision,
            )

            return add_diarization(transcript, audio_path, diarize, num_speakers)

        except Exception as e:
            if not isinstance(e, TranscriptionError):
                raise TranscriptionError(f"Transcription failed under transformers: {e}") from e
            raise
        finally:
            prep.cleanup()
