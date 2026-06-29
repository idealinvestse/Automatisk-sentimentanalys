"""WhisperX transcription backend.

WhisperX = faster-whisper + Whisper forced alignment (word-level timestamps)
+ optional integrated speaker diarization via pyannote.

This backend is valuable for call center use cases because:
- Superior word-level alignment improves evidence span extraction for
  Aspect-Based Sentiment (ABSA) and root-cause analysis (e.g. "faktureringen
  var fel" -> precise span for fakturering_pris aspect).
- Built-in diarization often gives better speaker boundaries than post-hoc
  heuristic, which is a prerequisite for role inference (agent vs customer)
  and per-role metrics such as agent empathy or customer trajectory.
- Chunking/overlap and batching are handled efficiently inside the pipeline,
  reducing OOM risk on long calls compared to naive long-audio processing.

Implementation follows the exact style of FasterWhisperTranscriber and
TransformersTranscriber:
- lazy loading of heavy models (main ASR + align + diarize)
- @lru_cache via factory for the Transcriber instance itself
- full type hints
- TranscriptionError wrapper
- detailed logging with context
- identical transcribe(...) signature (so CLI / pipeline / API are unchanged)
- graceful fallback to post-diarization (add_diarization) when whisperx
  diarization cannot be used (missing HF token etc.)

The default model for whisperx is mapped from KB aliases to "large-v3"
because WhisperX alignment models are trained against the original OpenAI
Whisper checkpoints. KB-Whisper can still be used via the "faster" or
"transformers" backends (recommended for highest Swedish WER).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ..core.device import normalize_device_for_asr
from ..core.errors import TranscriptionError
from ..core.models import Segment, Transcript, Word
from .base import add_diarization, format_hotwords_for_asr, resolve_model_name_for_backend

logger = logging.getLogger(__name__)

try:
    import whisperx  # type: ignore

    _HAS_WHISPERX = True
except Exception:
    whisperx = None  # type: ignore
    _HAS_WHISPERX = False


class WhisperXTranscriber:
    """ASR transcriber powered by WhisperX.

    Provides high-quality word alignment and optional end-to-end diarization.
    Use via backend="whisperx" in get_transcriber / CLI / pipeline.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if not _HAS_WHISPERX:
            raise ImportError(
                "whisperx is not installed. Install it with 'pip install whisperx' "
                "(this also pulls faster-whisper) or choose another backend (faster/transformers)."
            )
        self.model_name = self._resolve_for_whisperx(model_name)
        self.device = device
        self._model: Any = None
        self._align_model: Any = None
        self._align_metadata: Any = None
        self._diarize_model: Any = None

    def _resolve_for_whisperx(self, name: str) -> str:
        """Map common aliases to a WhisperX-friendly checkpoint name."""
        resolved = resolve_model_name_for_backend(name, "whisperx")
        lower = resolved.lower()
        if "kb-whisper" in lower or "kblab" in lower:
            logger.info(
                "WhisperX backend received KB-Whisper alias '%s' -> mapping to 'large-v3' "
                "(WhisperX alignment targets original Whisper checkpoints; "
                "use backend=faster for pure KB-Whisper WER advantage on Swedish).",
                resolved,
            )
            return "large-v3"
        return resolved

    def _get_device_str(self) -> str:
        """Return device string in the form whisperx expects (cuda:0, cpu, mps)."""
        dev_kind, cuda_idx = normalize_device_for_asr(self.device)
        if dev_kind == "cuda":
            return f"cuda:{cuda_idx or 0}"
        if dev_kind == "mps":
            return "mps"
        return "cpu"

    def _get_compute_type(self) -> str:
        dev_kind, _ = normalize_device_for_asr(self.device)
        if dev_kind == "cuda":
            return "float16"
        if dev_kind == "cpu":
            return "int8"
        return "float32"

    def _get_model(self, revision: str | None = None) -> Any:
        """Lazy-load (and cache) the main WhisperX ASR model."""
        if self._model is not None:
            return self._model

        device_str = self._get_device_str()
        compute_type = self._get_compute_type()

        logger.debug(
            "Loading whisperx ASR model '%s' | device=%s | compute_type=%s | revision=%s",
            self.model_name,
            device_str,
            compute_type,
            revision or "default",
        )

        try:
            # WhisperX load_model signature: (whisper_arch, device, compute_type, ...)
            # revision is rarely used for the main whisper model; we log and ignore for now
            if revision:
                logger.debug(
                    "WhisperX backend received revision=%s (passed to load if supported)", revision
                )

            self._model = whisperx.load_model(
                self.model_name,
                device_str,
                compute_type=compute_type,
                # We deliberately do not force language here; it is supplied at transcribe time
            )
            logger.debug("WhisperX ASR model loaded successfully: %s", self.model_name)
            return self._model
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load whisperx model '{self.model_name}': {e}"
            ) from e

    def _get_align_model(self, language: str) -> tuple[Any | None, Any | None]:
        """Lazy-load the language-specific alignment model (for word timestamps)."""
        if self._align_model is not None and self._align_metadata is not None:
            return self._align_model, self._align_metadata

        device_str = self._get_device_str()
        logger.debug("Loading whisperx alignment model for language=%s on %s", language, device_str)

        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=device_str
            )
            self._align_model = align_model
            self._align_metadata = metadata
            return align_model, metadata
        except Exception as e:
            logger.warning(
                "Failed to load whisperx align model for '%s': %s. "
                "Continuing with coarser timestamps from the base model. "
                "Word-level evidence for ABSA/trajectory will be less precise.",
                language,
                e,
            )
            self._align_model = None
            self._align_metadata = None
            return None, None

    def _get_diarize_model(self) -> Any | None:
        """Lazy-load WhisperX diarization pipeline (requires HF token for pyannote)."""
        if self._diarize_model is not None:
            return self._diarize_model

        device_str = self._get_device_str()
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        if not hf_token:
            logger.info(
                "No HF_TOKEN found in environment. WhisperX diarization will be unavailable "
                "(pyannote/speaker-diarization-3.1 requires authentication). "
                "Falling back to post-hoc diarization (heuristic or pyannote) for --diarize."
            )
            return None

        logger.debug("Initializing WhisperX diarization pipeline on %s", device_str)
        try:
            self._diarize_model = whisperx.DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
                device=device_str,
            )
            logger.info("WhisperX diarization pipeline ready")
            return self._diarize_model
        except Exception as e:
            logger.warning(
                "WhisperX diarization initialization failed: %s. "
                "This is common without a valid HF token with gated access. "
                "Post-processing diarization will be used instead.",
                e,
            )
            self._diarize_model = None
            return None

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
        preprocess_mode: str | None = None,
    ) -> Transcript:
        """Transcribe audio using WhisperX (with optional alignment + diarization).

        The implementation deliberately keeps the exact same public signature
        as the other backends so that the rest of the system (CLI, API, pipeline,
        CallAnalysisReport) requires no signature changes.
        """
        t0 = time.time()

        if revision and revision not in {"standard", "strict", "subtitle"}:
            # We still accept it for interface parity but whisperx rarely uses KB revisions.
            logger.debug(
                "WhisperX received revision=%s (will be ignored; not a KB model)", revision
            )

        logger.info(
            "ASR (whisperx) start | path=%s | model=%s | revision=%s | device=%s | lang=%s | diarize=%s | hotwords=%s | prompt=%s | preprocess=%s | preprocess_mode=%s",
            audio_path,
            self.model_name,
            revision or "default",
            self.device,
            language,
            diarize,
            bool(hotwords),
            bool(initial_prompt),
            preprocess,
            preprocess_mode,
        )

        from .preprocess import prepare_asr_audio
        from .vad_callcenter import vad_options_for_mode

        prep, resolved_preprocess_mode = prepare_asr_audio(
            audio_path,
            preprocess=preprocess,
            preprocess_mode=preprocess_mode,
        )
        vad_parameters = vad_options_for_mode(resolved_preprocess_mode, vad_enabled=vad)

        asr_audio_path = prep.path
        try:
            wmodel = self._get_model(revision=revision)
            audio = whisperx.load_audio(asr_audio_path)

            # WhisperX transcribe call. batch_size chosen for reasonable VRAM usage.
            # vad_filter is applied inside the underlying faster-whisper call when possible.
            transcribe_kwargs: dict[str, Any] = {
                "batch_size": 16,
                "language": language,
            }
            # beam_size is supported via the inner whisper decoding in recent whisperx
            if beam_size and beam_size != 5:
                transcribe_kwargs["beam_size"] = beam_size

            # Hotwords (list or str) and initial_prompt are forwarded to the underlying
            # faster-whisper model. This is the primary way to improve WER on domain
            # terms for call center (e.g. "fakturering", company names).
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt
            hotwords_str = format_hotwords_for_asr(hotwords)
            if hotwords_str:
                transcribe_kwargs["hotwords"] = hotwords_str

            # Some versions expose vad_filter; we pass it defensively
            if not vad:
                transcribe_kwargs["vad_filter"] = False
            elif vad_parameters is not None:
                transcribe_kwargs["vad_parameters"] = vad_parameters

            result: dict[str, Any] = wmodel.transcribe(audio, **transcribe_kwargs)

            # --- Forced alignment for accurate word timestamps (key for call center evidence) ---
            if word_timestamps:
                align_model, metadata = self._get_align_model(language)
                if align_model is not None and metadata is not None:
                    try:
                        result = whisperx.align(
                            result["segments"],
                            align_model,
                            metadata,
                            audio,
                            device=self._get_device_str(),
                            return_char_alignments=False,
                        )
                    except Exception as ae:
                        logger.warning(
                            "WhisperX align step failed: %s. Using unaligned segments.", ae
                        )
                else:
                    logger.debug(
                        "No alignment model available; using base transcription timestamps."
                    )

            # Convert WhisperX segments (dicts) into our Segment / Word dataclasses
            segments_raw: list[dict[str, Any]] = result.get("segments", []) or []
            segs: list[Segment] = []
            for s in segments_raw:
                words: list[Word] = []
                raw_words = s.get("words") or []
                for w in raw_words:
                    # WhisperX word dicts use "word", "start", "end", "score" (or "probability")
                    words.append(
                        Word(
                            start=float(w.get("start", 0.0) or 0.0),
                            end=float(w.get("end", 0.0) or 0.0),
                            word=str(w.get("word", w.get("text", ""))).strip(),
                            prob=float(w.get("score", w.get("probability", 0.0)) or 0.0),
                        )
                    )

                avg_conf: float | None = None
                if words:
                    avg_conf = sum(w.prob for w in words) / max(1, len(words))

                low_conf = avg_conf is not None and avg_conf < 0.60
                seg = Segment(
                    start=float(s.get("start", 0.0) or 0.0),
                    end=float(s.get("end", 0.0) or 0.0),
                    text=str(s.get("text", "")).strip(),
                    words=words,
                    speaker=s.get("speaker"),  # may be present if diarization ran inside whisperx
                    avg_confidence=avg_conf,
                    confidence=avg_conf,
                    low_confidence=low_conf,
                    properties={"backend": "whisperx"},
                )
                segs.append(seg)

            proc = time.time() - t0
            logger.info(
                "ASR done (whisperx) | segs=%d | proc=%.2fs | aligned=%s",
                len(segs),
                proc,
                word_timestamps,
            )

            transcript = Transcript(
                model=self.model_name,
                backend="whisperx",
                language=language,
                duration=result.get("duration"),
                processing_time=proc,
                segments=segs,
                revision=revision,
            )

            # --- Integrated diarization (preferred path when available) ---
            used_internal_diar = False
            if diarize:
                diar_model = self._get_diarize_model()
                if diar_model is not None:
                    try:
                        t_d = time.time()
                        # Run diarization on the raw audio
                        diarize_segments = diar_model(
                            audio,
                            min_speakers=num_speakers or 1,
                            max_speakers=num_speakers or 10,
                        )
                        # Merge speaker labels back into the result (updates words + segments)
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                        # Rebuild segments so that speaker labels are reflected in our model objects
                        segments_raw = result.get("segments", segments_raw)
                        segs = []
                        for s in segments_raw:
                            words = []
                            for w in s.get("words") or []:
                                words.append(
                                    Word(
                                        start=float(w.get("start", 0.0) or 0.0),
                                        end=float(w.get("end", 0.0) or 0.0),
                                        word=str(w.get("word", w.get("text", ""))).strip(),
                                        prob=float(
                                            w.get("score", w.get("probability", 0.0)) or 0.0
                                        ),
                                    )
                                )
                            avg_conf = (
                                sum(w.prob for w in words) / max(1, len(words)) if words else None
                            )
                            low_conf = avg_conf is not None and avg_conf < 0.60
                            segs.append(
                                Segment(
                                    start=float(s.get("start", 0.0) or 0.0),
                                    end=float(s.get("end", 0.0) or 0.0),
                                    text=str(s.get("text", "")).strip(),
                                    words=words,
                                    speaker=s.get("speaker"),
                                    avg_confidence=avg_conf,
                                    confidence=avg_conf,
                                    low_confidence=low_conf,
                                    properties={"backend": "whisperx"},
                                )
                            )

                        speakers = sorted({s.speaker for s in segs if s.speaker})
                        transcript = Transcript(
                            model=transcript.model,
                            backend=transcript.backend,
                            language=transcript.language,
                            duration=transcript.duration,
                            processing_time=transcript.processing_time,
                            segments=segs,
                            revision=transcript.revision,
                            diarization={
                                "backend": "whisperx",
                                "num_speakers": len(speakers),
                                "speakers": speakers,
                                "processing_time_s": round(time.time() - t_d, 2),
                            },
                        )
                        used_internal_diar = True
                        logger.info(
                            "WhisperX diarization completed | speakers=%s",
                            speakers,
                        )
                    except Exception as de:
                        logger.warning(
                            "WhisperX assign_word_speakers failed: %s. "
                            "Falling back to external diarization pipeline.",
                            de,
                        )

            # If caller asked for diarization but we did not succeed with internal whisperx diar,
            # fall back to the project's existing add_diarization (heuristic or pyannote).
            # This preserves the exact same behavior and output shape for downstream code.
            if diarize and not used_internal_diar:
                transcript = add_diarization(
                    transcript, audio_path, diarize=True, num_speakers=num_speakers
                )

            return transcript

        except Exception as e:
            if not isinstance(e, TranscriptionError):
                raise TranscriptionError(f"Transcription failed under whisperx: {e}") from e
            raise
        finally:
            prep.cleanup()
