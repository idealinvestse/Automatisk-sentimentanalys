"""Faster-Whisper transcription backend."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.config import KB_REVISIONS
from ..core.device import normalize_device_for_asr
from ..core.errors import TranscriptionError
from ..core.models import Segment, Transcript, Word
from .base import add_diarization, format_hotwords_for_asr, resolve_model_name, resolve_model_name_for_backend

logger = logging.getLogger(__name__)


class _SkipChunked(Exception):
    """Internal control-flow exception used to jump out of the chunked branch
    into the already-built fallback segments list."""
    pass


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
        self._load_model_name = resolve_model_name_for_backend(model_name, "faster")
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
            "Loading faster-whisper model '%s' (load=%s) | compute_type=%s | revision=%s",
            self.model_name,
            self._load_model_name,
            compute_type,
            revision or "default",
        )

        model_kwargs: dict[str, Any] = {
            "device": dev_kind,
            "compute_type": compute_type,
        }
        if cuda_idx is not None:
            model_kwargs["device_index"] = cuda_idx
        if revision:
            model_kwargs["revision"] = revision

        try:
            self._model = WhisperModel(self._load_model_name, **model_kwargs)
            logger.debug("Model loaded successfully: %s", self._load_model_name)
            return self._model
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load faster-whisper model '{self._load_model_name}': {e}"
            ) from e

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
            "ASR (faster-whisper) start | path=%s | model=%s | revision=%s | device=%s | lang=%s | hotwords=%s | prompt=%s | preprocess=%s",
            audio_path,
            self.model_name,
            revision or "default",
            self.device,
            language,
            bool(hotwords),
            bool(initial_prompt),
            preprocess,
        )

        # Optional preprocessing (high-pass + noise reduction) – Task 1.4
        # Applied to the audio fed to the ASR model. Diarization (if requested)
        # still runs on the original file.
        asr_audio_path = audio_path
        if preprocess:
            try:
                from .preprocess import maybe_preprocess

                asr_audio_path = maybe_preprocess(audio_path, preprocess=True)
                logger.info("Preprocessing enabled – using cleaned audio for ASR")
            except Exception as e:
                logger.warning("Preprocessing failed, falling back to original audio: %s", e)
                asr_audio_path = audio_path

        try:
            wmodel = self._get_model(revision=revision)
            dur = None
            segs: list[Segment] = []

            # ------------------------------------------------------------------
            # Chunking + overlap (Task 1.2)
            # ------------------------------------------------------------------
            # faster-whisper does not chunk the input file by default. For long
            # call recordings (> ~10-15 min) a full decode + decode can cause
            # high memory usage or very slow processing on CPU. We therefore
            # implement explicit chunking (default 30s) with 5s overlap when
            # chunk_length_s is positive and the audio is sufficiently long.
            #
            # Overlap allows the decoder to see context across boundaries
            # (important for correct Swedish compound words and prosody).
            # After all chunks we run a smart merge that prefers the higher-
            # confidence version of any overlapping region and avoids
            # duplicating text in the final segment list.
            #
            # Low-confidence segments (avg word prob < LOW_CONF_THRESHOLD) are
            # flagged. Downstream (lexicon blending) will automatically give
            # them a higher lexicon_weight so that the more trustworthy
            # rule-based signal compensates for uncertain ASR.
            # ------------------------------------------------------------------
            LOW_CONF_THRESHOLD = 0.60
            USE_CHUNKING = chunk_length_s and chunk_length_s > 0
            hotwords_str = format_hotwords_for_asr(hotwords)

            # Fast path: no chunking requested or very short audio
            if not USE_CHUNKING:
                segments_iter, info = wmodel.transcribe(
                    asr_audio_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad,
                    word_timestamps=word_timestamps,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords_str,
                )
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

                    seg = Segment(
                        start=float(getattr(s, "start", 0.0) or 0.0),
                        end=float(getattr(s, "end", 0.0) or 0.0),
                        text=getattr(s, "text", "").strip(),
                        words=words,
                        avg_confidence=avg_conf,
                        confidence=avg_conf,
                        low_confidence=(avg_conf is not None and avg_conf < LOW_CONF_THRESHOLD),
                    )
                    # also mirror into properties for consumers that only look there
                    if seg.low_confidence:
                        seg.properties.setdefault("low_confidence", True)
                        seg.properties.setdefault("confidence", avg_conf)

                    segs.append(seg)

                    if logger.isEnabledFor(logging.DEBUG):
                        preview = getattr(s, "text", "").strip()[:80].replace("\n", " ")
                        logger.debug(
                            "Segment %d | %.2f -> %.2f | '%s'",
                            seg_count,
                            getattr(s, "start", 0.0),
                            getattr(s, "end", 0.0),
                            preview,
                        )
            else:
                # Chunked path with overlap
                logger.info(
                    "Using chunked transcription | chunk_length_s=%d | overlap=5s",
                    chunk_length_s,
                )

                # decode_audio is provided by faster-whisper (no extra dependency)
                try:
                    from faster_whisper.audio import decode_audio  # type: ignore
                except Exception:
                    # Fallback: fall back to non-chunked to avoid hard failure
                    logger.warning("Could not import decode_audio; falling back to full-file transcription.")
                    segments_iter, info = wmodel.transcribe(
                        asr_audio_path,
                        language=language,
                        beam_size=beam_size,
                        vad_filter=vad,
                        word_timestamps=word_timestamps,
                        initial_prompt=initial_prompt,
                        hotwords=hotwords_str,
                    )
                    dur = getattr(info, "duration", None)
                    # (the non-chunked building code is duplicated below for the fallback)
                    for s in segments_iter:
                        # minimal build (same as fast path but without debug)
                        words = []
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
                        avg_conf = float(sum(w.prob for w in words) / max(1, len(words))) if words else None
                        segs.append(
                            Segment(
                                start=float(getattr(s, "start", 0.0) or 0.0),
                                end=float(getattr(s, "end", 0.0) or 0.0),
                                text=getattr(s, "text", "").strip(),
                                words=words,
                                avg_confidence=avg_conf,
                                confidence=avg_conf,
                                low_confidence=(avg_conf is not None and avg_conf < LOW_CONF_THRESHOLD),
                            )
                        )
                    # jump to post-processing
                    raise _SkipChunked  # ugly but keeps control flow simple for the fallback case

                audio = decode_audio(asr_audio_path)
                sr = 16000
                chunk_samples = int(chunk_length_s * sr)
                overlap_samples = int(5 * sr)
                step = max(1, chunk_samples - overlap_samples)

                raw_segments: list[Segment] = []
                pos = 0
                chunk_index = 0
                while pos < len(audio):
                    chunk = audio[pos : pos + chunk_samples]
                    if len(chunk) < sr * 2:  # ignore tiny trailing chunk
                        break

                    chunk_start_time = pos / sr
                    chunk_index += 1

                    # Transcribe the numpy chunk (faster-whisper accepts ndarray)
                    try:
                        segments_iter, _ = wmodel.transcribe(
                            chunk,
                            language=language,
                            beam_size=beam_size,
                            vad_filter=vad,
                            word_timestamps=word_timestamps,
                            initial_prompt=initial_prompt,
                            hotwords=hotwords_str,
                        )
                    except Exception as ce:
                        logger.warning("Chunk %d transcription failed: %s. Skipping chunk.", chunk_index, ce)
                        pos += step
                        continue

                    for s in segments_iter:
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

                        shifted_start = float(getattr(s, "start", 0.0) or 0.0) + chunk_start_time
                        shifted_end = float(getattr(s, "end", 0.0) or 0.0) + chunk_start_time

                        seg = Segment(
                            start=shifted_start,
                            end=shifted_end,
                            text=getattr(s, "text", "").strip(),
                            words=words,
                            avg_confidence=avg_conf,
                            confidence=avg_conf,
                            low_confidence=(avg_conf is not None and avg_conf < LOW_CONF_THRESHOLD),
                        )
                        if seg.low_confidence:
                            seg.properties["low_confidence"] = True
                            seg.properties["confidence"] = avg_conf

                        raw_segments.append(seg)

                    pos += step

                # Smart merge of overlapping segments
                segs = self._merge_overlapping_segments(raw_segments, overlap_seconds=4.0)

                # Approximate duration if we couldn't get it from the model
                if dur is None and segs:
                    dur = max((s.end for s in segs), default=0.0)

                logger.info(
                    "Chunked ASR done | raw_chunks=%d | final_segments=%d",
                    len(raw_segments),
                    len(segs),
                )

            # Common post-processing for both paths
            proc = time.time() - t0
            logger.info(
                "ASR done (faster-whisper) | segs=%d | audio_dur=%s | proc=%.2fs | chunked=%s",
                len(segs),
                dur,
                proc,
                USE_CHUNKING,
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

        except _SkipChunked:
            # Fallback path already built segs
            proc = time.time() - t0
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

    # ------------------------------------------------------------------
    # Helper for Task 1.2
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_overlapping_segments(
        segments: list[Segment], overlap_seconds: float = 4.0
    ) -> list[Segment]:
        """Merge segments that overlap significantly (from chunked transcription).

        Strategy (simple but effective for ASR):
        - Sort by start time.
        - When two consecutive segments overlap by more than `overlap_seconds`
          (or a large fraction of the shorter one), keep the version with the
          higher confidence and extend its time span. This removes duplicate
          text that appears in the overlap region of adjacent chunks.
        - Low-confidence flags are preserved (OR-ed).
        """
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: (s.start, s.end))
        merged: list[Segment] = []
        current = sorted_segs[0]

        for nxt in sorted_segs[1:]:
            overlap = min(current.end, nxt.end) - max(current.start, nxt.start)
            short_dur = min(current.end - current.start, nxt.end - nxt.start) or 1.0
            # Be generous with overlap detection for chunked ASR (5s overlap is common)
            significant_overlap = (overlap > max(1.5, overlap_seconds * 0.6)) or (overlap / short_dur > 0.20)

            if significant_overlap:
                # Prefer the higher-confidence segment for the merged region
                cur_conf = current.avg_confidence or current.confidence or 0.0
                nxt_conf = nxt.avg_confidence or nxt.confidence or 0.0

                if nxt_conf > cur_conf:
                    # switch to nxt as base and extend
                    base = nxt
                    base = Segment(
                        start=min(current.start, nxt.start),
                        end=max(current.end, nxt.end),
                        text=nxt.text,  # the later chunk usually has fresher context
                        words=nxt.words,
                        avg_confidence=max(cur_conf, nxt_conf),
                        confidence=max(cur_conf, nxt_conf),
                        low_confidence=current.low_confidence or nxt.low_confidence,
                        properties={**current.properties, **nxt.properties},
                    )
                    current = base
                else:
                    # extend current
                    current = Segment(
                        start=min(current.start, nxt.start),
                        end=max(current.end, nxt.end),
                        text=current.text,
                        words=current.words,
                        avg_confidence=max(cur_conf, nxt_conf),
                        confidence=max(cur_conf, nxt_conf),
                        low_confidence=current.low_confidence or nxt.low_confidence,
                        properties={**current.properties, **nxt.properties},
                    )
            else:
                merged.append(current)
                current = nxt

        merged.append(current)
        return merged
