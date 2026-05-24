from __future__ import annotations

import logging
import time
from typing import Any

# Optional imports; we will guard usage
try:
    from faster_whisper import WhisperModel  # type: ignore

    _HAS_FASTER = True
except Exception:
    WhisperModel = None  # type: ignore
    _HAS_FASTER = False

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


_MODEL_ALIASES = {
    "kb-whisper-large": "KBLab/kb-whisper-large",
    "large-v3": "openai/whisper-large-v3",
}

# Available KB-Whisper revisions (KBLab releases, May 2025)
# - "standard": Default, good general-purpose transcription
# - "strict":   Verbatim transcription – recommended for call center (preserves filler words, repetitions)
# - "subtitle": Better readability, punctuation, casing – recommended for subtitles/display
KB_REVISIONS = {"standard", "strict", "subtitle"}


def _resolve_model_name(name: str) -> str:
    if not name:
        return _MODEL_ALIASES["kb-whisper-large"]
    key = name.strip()
    return _MODEL_ALIASES.get(key, key)


def _normalize_device(device: str | None = "auto") -> tuple[str, int | None]:
    """Return (backend_device_string, cuda_index or None)."""
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return "cuda", 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", None
        return "cpu", None
    d = str(device).lower()
    if d.startswith("cuda"):
        idx = 0
        if ":" in d:
            try:
                idx = int(d.split(":", 1)[1])
            except Exception:
                idx = 0
        return "cuda", idx
    if d == "mps":
        return "mps", None
    return "cpu", None


def _add_diarization(
    result: dict[str, Any],
    audio_path: str,
    diarize: bool,
    num_speakers: int | None,
) -> dict[str, Any]:
    """Add speaker diarization to a transcript result if requested."""
    if not diarize:
        return result

    try:
        from .diarization import DiarizationPipeline

        dp = DiarizationPipeline(backend="heuristic")
        diar_result = dp.diarize(audio_path, num_speakers=num_speakers)
        segments = result.get("segments", [])
        result["segments"] = dp.assign_speakers_to_segments(segments, diar_result)
        result["diarization"] = diar_result.to_dict()
        logger.info(
            "Diarization added | speakers=%d | segments=%d", diar_result.num_speakers, len(segments)
        )
    except Exception as e:
        logger.warning("Diarization failed: %s. Continuing without speaker labels.", e)
        result["diarization"] = {"error": str(e), "backend": "failed"}

    return result


def _to_transcript(
    segments: list[dict[str, Any]],
    model_name: str,
    backend: str,
    language: str,
    duration: float | None,
    proc_time: float,
    revision: str | None = None,
) -> dict[str, Any]:
    result = {
        "model": model_name,
        "backend": backend,
        "language": language,
        "duration": duration,
        "processing_time": proc_time,
        "segments": segments,
    }
    if revision:
        result["revision"] = revision
    return result


def transcribe(
    audio_path: str,
    model: str = "kb-whisper-large",
    backend: str = "faster",
    device: str = "auto",
    language: str = "sv",
    beam_size: int = 5,
    vad: bool = True,
    word_timestamps: bool = True,
    chunk_length_s: int = 30,
    revision: str | None = None,
    diarize: bool = False,
    num_speakers: int | None = 2,
) -> dict[str, Any]:
    """Transcribe audio and return a normalized transcript dict.

    Args:
        audio_path: Path to the audio file.
        model: Model alias or HuggingFace model ID.
               Aliases: kb-whisper-large, large-v3.
        backend: 'faster' (faster-whisper/ctranslate2) or 'transformers' (HF pipeline).
        device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'.
        language: Language code, e.g. 'sv'.
        beam_size: Beam size for decoding (1-10).
        vad: Enable VAD filter (faster-whisper only).
        word_timestamps: Return word-level timestamps.
        chunk_length_s: Chunk length in seconds (transformers only).
        revision: KB-Whisper revision: 'standard', 'strict', 'subtitle'.
                  'strict' is recommended for call center (verbatim transcription).
        diarize: Run speaker diarization and annotate segments with speaker labels.
        num_speakers: Expected number of speakers for diarization (None = auto).

    Returns:
        Transcript dict with model, backend, language, duration,
        processing_time, segments, optional revision, and optional diarization.
    """
    t0 = time.time()
    model_name = _resolve_model_name(model)
    dev_kind, cuda_idx = _normalize_device(device)

    # Validate revision
    if revision and revision not in KB_REVISIONS:
        logger.warning(
            "Unknown revision '%s', using default. Valid: %s",
            revision,
            sorted(KB_REVISIONS),
        )
        revision = None

    logger.info(
        "ASR start | path=%s | backend=%s | model=%s | revision=%s | device=%s%s | lang=%s | beam=%s | vad=%s | word_ts=%s",
        audio_path,
        backend,
        model_name,
        revision or "default",
        dev_kind,
        f":{cuda_idx}" if dev_kind == "cuda" and cuda_idx is not None else "",
        language,
        beam_size,
        vad,
        word_timestamps,
    )

    if backend == "faster":
        if not _HAS_FASTER:
            raise RuntimeError(
                "faster-whisper not installed; install 'faster-whisper' or use backend='transformers'"
            )
        # Choose compute type
        compute_type = (
            "float16" if dev_kind == "cuda" else ("int8" if dev_kind == "cpu" else "float32")
        )
        logger.debug(
            "Loading faster-whisper model | compute_type=%s | revision=%s",
            compute_type,
            revision or "default",
        )

        # faster-whisper loads from HuggingFace; pass revision if supported
        model_kwargs: dict[str, Any] = {
            "device": dev_kind,
            "device_index": cuda_idx,
            "compute_type": compute_type,
        }
        if revision:
            model_kwargs["revision"] = revision

        wmodel = WhisperModel(model_name, **model_kwargs)
        logger.debug("Model loaded: %s", model_name)

        segments_iter, info = wmodel.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad,
            word_timestamps=word_timestamps,
        )
        logger.info("Transcription started (faster-whisper)")
        segs: list[dict[str, Any]] = []
        dur = getattr(info, "duration", None)
        for seg_count, s in enumerate(segments_iter, start=1):
            words = []
            if word_timestamps and getattr(s, "words", None):
                for w in s.words:
                    words.append(
                        {
                            "start": float(getattr(w, "start", 0.0) or 0.0),
                            "end": float(getattr(w, "end", 0.0) or 0.0),
                            "word": getattr(w, "word", ""),
                            "prob": float(getattr(w, "probability", 0.0) or 0.0),
                        }
                    )
            avg_conf = None
            if words:
                ps = [w.get("prob", 0.0) for w in words]
                avg_conf = float(sum(ps) / max(1, len(ps)))
            segs.append(
                {
                    "start": float(getattr(s, "start", 0.0) or 0.0),
                    "end": float(getattr(s, "end", 0.0) or 0.0),
                    "text": getattr(s, "text", "").strip(),
                    "words": words if word_timestamps else [],
                    "avg_confidence": avg_conf,
                }
            )
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    preview = getattr(s, "text", "").strip()[:80].replace("\n", " ")
                except Exception:
                    preview = ""
                logger.debug(
                    "Segment %d | %s -> %s | '%s'",
                    seg_count,
                    getattr(s, "start", 0.0),
                    getattr(s, "end", 0.0),
                    preview,
                )
        proc = time.time() - t0
        logger.info("ASR done (faster) | segs=%d | audio_dur=%s | proc=%.2fs", len(segs), dur, proc)
        result = _to_transcript(segs, model_name, "faster", language, dur, proc, revision)
        return _add_diarization(result, audio_path, diarize, num_speakers)

    # transformers backend
    logger.debug(
        "Creating transformers ASR pipeline | device=%s",
        f"cuda:{cuda_idx}" if dev_kind == "cuda" else dev_kind,
    )

    pipeline_kwargs: dict[str, Any] = {
        "task": "automatic-speech-recognition",
        "model": model_name,
        "device": (cuda_idx if dev_kind == "cuda" else -1),
    }
    if revision:
        pipeline_kwargs["revision"] = revision

    asr = pipeline(**pipeline_kwargs)

    generate_kwargs: dict[str, Any] = {
        "language": language,
        "task": "transcribe",
        "return_timestamps": "word" if word_timestamps else True,
        "chunk_length_s": chunk_length_s,
    }
    t_call = time.time()
    out = asr(audio_path, **generate_kwargs)
    logger.info("Transcription finished (transformers) | call_time=%.2fs", time.time() - t_call)

    # Output can be dict with 'chunks' for timestamps
    segs: list[dict[str, Any]] = []
    dur = None
    if isinstance(out, dict) and "chunks" in out:
        # Treat chunks as segments
        for ch in out["chunks"]:
            start = float(ch.get("timestamp", [0.0, 0.0])[0] or 0.0)
            end = float(ch.get("timestamp", [0.0, 0.0])[1] or 0.0)
            text = ch.get("text", "").strip()
            words = []
            if word_timestamps and "timestamps" in ch:
                for w in ch["timestamps"]:
                    words.append(
                        {
                            "start": float(w.get("timestamp", [0.0, 0.0])[0] or 0.0),
                            "end": float(w.get("timestamp", [0.0, 0.0])[1] or 0.0),
                            "word": w.get("text", ""),
                            "prob": float(w.get("avg_logprob", 0.0) or 0.0),
                        }
                    )
            avg_conf = None
            if words:
                ps = [w.get("prob", 0.0) for w in words]
                avg_conf = float(sum(ps) / max(1, len(ps)))
            segs.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": words if word_timestamps else [],
                    "avg_confidence": avg_conf,
                }
            )
    else:
        # Single segment fallback
        segs.append(
            {
                "start": 0.0,
                "end": None,
                "text": (out.get("text") if isinstance(out, dict) else str(out)).strip(),
                "words": [],
                "avg_confidence": None,
            }
        )
    proc = time.time() - t0
    logger.info("ASR done (transformers) | segs=%d | proc=%.2fs", len(segs), proc)
    result = _to_transcript(segs, model_name, "transformers", language, dur, proc, revision)
    return _add_diarization(result, audio_path, diarize, num_speakers)
