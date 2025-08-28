from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

# Optional imports; we will guard usage
try:
    from faster_whisper import WhisperModel  # type: ignore
    _HAS_FASTER = True
except Exception:
    WhisperModel = None  # type: ignore
    _HAS_FASTER = False

from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


_MODEL_ALIASES = {
    "kb-whisper-large": "KBLab/kb-whisper-large",
    "large-v3": "openai/whisper-large-v3",
}


def _resolve_model_name(name: str) -> str:
    if not name:
        return _MODEL_ALIASES["kb-whisper-large"]
    key = name.strip()
    return _MODEL_ALIASES.get(key, key)


def _normalize_device(device: Optional[str] = "auto") -> Tuple[str, Optional[int]]:
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


def _to_transcript(
    segments: List[Dict[str, Any]],
    model_name: str,
    backend: str,
    language: str,
    duration: Optional[float],
    proc_time: float,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "backend": backend,
        "language": language,
        "duration": duration,
        "processing_time": proc_time,
        "segments": segments,
    }


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
) -> Dict[str, Any]:
    """Transcribe audio and return a normalized transcript dict.

    - backend 'faster' uses faster-whisper (ctranslate2) when available
    - backend 'transformers' uses Hugging Face pipeline
    """
    t0 = time.time()
    model_name = _resolve_model_name(model)
    dev_kind, cuda_idx = _normalize_device(device)
    logger.info(
        "ASR start | path=%s | backend=%s | model=%s | device=%s%s | lang=%s | beam=%s | vad=%s | word_ts=%s",
        audio_path,
        backend,
        model_name,
        dev_kind,
        f":{cuda_idx}" if dev_kind == "cuda" and cuda_idx is not None else "",
        language,
        beam_size,
        vad,
        word_timestamps,
    )

    if backend == "faster":
        if not _HAS_FASTER:
            raise RuntimeError("faster-whisper not installed; install 'faster-whisper' or use backend='transformers'")
        # Choose compute type
        compute_type = "float16" if dev_kind == "cuda" else ("int8" if dev_kind == "cpu" else "float32")
        logger.debug("Loading faster-whisper model | compute_type=%s", compute_type)
        wmodel = WhisperModel(model_name, device=dev_kind, device_index=cuda_idx, compute_type=compute_type)
        logger.debug("Model loaded: %s", model_name)
        segments_iter, info = wmodel.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad,
            word_timestamps=word_timestamps,
        )
        logger.info("Transcription started (faster-whisper)")
        segs: List[Dict[str, Any]] = []
        dur = getattr(info, "duration", None)
        seg_count = 0
        for s in segments_iter:
            words = []
            if word_timestamps and getattr(s, "words", None):
                for w in s.words:
                    words.append({
                        "start": float(getattr(w, "start", 0.0) or 0.0),
                        "end": float(getattr(w, "end", 0.0) or 0.0),
                        "word": getattr(w, "word", ""),
                        "prob": float(getattr(w, "probability", 0.0) or 0.0),
                    })
            avg_conf = None
            if words:
                ps = [w.get("prob", 0.0) for w in words]
                avg_conf = float(sum(ps) / max(1, len(ps)))
            segs.append({
                "start": float(getattr(s, "start", 0.0) or 0.0),
                "end": float(getattr(s, "end", 0.0) or 0.0),
                "text": getattr(s, "text", "").strip(),
                "words": words if word_timestamps else [],
                "avg_confidence": avg_conf,
            })
            seg_count += 1
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    preview = getattr(s, "text", "").strip()[:80].replace("\n", " ")
                except Exception:
                    preview = ""
                logger.debug("Segment %d | %s -> %s | '%s'", seg_count, getattr(s, "start", 0.0), getattr(s, "end", 0.0), preview)
        proc = time.time() - t0
        logger.info("ASR done (faster) | segs=%d | audio_dur=%s | proc=%.2fs", len(segs), dur, proc)
        return _to_transcript(segs, model_name, "faster", language, dur, proc)

    # transformers backend
    logger.debug("Creating transformers ASR pipeline | device=%s", f"cuda:{cuda_idx}" if dev_kind == "cuda" else dev_kind)
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=(cuda_idx if dev_kind == "cuda" else -1),
    )
    generate_kwargs = {
        "language": language,
        "task": "transcribe",
        "return_timestamps": "word" if word_timestamps else True,
        "chunk_length_s": chunk_length_s,
    }
    t_call = time.time()
    out = asr(audio_path, **generate_kwargs)
    logger.info("Transcription finished (transformers) | call_time=%.2fs", time.time() - t_call)
    # Output can be dict with 'chunks' for timestamps
    segs: List[Dict[str, Any]] = []
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
                    words.append({
                        "start": float(w.get("timestamp", [0.0, 0.0])[0] or 0.0),
                        "end": float(w.get("timestamp", [0.0, 0.0])[1] or 0.0),
                        "word": w.get("text", ""),
                        "prob": float(w.get("avg_logprob", 0.0) or 0.0),
                    })
            avg_conf = None
            if words:
                ps = [w.get("prob", 0.0) for w in words]
                avg_conf = float(sum(ps) / max(1, len(ps)))
            segs.append({
                "start": start,
                "end": end,
                "text": text,
                "words": words if word_timestamps else [],
                "avg_confidence": avg_conf,
            })
    else:
        # Single segment fallback
        segs.append({
            "start": 0.0,
            "end": None,
            "text": (out.get("text") if isinstance(out, dict) else str(out)).strip(),
            "words": [],
            "avg_confidence": None,
        })
    proc = time.time() - t0
    logger.info("ASR done (transformers) | segs=%d | proc=%.2fs", len(segs), proc)
    return _to_transcript(segs, model_name, "transformers", language, dur, proc)
