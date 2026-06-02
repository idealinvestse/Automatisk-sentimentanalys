from __future__ import annotations

import glob
import json
import logging
import os
import threading

# Concurrency utilities for parallel processing of audio files
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

# Standard library imports for type hints, file operations, and JSON handling
from typing import Any

# FastAPI framework for building the REST API
from fastapi import FastAPI

# Pydantic for data validation and serialization
from pydantic import BaseModel, Field

from .core.config import AUDIO_EXTS
from .lexicon import (
    blend_distributions,
    load_lexicon,
    scalar_to_dist,
    score_text,
)

# Internal sentiment/ASR/lexicon modules
from .sentiment import analyze_smart
from .transcription import get_transcriber

logger = logging.getLogger(__name__)

# Initialize FastAPI app
# NOTE: This monolithic module is deprecated in favour of the modular src/api/ package.
# Use `uvicorn src.api:app` to run the up-to-date API (src/api/app.py, version 0.3.0).
app = FastAPI(title="Swedish Sentiment API [DEPRECATED]", version="0.3.0")


def _utc_now_iso(trim_microseconds: bool = True) -> str:
    """Return a UTC ISO timestamp with a trailing Z."""
    dt = datetime.now(UTC)
    if trim_microseconds:
        dt = dt.replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _score_dict(entries: Any) -> dict[str, float]:
    """Convert sentiment entries into a safe fixed-label score mapping."""
    scores = {"negativ": 0.0, "neutral": 0.0, "positiv": 0.0}
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return scores
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if label not in scores:
            continue
        try:
            scores[label] = float(entry.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid sentiment score for label %s: %r", label, entry)
    return scores


def _single_label_distribution(result: Any) -> dict[str, float]:
    scores = _score_dict(result)
    if any(scores.values()):
        return scores
    if isinstance(result, dict):
        label = result.get("label")
        if label in scores:
            scores[label] = 1.0
    return scores


def _blend_with_lexicon(
    texts: list[str],
    results: list[Any],
    lexicon_file: str | None,
    lexicon_weight: float,
) -> list[Any]:
    """Blend model sentiment outputs with lexicon scores when configured."""
    if not lexicon_file or lexicon_weight <= 0.0:
        return results
    if len(texts) != len(results):
        logger.warning(
            "Lexicon blending length mismatch: texts=%d results=%d", len(texts), len(results)
        )
    try:
        lex = load_lexicon(lexicon_file)
        blended_results: list[Any] = []
        full_distribution = bool(results and isinstance(results[0], list))
        for text, result in zip(texts, results, strict=False):
            scores = (
                _score_dict(result) if full_distribution else _single_label_distribution(result)
            )
            lex_dist = scalar_to_dist(score_text(text, lex))
            scores = blend_distributions(scores, lex_dist, lexicon_weight)
            if full_distribution:
                blended_results.append(
                    [
                        {"label": "negativ", "score": scores["negativ"]},
                        {"label": "neutral", "score": scores["neutral"]},
                        {"label": "positiv", "score": scores["positiv"]},
                    ]
                )
            else:
                top_label = max(scores.items(), key=lambda kv: kv[1])[0]
                blended_results.append({"label": top_label, "score": float(scores[top_label])})
        return blended_results
    except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
        logger.warning("Lexicon blending failed for %s: %s", lexicon_file, e, exc_info=True)
        return results


def _segment_time(segment: dict[str, Any], key: str) -> float | None:
    value = segment.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _map_results_to_segments(
    texts: list[str],
    results: list[Any],
    segments: list[dict[str, Any]],
) -> list[SegmentSentiment]:
    """Map sentiment distributions back onto transcript segments."""
    if len(texts) != len(results):
        logger.warning(
            "Segment sentiment length mismatch: texts=%d results=%d", len(texts), len(results)
        )
    seg_out: list[SegmentSentiment] = []
    for idx, (text, result) in enumerate(zip(texts, results, strict=False)):
        scores_map = _score_dict(result)
        top_label = max(scores_map.items(), key=lambda kv: kv[1])[0]
        top_score = float(scores_map[top_label])
        segment = segments[idx] if idx < len(segments) else {}
        seg_out.append(
            SegmentSentiment(
                index=idx,
                start=_segment_time(segment, "start"),
                end=_segment_time(segment, "end"),
                text=text,
                label=top_label,
                score=top_score,
                negativ=scores_map.get("negativ"),
                neutral=scores_map.get("neutral"),
                positiv=scores_map.get("positiv"),
            )
        )
    return seg_out


def _texts_from_segments(segments: list[dict[str, Any]]) -> list[str]:
    texts = [s.get("text", "").strip() for s in segments]
    if texts and any(texts):
        return texts
    joined = " ".join(s.get("text", "").strip() for s in segments if s.get("text")).strip()
    return [joined] if joined else []


def _transcribe_helper(
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
    num_speakers: int | None = None,
) -> dict[str, Any]:
    """Helper function to run ASR transcription using the new modular transcription package."""
    transcriber = get_transcriber(
        backend=backend,
        model_name=model,
        device=device,
    )
    transcript = transcriber.transcribe(
        audio_path=audio_path,
        language=language,
        beam_size=beam_size,
        vad=vad,
        word_timestamps=word_timestamps,
        chunk_length_s=chunk_length_s,
        revision=revision,
        diarize=diarize,
        num_speakers=num_speakers,
    )
    return transcript.to_dict()


class AnalyzeRequest(BaseModel):
    # List of texts to analyze for sentiment
    texts: list[str] = Field(..., description="List of texts to analyze")
    # Data type classification for profile selection
    datatype: str | None = Field(
        None,
        description=("Data type: post, comment, article, review, ..."),
    )
    # Source classification for profile selection
    source: str | None = Field(
        None,
        description=("Source: forum, magazine, news, social, ..."),
    )
    # Explicit profile name to use for sentiment analysis
    profile: str | None = Field(
        None,
        description="Explicit profile name to use",
    )
    # Optional model override for sentiment analysis
    model: str | None = Field(
        None,
        description="Optional model override",
    )
    # Device specification for model execution
    device: str | None = Field(
        "auto",
        description="Device: auto, cpu, cuda, cuda:0, mps",
    )
    # Batch size for processing multiple texts efficiently
    batch_size: int = Field(16, ge=1, le=128)
    # Whether to return all scores or just the top prediction
    return_all_scores: bool = Field(False)
    # Maximum text length for processing (truncates longer texts)
    max_length: int | None = Field(None, ge=8, le=4096)
    # Whether to clean text before analysis
    # (HTML unescaping, URL removal, etc.)
    clean: bool = Field(True)
    # Whether to normalize text (case, whitespace, etc.)
    normalize: bool = Field(True)
    # Path to Swedish lexicon file for blending with model predictions
    lexicon_file: str | None = Field(
        None,
        description=(
            "Path to Swedish lexicon (CSV/TSV) with columns term|word and polarity|score|sentiment"
        ),
    )
    # Blend weight for lexicon distribution [0..1]
    # where 0 = model only, 1 = lexicon only
    lexicon_weight: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Blend weight [0..1] for lexicon distribution",
    )


class AnalyzeResponse(BaseModel):
    meta: dict[str, Any]
    timestamp: str
    results: list[Any]


class TranscribeRequest(BaseModel):
    # Path to audio file (server/container must have access)
    audio_path: str = Field(
        ...,
        description=("Path to audio file accessible by the server/container"),
    )
    # ASR model to use for transcription (KB Swedish large by default)
    model: str = Field("kb-whisper-large")
    # Backend implementation (faster-whisper or transformers)
    backend: str = Field("faster", description="faster | transformers")
    # Device specification for model execution
    device: str = Field("auto")
    # Language code for transcription (sv = Swedish)
    language: str = Field("sv")
    # Beam size for decoding (higher = slower, maybe more accurate)
    beam_size: int = Field(5, ge=1, le=10)
    # Use Voice Activity Detection to split audio
    vad: bool = Field(True)
    # Include word-level timestamps in the output
    word_timestamps: bool = Field(True)
    # Length of audio chunks in seconds for processing
    # (affects memory usage and accuracy)
    chunk_length_s: int = Field(30, ge=5, le=60)
    # KB-Whisper revision: standard, strict, subtitle
    # 'strict' is recommended for call center (verbatim transcription)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers (None=auto)")


class TranscribeResponse(BaseModel):
    transcript: dict[str, Any]
    timestamp: str


class AnalyzeConversationRequest(BaseModel):
    # Path to audio file the server can access
    audio_path: str = Field(
        ...,
        description=("Path to audio file accessible by the server/container"),
    )
    # ASR parameters
    # ASR model to use (KB Swedish large by default)
    model: str = Field("kb-whisper-large")
    # Backend implementation (faster-whisper or transformers)
    backend: str = Field("faster")
    # Device specification for ASR execution
    device: str = Field("auto")
    # Language code (sv = Swedish)
    language: str = Field("sv")
    # Beam size (higher = slower, potentially more accurate)
    beam_size: int = Field(5, ge=1, le=10)
    # Use VAD to split audio into segments
    vad: bool = Field(True)
    # Include word-level timestamps in output
    word_timestamps: bool = Field(False)
    # Chunk length in seconds
    chunk_length_s: int = Field(30, ge=5, le=60)
    # KB-Whisper revision: standard, strict, subtitle
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")
    # Sentiment analysis parameters
    # Whether to return all scores or only the top prediction
    return_all_scores: bool = Field(True)
    # Optional model override for sentiment analysis
    sentiment_model: str | None = Field(
        None,
        description="Optional override for sentiment model",
    )
    # Path to Swedish lexicon file for blending with model predictions
    lexicon_file: str | None = Field(None)
    # Blend weight [0..1]; 0=model only, 1=lexicon only
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class SegmentSentiment(BaseModel):
    # Index of the segment in the conversation
    index: int
    # Start time of the segment in seconds
    start: float | None
    # End time of the segment in seconds
    end: float | None
    # Transcribed text of the segment
    text: str
    # Predicted sentiment label (negativ, neutral, or positiv)
    label: str
    # Confidence score for the predicted label
    score: float
    # Individual sentiment scores when return_all_scores=True
    negativ: float | None = None
    neutral: float | None = None
    positiv: float | None = None
    intent: str | None = None
    intent_confidence: float | None = None


class AnalyzeConversationResponse(BaseModel):
    # Full transcription result from ASR
    transcript: dict[str, Any]
    # Sentiment analysis results for each conversation segment
    segment_sentiments: list[SegmentSentiment]
    # Metadata about the analysis process (model used, device, timing, etc.)
    meta: dict[str, Any]
    # ISO timestamp when the analysis was completed
    timestamp: str


# --- Helpers for batch/scan ---


def _resolve_audio_paths(
    audio_paths: list[str] | None = None,
    directory: str | None = None,
    pattern: str | None = None,
    recursive: bool = True,
) -> list[str]:
    """Resolve audio file paths from various input sources.

    This helper function handles multiple ways of specifying audio files:
    1. Explicit file paths in a list
    2. Glob patterns in the paths list
    3. Directory traversal with optional pattern matching

    Args:
        audio_paths: List of explicit file paths or glob patterns
        directory: Base directory to scan for audio files
        pattern: Glob pattern to match files within directory
        recursive: Whether to scan subdirectories recursively

    Returns:
        Sorted list of absolute paths to valid audio files
        with deduplicated entries
    """
    files: list[str] = []
    # Process explicit paths: files, directories, or glob patterns
    for p in audio_paths or []:
        # Handle glob patterns (*, ?, []) in the path list
        if any(ch in p for ch in ["*", "?", "["]):
            for m in glob.glob(p, recursive=recursive):
                # Validate that matched item is a file with supported extension
                if os.path.isfile(m) and os.path.splitext(m)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.abspath(m))
        # Handle directory paths in the path list
        elif os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    # Check if file has supported audio extension
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(os.path.join(root, fn)))
                # Break after first level if not recursive
                if not recursive:
                    break
        # Handle individual file paths in the path list
        elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in AUDIO_EXTS:
            files.append(os.path.abspath(p))
    # Process directory + pattern combination if directory is specified
    if directory:
        if pattern:
            # Create full path pattern and match files
            pat = os.path.join(directory, pattern)
            for m in glob.glob(pat, recursive=recursive):
                # Validate file has supported extension
                if os.path.isfile(m) and os.path.splitext(m)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.abspath(m))
        else:
            # Scan entire directory for audio files
            for root, _, fnames in os.walk(directory):
                for fn in fnames:
                    # Check if file has supported audio extension
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(os.path.join(root, fn)))
                # Break after first level if not recursive
                if not recursive:
                    break
    # Remove duplicates and sort alphabetically for consistent processing order
    files = sorted(dict.fromkeys(files))
    return files


def _chunk(lst: list[str], size: int) -> list[list[str]]:
    """Divide a list into chunks of specified size.

    This helper function is used for batch processing to divide files into
    manageable groups for parallel processing.

    Args:
        lst: List to be chunked
        size: Maximum size of each chunk

    Returns:
        List of sublists (chunks) of the original list
    """
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _load_state(path: str | None) -> dict[str, Any]:
    """Load processing state from a JSON file.

    This helper is used by the scan process endpoint to track which files
    have already been processed, enabling incremental processing.

    Args:
        path: Path to the state file (can be None)

    Returns:
        Dictionary with processed file info,
        or an empty dict if no state
    """
    # Return empty state if no path provided
    if not path:
        return {"processed": {}}
    # Return empty state if path doesn't point to a valid file
    if not os.path.isfile(path):
        return {"processed": {}}
    try:
        # Load and validate state file structure
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
            # Ensure the loaded object has the expected structure
            if isinstance(obj, dict) and "processed" in obj and isinstance(obj["processed"], dict):
                return obj
    except (OSError, json.JSONDecodeError):
        # Return empty state if file reading or parsing fails
        pass
    return {"processed": {}}


def _save_state(path: str | None, state: dict[str, Any]) -> None:
    """Save processing state to a JSON file.

    This helper persists the processing state to enable incremental
    processing in subsequent runs.

    Args:
        path: Path to the state file (can be None)
        state: Dictionary containing processed file information
    """
    # Do nothing if no path provided
    if not path:
        return
    # Create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Write state to file with proper encoding and formatting
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# Health check endpoint for monitoring service availability
@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint.

    Returns a basic status response to indicate the API is running.
    Useful for monitoring and container orchestration.

    Returns:
        Dict[str, str]: Simple status response
    """
    return {"status": "ok"}


# Main endpoint for analyzing sentiment of text inputs
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze sentiment of Swedish texts.

    This endpoint processes text inputs through the sentiment analysis
    pipeline, supporting profile-based analysis and optional lexicon
    blending for improved accuracy with domain-specific language.

    Args:
        req: AnalyzeRequest containing texts and processing parameters

    Returns:
        AnalyzeResponse with sentiment results, metadata, and timestamp
    """
    # Run sentiment analysis using the smart analyzer with profile resolution
    results, meta = analyze_smart(
        texts=req.texts,
        datatype=req.datatype,
        source=req.source,
        profile=req.profile,
        model_name=req.model,
        device=req.device,
        batch_size=req.batch_size,
        normalize=req.normalize,
        return_all_scores=req.return_all_scores,
        max_length=req.max_length,
        clean=req.clean,
    )
    results = _blend_with_lexicon(req.texts, results, req.lexicon_file, req.lexicon_weight)
    # Generate timestamp for response
    now_iso = _utc_now_iso()
    return AnalyzeResponse(meta=meta, timestamp=now_iso, results=results)


# Endpoint for transcribing audio files to text
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """Transcribe an audio file using ASR.

    This endpoint converts speech in audio files to text using either the
    faster-whisper or Hugging Face transformers backend.

    Args:
        req: TranscribeRequest containing audio path and ASR parameters

    Returns:
        TranscribeResponse with transcription results and timestamp
    """
    # Run ASR transcription with specified parameters
    tr = _transcribe_helper(
        audio_path=req.audio_path,
        model=req.model,
        backend=req.backend,
        device=req.device,
        language=req.language,
        beam_size=req.beam_size,
        vad=req.vad,
        word_timestamps=req.word_timestamps,
        chunk_length_s=req.chunk_length_s,
        revision=req.revision,
        diarize=req.diarize,
        num_speakers=req.num_speakers,
    )
    # Generate timestamp for response
    now_iso = _utc_now_iso()
    return TranscribeResponse(transcript=tr, timestamp=now_iso)


# Endpoint for analyzing sentiment of conversation segments
@app.post("/analyze_conversation", response_model=AnalyzeConversationResponse)
async def analyze_conversation(
    req: AnalyzeConversationRequest,
) -> AnalyzeConversationResponse:
    """Transcribe a call and run sentiment per segment.

    This uses the specialized 'call' profile. It combines ASR and
    sentiment analysis for conversation audio files. It transcribes the
    audio, splits it into segments, and analyzes sentiment for each
    segment.

    Args:
        req: AnalyzeConversationRequest containing audio path and
            processing parameters

    Returns:
        AnalyzeConversationResponse with transcription, segment sentiments,
        metadata, and timestamp
    """
    # Transcribe the conversation audio file
    tr = _transcribe_helper(
        audio_path=req.audio_path,
        model=req.model,
        backend=req.backend,
        device=req.device,
        language=req.language,
        beam_size=req.beam_size,
        vad=req.vad,
        word_timestamps=req.word_timestamps,
        chunk_length_s=req.chunk_length_s,
        revision=req.revision,
        diarize=req.diarize,
        num_speakers=req.num_speakers,
    )
    # Extract text segments from transcription
    segments = tr.get("segments", []) or []
    tr_texts = _texts_from_segments(segments)

    # Run sentiment analysis on each segment using the 'call' profile
    # The 'call' profile is optimized for conversation analysis
    results, meta = analyze_smart(
        tr_texts,
        profile="call",
        model_name=req.sentiment_model,
        device=req.device,
        batch_size=16,
        normalize=True,
        return_all_scores=True,
        max_length=None,
        clean=True,
    )
    results = _blend_with_lexicon(tr_texts, results, req.lexicon_file, req.lexicon_weight)
    seg_out = _map_results_to_segments(tr_texts, results, segments)

    # Build response
    now_iso = _utc_now_iso()
    return AnalyzeConversationResponse(
        transcript=tr,
        segment_sentiments=seg_out,
        meta=meta,
        timestamp=now_iso,
    )


# --- Full pipeline endpoint ---


class PipelineRequest(BaseModel):
    """Request for the full call analysis pipeline."""

    segments: list[dict[str, Any]] = Field(
        ...,
        description="ASR segments with 'text' and optionally 'speaker' keys",
    )
    sentiment_model: str | None = Field(
        None,
        description="Optional sentiment model override",
    )
    device: str = Field("auto")


class PipelineResponse(BaseModel):
    """Response from the full call analysis pipeline."""

    sentiment_results: list[dict[str, Any]]
    intent_results: list[dict[str, Any]]
    summary: dict[str, Any]
    topics: dict[str, Any]
    insights: dict[str, Any]
    risks: dict[str, Any]
    processing_time_s: float
    timestamp: str


@app.post("/analyze_pipeline", response_model=PipelineResponse)
async def analyze_pipeline(req: PipelineRequest) -> PipelineResponse:
    """Run the full call analysis pipeline on pre-transcribed segments.

    This endpoint orchestrates sentiment, intent, summarization,
    topic modeling, insights, and predictive analytics in a single call.

    Args:
        req: PipelineRequest with ASR segments.

    Returns:
        PipelineResponse with complete analysis results.
    """
    from .pipeline import CallAnalysisPipeline

    pipe = CallAnalysisPipeline(
        sentiment_model=req.sentiment_model or "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=req.device,
    )
    report = pipe.analyze_segments(req.segments)

    now_iso = _utc_now_iso()
    return PipelineResponse(
        sentiment_results=report.sentiment_results,
        intent_results=[{"intent": i, "confidence": round(c, 3)} for i, c in report.intent_results],
        summary=report.summary,
        topics=report.topics,
        insights=report.insights,
        risks=report.risks,
        processing_time_s=report.processing_time_s,
        timestamp=now_iso,
    )


# --- Batch endpoints ---


class BatchTranscribeRequest(BaseModel):
    audio_paths: list[str] | None = None
    directory: str | None = None
    glob: str | None = Field(
        None,
        description="Glob pattern within directory, e.g. **/*.wav",
    )
    recursive: bool = True
    limit: int | None = Field(None, ge=1)
    workers: int = Field(1, ge=1, le=8)
    # ASR params
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(True)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")


class BatchTranscribeItem(BaseModel):
    file: str
    transcript: dict[str, Any] | None = None
    error: str | None = None


class BatchTranscribeResponse(BaseModel):
    items: list[BatchTranscribeItem]
    ok: int
    failed: int
    total: int
    timestamp: str


# Alias to keep signature short
BTResp = BatchTranscribeResponse


@app.post(
    "/batch_transcribe",
    response_model=BatchTranscribeResponse,
)
async def batch_transcribe(req: BatchTranscribeRequest) -> BTResp:
    """Transcribe multiple audio files concurrently.

    This endpoint supports processing multiple audio files in parallel
    using ThreadPoolExecutor for improved performance.
    Files can be specified explicitly, via a directory scan,
    or using glob patterns.

    Args:
        req: BatchTranscribeRequest containing file paths and ASR
            parameters

    Returns:
        BatchTranscribeResponse with transcriptions for all files and
        timestamp
    """
    # Resolve audio file paths from various input methods
    files = _resolve_audio_paths(
        req.audio_paths,
        req.directory,
        req.glob,
        req.recursive,
    )
    if req.limit:
        files = files[: req.limit]
    items: list[BatchTranscribeItem] = []
    ok = 0
    failed = 0

    def _worker(p: str) -> tuple[str, dict[str, Any]]:
        """Worker function for processing individual audio files."""
        tr = _transcribe_helper(
            audio_path=p,
            model=req.model,
            backend=req.backend,
            device=req.device,
            language=req.language,
            beam_size=req.beam_size,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            chunk_length_s=req.chunk_length_s,
            revision=req.revision,
            diarize=req.diarize,
            num_speakers=req.num_speakers,
        )
        return p, tr

    # Process files concurrently using ThreadPoolExecutor
    # Enables parallel processing while avoiding blocking the async
    # event loop
    if req.workers == 1:
        for p in files:
            try:
                _, tr = _worker(p)
                items.append(BatchTranscribeItem(file=p, transcript=tr))
                ok += 1
            except Exception as e:
                items.append(BatchTranscribeItem(file=p, error=str(e)))
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=req.workers) as ex:
            futs = {ex.submit(_worker, p): p for p in files}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    _, tr = fut.result()
                    items.append(BatchTranscribeItem(file=p, transcript=tr))
                    ok += 1
                except Exception as e:
                    items.append(BatchTranscribeItem(file=p, error=str(e)))
                    failed += 1

    # Generate timestamp for response
    now_iso = _utc_now_iso()
    return BatchTranscribeResponse(
        items=items,
        ok=ok,
        failed=failed,
        total=len(files),
        timestamp=now_iso,
    )


class BatchAnalyzeConversationRequest(BaseModel):
    audio_paths: list[str] | None = None
    directory: str | None = None
    glob: str | None = Field(None)
    recursive: bool = True
    limit: int | None = Field(None, ge=1)
    workers: int = Field(1, ge=1, le=8)
    # ASR
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(False)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")
    # Sentiment
    sentiment_model: str | None = Field(None)
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class BatchAnalyzeConversationItem(BaseModel):
    file: str
    transcript: dict[str, Any] | None = None
    segment_sentiments: list[SegmentSentiment] | None = None
    meta: dict[str, Any] | None = None
    error: str | None = None


class BatchAnalyzeConversationResponse(BaseModel):
    items: list[BatchAnalyzeConversationItem]
    ok: int
    failed: int
    total: int
    timestamp: str


@app.post(
    "/batch_analyze_conversation",
    response_model=BatchAnalyzeConversationResponse,
)
async def batch_analyze_conversation(
    req: BatchAnalyzeConversationRequest,
) -> BatchAnalyzeConversationResponse:
    """Analyze sentiment for multiple conversation audio files concurrently.

    This endpoint combines ASR transcription and sentiment analysis for
    multiple conversation audio files. It processes them in parallel
    using ThreadPoolExecutor. Each conversation is split into segments,
    and sentiment is analyzed per segment using the specialized
    'call' profile.

    Args:
        req: BatchAnalyzeConversationRequest containing file paths and
            processing parameters

    Returns:
        BatchAnalyzeConversationResponse with sentiment analysis results
        for all conversations
    """
    # Resolve audio file paths from various input methods
    files = _resolve_audio_paths(
        req.audio_paths,
        req.directory,
        req.glob,
        req.recursive,
    )
    if req.limit:
        files = files[: req.limit]
    items: list[BatchAnalyzeConversationItem] = []
    ok = 0
    failed = 0

    # Local alias to keep type annotation lines short
    worker_result_type = tuple[
        str,
        dict[str, Any],
        list[SegmentSentiment],
        dict[str, Any],
    ]

    def _worker(
        p: str,
    ) -> worker_result_type:
        """Worker for processing conversation audio files.

        Processes a single file: transcribe, analyze, and package.
        """
        # Transcribe the conversation audio file
        tr = _transcribe_helper(
            audio_path=p,
            model=req.model,
            backend=req.backend,
            device=req.device,
            language=req.language,
            beam_size=req.beam_size,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            chunk_length_s=req.chunk_length_s,
            revision=req.revision,
            diarize=req.diarize,
            num_speakers=req.num_speakers,
        )
        # Extract text segments from transcription
        segments = tr.get("segments", []) or []
        tr_texts = _texts_from_segments(segments)

        # Analyze sentiment for each segment using the 'call' profile
        results, meta = analyze_smart(
            tr_texts,
            profile="call",
            model_name=req.sentiment_model,
            device=req.device,
            batch_size=req.batch_size,
            normalize=True,
            return_all_scores=True,
            max_length=None,
            clean=True,
        )

        results = _blend_with_lexicon(tr_texts, results, req.lexicon_file, req.lexicon_weight)
        seg_out = _map_results_to_segments(tr_texts, results, segments)

        return p, tr, seg_out, meta

    # Process files concurrently using ThreadPoolExecutor
    # This enables parallel processing of multiple conversation files
    with ThreadPoolExecutor(max_workers=req.workers) as pool:
        futures = {pool.submit(_worker, p): p for p in files}
        for fut in as_completed(futures):
            file_path = futures[fut]
            try:
                _, tr, segs, meta = fut.result()
                items.append(
                    BatchAnalyzeConversationItem(
                        file=file_path,
                        transcript=tr,
                        segment_sentiments=segs,
                        meta=meta,
                    )
                )
                ok += 1
            except Exception as e:
                logger.error(
                    "Batch conversation analysis failed for %s: %s", file_path, e, exc_info=True
                )
                items.append(BatchAnalyzeConversationItem(file=file_path, error=str(e)))
                failed += 1

    now_iso = _utc_now_iso()
    return BatchAnalyzeConversationResponse(
        items=items,
        ok=ok,
        failed=failed,
        total=len(files),
        timestamp=now_iso,
    )


# --- Directory scan + small-batch processing ---


class ScanProcessRequest(BaseModel):
    # Directory to scan for audio files
    directory: str = Field(..., description="Directory to scan")
    # Optional glob pattern to filter files within the directory
    pattern: str | None = Field(
        None,
        description="Glob pattern relative to directory (e.g., **/*.wav)",
    )
    # Whether to recursively scan subdirectories
    recursive: bool = True
    # Number of files to process in each batch
    batch_size: int = Field(4, ge=1, le=64)
    # Optional limit on the total number of files to process
    max_files: int | None = Field(None, ge=1)
    # Optional JSON file to track processed files for incremental processing
    state_file: str | None = Field(
        None,
        description="Optional JSON file to track processed files",
    )
    # Number of parallel workers to use per batch
    workers: int = Field(
        1,
        ge=1,
        le=8,
        description="Parallel workers per batch",
    )
    # Operation to perform on each file (transcribe or analyze_conversation)
    operation: str = Field(
        "transcribe",
        description="transcribe | analyze_conversation",
    )
    # ASR parameters for transcription
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(False)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")
    # Sentiment analysis parameters (only used when
    # operation=analyze_conversation)
    sentiment_model: str | None = Field(None)
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class ScanItem(BaseModel):
    # Path to the processed file
    file: str
    # Whether processing was successful
    ok: bool
    # Error message if processing failed
    error: str | None = None
    # Processing result data (transcript or analysis payload)
    data: dict[str, Any] | None = None
    # Index of the batch this file was processed in
    batch_index: int


class ScanProcessResponse(BaseModel):
    # List of processing results for each file
    items: list[ScanItem]
    # Number of successfully processed files
    ok: int
    # Number of files that failed processing
    failed: int
    # Total number of files processed
    total: int
    # Number of batches processed
    batches: int
    # ISO timestamp when processing completed
    timestamp: str


@app.post("/scan_process", response_model=ScanProcessResponse)
async def scan_process(req: ScanProcessRequest) -> ScanProcessResponse:
    """Process audio files with incremental scanning and batched processing.

    This endpoint scans a directory for audio files and processes them
    in small batches,
    supporting both transcription and
    conversation sentiment analysis.
    It tracks processed
    files in a state file to enable incremental processing, avoiding
    reprocessing of
    unchanged files in subsequent runs.

    Args:
        req: ScanProcessRequest containing directory scanning and
            processing parameters

    Returns:
        ScanProcessResponse with processing results for all files
    """
    # Discover audio files in the specified directory using
    # glob patterns if provided
    files = _resolve_audio_paths(
        directory=req.directory, pattern=req.pattern, recursive=req.recursive
    )

    # Load processing state and filter for new or changed files only
    # This enables incremental processing by skipping
    # previously processed files
    state = _load_state(req.state_file)
    processed = state.get("processed", {})
    new_files: list[str] = []
    for p in files:
        try:
            # Get file modification time to check if it's been updated
            mtime = os.path.getmtime(p)
        except Exception:
            # Skip files that can't be accessed
            continue
        info = processed.get(p)
        # Process file if it hasn't been processed before
        # or if it's been modified
        if not info or not isinstance(info, dict) or float(info.get("mtime", 0.0)) < float(mtime):
            new_files.append(p)

    # Apply file limit if specified
    if req.max_files:
        new_files = new_files[: req.max_files]

    # Divide files into batches for processing
    batches = _chunk(new_files, req.batch_size)
    items: list[ScanItem] = []
    ok = 0
    failed = 0

    def _do_transcribe(p: str) -> dict[str, Any]:
        """Transcribe a single audio file using ASR."""
        return _transcribe_helper(
            audio_path=p,
            model=req.model,
            backend=req.backend,
            device=req.device,
            language=req.language,
            beam_size=req.beam_size,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            chunk_length_s=req.chunk_length_s,
            revision=req.revision,
            diarize=req.diarize,
            num_speakers=req.num_speakers,
        )

    def _do_analyze(p: str) -> dict[str, Any]:
        """Analyze sentiment of conversation segments in an audio file."""
        # Transcribe the audio file first
        tr = _do_transcribe(p)

        # Extract text segments from transcription
        segments = tr.get("segments", []) or []
        texts = _texts_from_segments(segments)

        # Run sentiment analysis on each segment using the 'call' profile
        results, meta = analyze_smart(
            texts,
            profile="call",
            model_name=req.sentiment_model,
            device=req.device,
            batch_size=req.batch_size,
            normalize=True,
            return_all_scores=True,
            max_length=None,
            clean=True,
        )

        results = _blend_with_lexicon(texts, results, req.lexicon_file, req.lexicon_weight)
        seg_out = _map_results_to_segments(texts, results, segments)
        return {
            "transcript": tr,
            "segment_sentiments": [s.model_dump() for s in seg_out],
            "meta": meta,
        }

    # Process batches with optional concurrency
    for bidx, batch in enumerate(batches):
        if req.workers == 1:
            # Sequential processing when no concurrency requested
            for p in batch:
                try:
                    # Perform requested operation (transcribe or analyze)
                    data = _do_transcribe(p) if req.operation == "transcribe" else _do_analyze(p)
                    items.append(ScanItem(file=p, ok=True, data=data, batch_index=bidx))
                    ok += 1
                    # Update state with file processing information
                    processed[p] = {
                        "mtime": os.path.getmtime(p),
                        "when": _utc_now_iso(trim_microseconds=False),
                    }
                except Exception as e:
                    # Record failed processing attempts
                    items.append(
                        ScanItem(
                            file=p,
                            ok=False,
                            error=str(e),
                            batch_index=bidx,
                        )
                    )
                    failed += 1
        else:
            # Concurrent processing using ThreadPoolExecutor
            def _wrap(pth: str):
                """Wrapper function for concurrent processing."""
                return (
                    pth,
                    (_do_transcribe(pth) if req.operation == "transcribe" else _do_analyze(pth)),
                )

            _lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=req.workers) as ex:
                # Submit all files in the batch for concurrent processing
                futs = {ex.submit(_wrap, p): p for p in batch}
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        # Extract results from completed future
                        _, data = fut.result()
                        with _lock:
                            items.append(
                                ScanItem(
                                    file=p,
                                    ok=True,
                                    data=data,
                                    batch_index=bidx,
                                )
                            )
                            ok += 1
                            # Update state with file processing information
                            processed[p] = {
                                "mtime": os.path.getmtime(p),
                                "when": _utc_now_iso(trim_microseconds=False),
                            }
                    except Exception as e:
                        # Record failed processing attempts
                        with _lock:
                            items.append(
                                ScanItem(
                                    file=p,
                                    ok=False,
                                    error=str(e),
                                    batch_index=bidx,
                                )
                            )
                            failed += 1

    # Persist processing state for incremental processing in future runs
    state["processed"] = processed
    _save_state(req.state_file, state)

    # Generate timestamp for response
    now_iso = _utc_now_iso()
    return ScanProcessResponse(
        items=items,
        ok=ok,
        failed=failed,
        total=len(new_files),
        batches=len(batches),
        timestamp=now_iso,
    )
