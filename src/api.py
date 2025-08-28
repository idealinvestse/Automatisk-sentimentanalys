from __future__ import annotations

# Standard library imports for type hints, file operations, and JSON handling
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import os
import glob
import json
# Concurrency utilities for parallel processing of audio files
from concurrent.futures import ThreadPoolExecutor, as_completed

# FastAPI framework for building the REST API
from fastapi import FastAPI
# Pydantic for data validation and serialization
from pydantic import BaseModel, Field

# Internal sentiment/ASR/lexicon modules
from .sentiment import analyze_smart
from .asr import transcribe as asr_transcribe
from .lexicon import (
    load_lexicon,
    score_text,
    scalar_to_dist,
    blend_distributions,
)


# Initialize FastAPI app
app = FastAPI(title="Swedish Sentiment API", version="0.2.0")


class AnalyzeRequest(BaseModel):
    # List of texts to analyze for sentiment
    texts: List[str] = Field(..., description="List of texts to analyze")
    # Data type classification for profile selection
    datatype: Optional[str] = Field(
        None,
        description=(
            "Data type: post, comment, article, review, ..."
        ),
    )
    # Source classification for profile selection
    source: Optional[str] = Field(
        None,
        description=(
            "Source: forum, magazine, news, social, ..."
        ),
    )
    # Explicit profile name to use for sentiment analysis
    profile: Optional[str] = Field(
        None,
        description="Explicit profile name to use",
    )
    # Optional model override for sentiment analysis
    model: Optional[str] = Field(
        None,
        description="Optional model override",
    )
    # Device specification for model execution
    device: Optional[str] = Field(
        "auto",
        description="Device: auto, cpu, cuda, cuda:0, mps",
    )
    # Batch size for processing multiple texts efficiently
    batch_size: int = Field(16, ge=1, le=128)
    # Whether to return all scores or just the top prediction
    return_all_scores: bool = Field(False)
    # Maximum text length for processing (truncates longer texts)
    max_length: Optional[int] = Field(None, ge=8, le=4096)
    # Whether to clean text before analysis
    # (HTML unescaping, URL removal, etc.)
    clean: bool = Field(True)
    # Whether to normalize text (case, whitespace, etc.)
    normalize: bool = Field(True)
    # Path to Swedish lexicon file for blending with model predictions
    lexicon_file: Optional[str] = Field(
        None,
        description=(
            "Path to Swedish lexicon (CSV/TSV) with columns "
            "term|word and polarity|score|sentiment"
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
    meta: Dict[str, Any]
    timestamp: str
    results: List[Any]


class TranscribeRequest(BaseModel):
    # Path to audio file (server/container must have access)
    audio_path: str = Field(
        ...,
        description=(
            "Path to audio file accessible by the "
            "server/container"
        ),
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


class TranscribeResponse(BaseModel):
    transcript: Dict[str, Any]
    timestamp: str


class AnalyzeConversationRequest(BaseModel):
    # Path to audio file the server can access
    audio_path: str = Field(
        ...,
        description=(
            "Path to audio file accessible by the server/"
            "container"
        ),
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
    # Sentiment analysis parameters
    # Whether to return all scores or only the top prediction
    return_all_scores: bool = Field(True)
    # Optional model override for sentiment analysis
    sentiment_model: Optional[str] = Field(
        None,
        description="Optional override for sentiment model",
    )
    # Path to Swedish lexicon file for blending with model predictions
    lexicon_file: Optional[str] = Field(None)
    # Blend weight [0..1]; 0=model only, 1=lexicon only
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class SegmentSentiment(BaseModel):
    # Index of the segment in the conversation
    index: int
    # Start time of the segment in seconds
    start: Optional[float]
    # End time of the segment in seconds
    end: Optional[float]
    # Transcribed text of the segment
    text: str
    # Predicted sentiment label (negativ, neutral, or positiv)
    label: str
    # Confidence score for the predicted label
    score: float
    # Individual sentiment scores when return_all_scores=True
    negativ: Optional[float] = None
    neutral: Optional[float] = None
    positiv: Optional[float] = None


class AnalyzeConversationResponse(BaseModel):
    # Full transcription result from ASR
    transcript: Dict[str, Any]
    # Sentiment analysis results for each conversation segment
    segment_sentiments: List[SegmentSentiment]
    # Metadata about the analysis process (model used, device, timing, etc.)
    meta: Dict[str, Any]
    # ISO timestamp when the analysis was completed
    timestamp: str


# --- Helpers for batch/scan ---
# Set of supported audio file extensions for batch processing
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wma", ".aac"}


def _resolve_audio_paths(audio_paths: Optional[List[str]] = None,
                         directory: Optional[str] = None,
                         pattern: Optional[str] = None,
                         recursive: bool = True) -> List[str]:
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
    files: List[str] = []
    # Process explicit paths: files, directories, or glob patterns
    for p in (audio_paths or []):
        # Handle glob patterns (*, ?, []) in the path list
        if any(ch in p for ch in ["*", "?", "["]):
            for m in glob.glob(p, recursive=recursive):
                # Validate that matched item is a file with supported extension
                if (
                    os.path.isfile(m)
                    and os.path.splitext(m)[1].lower() in AUDIO_EXTS
                ):
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
        elif (
            os.path.isfile(p)
            and os.path.splitext(p)[1].lower() in AUDIO_EXTS
        ):
            files.append(os.path.abspath(p))
    # Process directory + pattern combination if directory is specified
    if directory:
        if pattern:
            # Create full path pattern and match files
            pat = os.path.join(directory, pattern)
            for m in glob.glob(pat, recursive=recursive):
                # Validate file has supported extension
                if (
                    os.path.isfile(m)
                    and os.path.splitext(m)[1].lower() in AUDIO_EXTS
                ):
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


def _chunk(lst: List[str], size: int) -> List[List[str]]:
    """Divide a list into chunks of specified size.

    This helper function is used for batch processing to divide files into
    manageable groups for parallel processing.

    Args:
        lst: List to be chunked
        size: Maximum size of each chunk

    Returns:
        List of sublists (chunks) of the original list
    """
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _load_state(path: Optional[str]) -> Dict[str, Any]:
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
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            # Ensure the loaded object has the expected structure
            if (
                isinstance(obj, dict)
                and "processed" in obj
                and isinstance(obj["processed"], dict)
            ):
                return obj
    except Exception:
        # Return empty state if file reading or parsing fails
        pass
    return {"processed": {}}


def _save_state(path: Optional[str], state: Dict[str, Any]) -> None:
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
async def health() -> Dict[str, str]:
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
    # Optional lexicon blending (combine model predictions with lexicon)
    use_lex = (
        req.lexicon_file is not None
        and req.lexicon_weight
        and req.lexicon_weight > 0.0
    )
    if use_lex:
        try:
            # Load lexicon for blending
            lex = load_lexicon(req.lexicon_file)
            # Blend per item - process each text individually
            if results and isinstance(results[0], list):
                blended_results = []
                for t, inner in zip(req.texts, results):
                    # Extract scores from model results
                    scores = {
                        e.get("label"): float(e.get("score", 0.0))
                        for e in inner
                    }
                    # Ensure all sentiment labels are present
                    for k in ["negativ", "neutral", "positiv"]:
                        scores.setdefault(k, 0.0)
                    # Get lexicon-based scalar score for the text
                    s_scalar = score_text(t, lex)
                    # Convert scalar score to distribution
                    ln, le, lp = scalar_to_dist(s_scalar)
                    # Blend with lexicon (weighted)
                    scores = blend_distributions(
                        scores,
                        (ln, le, lp),
                        req.lexicon_weight,
                    )
                    # Convert back to list of dicts preserving order
                    blended_inner = [
                        {"label": "negativ", "score": scores["negativ"]},
                        {"label": "neutral", "score": scores["neutral"]},
                        {"label": "positiv", "score": scores["positiv"]},
                    ]
                    blended_results.append(blended_inner)
                results = blended_results
            else:
                # Approximate dist; blend; re-pick top-1
                # Handle case where only top prediction is returned
                blended_results = []
                for t, r in zip(req.texts, results):
                    # Extract label (score not used)
                    label = r.get("label")
                    # Create distribution based on top prediction
                    neg = 1.0 if label == "negativ" else 0.0
                    neu = 1.0 if label == "neutral" else 0.0
                    pos = 1.0 if label == "positiv" else 0.0
                    model_dist = {
                        "negativ": neg,
                        "neutral": neu,
                        "positiv": pos,
                    }
                    # Get lexicon-based scalar score
                    s_scalar = score_text(t, lex)
                    ln, le, lp = scalar_to_dist(s_scalar)
                    # Blend distributions
                    scores = blend_distributions(
                        model_dist,
                        (ln, le, lp),
                        req.lexicon_weight,
                    )
                    # Select top label from blended scores
                    top_label = max(scores.items(), key=lambda kv: kv[1])[0]
                    blended_results.append(
                        {
                            "label": top_label,
                            "score": float(scores[top_label]),
                        }
                    )
                results = blended_results
        except Exception:
            # Continue with model-only results if lexicon processing fails
            pass
    # Generate timestamp for response
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
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
    tr = asr_transcribe(
        audio_path=req.audio_path,
        model=req.model,
        backend=req.backend,
        device=req.device,
        language=req.language,
        beam_size=req.beam_size,
        vad=req.vad,
        word_timestamps=req.word_timestamps,
        chunk_length_s=req.chunk_length_s,
    )
    # Generate timestamp for response
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
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
    tr = asr_transcribe(
        audio_path=req.audio_path,
        model=req.model,
        backend=req.backend,
        device=req.device,
        language=req.language,
        beam_size=req.beam_size,
        vad=req.vad,
        word_timestamps=req.word_timestamps,
        chunk_length_s=req.chunk_length_s,
    )
    # Extract text segments from transcription
    segments = tr.get("segments", []) or []
    texts = [s.get("text", "").strip() for s in segments]
    # Fallback: single full transcript if no segments detected
    # This handles cases where VAD doesn't split the audio into segments
    if not texts or all(not t for t in texts):
        texts = (
            [
                " ".join(
                    [
                        s.get("text", "").strip()
                        for s in segments
                        if s.get("text")
                    ]
                ).strip()
            ]
            if segments
            else []
        )
        if not texts:
            tr_texts = []
        else:
            tr_texts = texts
    else:
        tr_texts = texts

    # Run sentiment analysis on each segment using the 'call' profile
    # The 'call' profile is optimized for conversation analysis
    results, meta = analyze_smart(
        tr_texts,
        profile="call",
        model_name=req.sentiment_model,
        device="auto",
        batch_size=16,
        normalize=True,
        return_all_scores=True,
        max_length=None,
        clean=True,
    )
    # Optional lexicon blending for improved sentiment accuracy
    use_lex = (
        req.lexicon_file is not None
        and req.lexicon_weight
        and req.lexicon_weight > 0.0
    )
    if use_lex:
        try:
            # Load lexicon for blending
            lex = load_lexicon(req.lexicon_file)
            blended_results = []
            for t, inner in zip(tr_texts, results):
                # Extract scores from model results
                scores = {
                    e.get("label"): float(e.get("score", 0.0))
                    for e in inner
                }
                # Ensure all sentiment labels are present
                for k in ["negativ", "neutral", "positiv"]:
                    scores.setdefault(k, 0.0)
                # Get lexicon-based scalar score
                s_scalar = score_text(t, lex)
                # Convert scalar score to distribution
                ln, le, lp = scalar_to_dist(s_scalar)
                # Blend model and lexicon distributions
                scores = blend_distributions(
                    scores,
                    (ln, le, lp),
                    req.lexicon_weight,
                )
                # Package blended results
                blended_inner = [
                    {"label": "negativ", "score": scores["negativ"]},
                    {"label": "neutral", "score": scores["neutral"]},
                    {"label": "positiv", "score": scores["positiv"]},
                ]
                blended_results.append(blended_inner)
            results = blended_results
        except Exception:
            # Continue with model-only results if lexicon processing fails
            pass

    # Map sentiment analysis results back to segments with timing information
    seg_out: List[SegmentSentiment] = []
    for idx, (t, inner) in enumerate(zip(tr_texts, results)):
        # Create scores map from analysis results
        scores_map = {
            e.get("label"): float(e.get("score", 0.0))
            for e in inner
        }
        # Ensure all sentiment labels are present in scores map
        for k in ["negativ", "neutral", "positiv"]:
            scores_map.setdefault(k, 0.0)
        # Select top sentiment label
        top_label = max(scores_map.items(), key=lambda kv: kv[1])[0]
        top_score = float(scores_map[top_label])
        start = None
        end = None
        if idx < len(segments):
            start = (
                float(segments[idx].get("start", 0.0) or 0.0)
                if isinstance(segments[idx].get("start"), (int, float))
                else None
            )
            end = (
                float(segments[idx].get("end", 0.0) or 0.0)
                if isinstance(segments[idx].get("end"), (int, float))
                else None
            )
        seg_out.append(
            SegmentSentiment(
                index=idx,
                start=start,
                end=end,
                text=t,
                label=top_label,
                score=top_score,
                negativ=scores_map.get("negativ"),
                neutral=scores_map.get("neutral"),
                positiv=scores_map.get("positiv"),
            )
        )

    # Build response
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return AnalyzeConversationResponse(
        transcript=tr,
        segment_sentiments=seg_out,
        meta=meta,
        timestamp=now_iso,
    )

# --- Batch endpoints ---


class BatchTranscribeRequest(BaseModel):
    audio_paths: Optional[List[str]] = None
    directory: Optional[str] = None
    glob: Optional[str] = Field(
        None,
        description="Glob pattern within directory, e.g. **/*.wav",
    )
    recursive: bool = True
    limit: Optional[int] = Field(None, ge=1)
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


class BatchTranscribeItem(BaseModel):
    file: str
    transcript: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchTranscribeResponse(BaseModel):
    items: List[BatchTranscribeItem]
    ok: int
    failed: int


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
    items: List[BatchTranscribeItem] = []
    ok = 0
    failed = 0

    def _worker(p: str) -> Tuple[str, Dict[str, Any]]:
        """Worker function for processing individual audio files."""
        tr = asr_transcribe(
            audio_path=p,
            model=req.model,
            backend=req.backend,
            device=req.device,
            language=req.language,
            beam_size=req.beam_size,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            chunk_length_s=req.chunk_length_s,
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
            futs = {
                ex.submit(_worker, p): p
                for p in files
            }
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
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return BatchTranscribeResponse(
        items=items,
        ok=ok,
        failed=failed,
        total=len(files),
        timestamp=now_iso,
    )


class BatchAnalyzeConversationRequest(BaseModel):
    audio_paths: Optional[List[str]] = None
    directory: Optional[str] = None
    glob: Optional[str] = Field(None)
    recursive: bool = True
    limit: Optional[int] = Field(None, ge=1)
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
    # Sentiment
    sentiment_model: Optional[str] = Field(None)
    lexicon_file: Optional[str] = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class BatchAnalyzeConversationItem(BaseModel):
    file: str
    transcript: Optional[Dict[str, Any]] = None
    segment_sentiments: Optional[List[SegmentSentiment]] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchAnalyzeConversationResponse(BaseModel):
    items: List[BatchAnalyzeConversationItem]
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
    items: List[BatchAnalyzeConversationItem] = []
    ok = 0
    failed = 0

    # Local alias to keep type annotation lines short
    WorkerResult = Tuple[
        str,
        Dict[str, Any],
        List[SegmentSentiment],
        Dict[str, Any],
    ]

    def _worker(
        p: str,
    ) -> WorkerResult:
        """Worker for processing conversation audio files.

        Processes a single file: transcribe, analyze, and package.
        """
        # Transcribe the conversation audio file
        tr = asr_transcribe(
            audio_path=p,
            model=req.model,
            backend=req.backend,
            device=req.device,
            language=req.language,
            beam_size=req.beam_size,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            chunk_length_s=req.chunk_length_s,
        )
        # Extract text segments from transcription
        segments = tr.get("segments", []) or []
        texts = [s.get("text", "").strip() for s in segments]
        # Fallback: single full transcript if no segments detected
        if not texts or all(not t for t in texts):
            texts = (
                [
                    " ".join(
                        [
                            s.get("text", "").strip()
                            for s in segments
                            if s.get("text")
                        ]
                    ).strip()
                ]
                if segments
                else []
            )
            if not texts:
                tr_texts = []
            else:
                tr_texts = texts
        else:
            tr_texts = texts

        # Analyze sentiment for each segment using the 'call' profile
        results, meta = analyze_smart(
            tr_texts,
            profile="call",
            model_name=req.sentiment_model,
            device="auto",
            batch_size=16,
            normalize=True,
            return_all_scores=True,
            max_length=None,
            clean=True,
        )

        # Optional lexicon blending for improved sentiment accuracy
        use_lex = (
            req.lexicon_file is not None
            and req.lexicon_weight
            and req.lexicon_weight > 0.0
        )
        if use_lex:
            try:
                lex = load_lexicon(req.lexicon_file)
                blended_results = []
                for t, inner in zip(tr_texts, results):
                    scores = {
                        e.get("label"): float(e.get("score", 0.0))
                        for e in inner
                    }
                    for k in ["negativ", "neutral", "positiv"]:
                        scores.setdefault(k, 0.0)
                    s_scalar = score_text(t, lex)
                    ln, le, lp = scalar_to_dist(s_scalar)
                    scores = blend_distributions(
                        scores,
                        (ln, le, lp),
                        req.lexicon_weight,
                    )
                    blended_inner = [
                        {"label": "negativ", "score": scores["negativ"]},
                        {"label": "neutral", "score": scores["neutral"]},
                        {"label": "positiv", "score": scores["positiv"]},
                    ]
                    blended_results.append(blended_inner)
                results = blended_results
            except Exception:
                pass

        # Map results back to segments with timing info
        seg_out: List[SegmentSentiment] = []
        for idx, (t, inner) in enumerate(zip(tr_texts, results)):
            scores_map = {
                e.get("label"): float(e.get("score", 0.0))
                for e in inner
            }
            for k in ["negativ", "neutral", "positiv"]:
                scores_map.setdefault(k, 0.0)
            top_label = max(scores_map.items(), key=lambda kv: kv[1])[0]
            top_score = float(scores_map[top_label])
            start = None
            end = None
            if idx < len(segments):
                start = (
                    float(segments[idx].get("start", 0.0) or 0.0)
                    if isinstance(segments[idx].get("start"), (int, float))
                    else None
                )
                end = (
                    float(segments[idx].get("end", 0.0) or 0.0)
                    if isinstance(segments[idx].get("end"), (int, float))
                    else None
                )
            seg_out.append(
                SegmentSentiment(
                    index=idx,
                    start=start,
                    end=end,
                    text=t,
                    label=top_label,
                    score=top_score,
                    negativ=scores_map.get("negativ"),
                    neutral=scores_map.get("neutral"),
                    positiv=scores_map.get("positiv"),
                )
            )

        return p, tr, seg_out, meta

    # Process files concurrently using ThreadPoolExecutor
    # This enables parallel processing of multiple conversation files
    with ThreadPoolExecutor(max_workers=req.workers) as pool:
        futures = [pool.submit(_worker, p) for p in files]
        for fut in futures:
            try:
                file, tr, segs, meta = fut.result()
                items.append(
                    BatchAnalyzeConversationItem(
                        file=file,
                        transcript=tr,
                        segment_sentiments=segs,
                        meta=meta,
                    )
                )
                ok += 1
            except Exception as e:
                items.append(
                    BatchAnalyzeConversationItem(
                        file=file,
                        error=str(e),
                    )
                )
                failed += 1

    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
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
    pattern: Optional[str] = Field(
        None,
        description="Glob pattern relative to directory (e.g., **/*.wav)",
    )
    # Whether to recursively scan subdirectories
    recursive: bool = True
    # Number of files to process in each batch
    batch_size: int = Field(4, ge=1, le=64)
    # Optional limit on the total number of files to process
    max_files: Optional[int] = Field(None, ge=1)
    # Optional JSON file to track processed files for incremental processing
    state_file: Optional[str] = Field(
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
    # Sentiment analysis parameters (only used when
    # operation=analyze_conversation)
    sentiment_model: Optional[str] = Field(None)
    lexicon_file: Optional[str] = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class ScanItem(BaseModel):
    # Path to the processed file
    file: str
    # Whether processing was successful
    ok: bool
    # Error message if processing failed
    error: Optional[str] = None
    # Processing result data (transcript or analysis payload)
    data: Optional[Dict[str, Any]] = None
    # Index of the batch this file was processed in
    batch_index: int


class ScanProcessResponse(BaseModel):
    # List of processing results for each file
    items: List[ScanItem]
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
    files = _resolve_audio_paths(directory=req.directory,
                                 pattern=req.pattern,
                                 recursive=req.recursive)

    # Load processing state and filter for new or changed files only
    # This enables incremental processing by skipping
    # previously processed files
    state = _load_state(req.state_file)
    processed = state.get("processed", {})
    new_files: List[str] = []
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
        if not info or not isinstance(info, dict) or \
           float(info.get("mtime", 0.0)) < float(mtime):
            new_files.append(p)

    # Apply file limit if specified
    if req.max_files:
        new_files = new_files[:req.max_files]

    # Divide files into batches for processing
    batches = _chunk(new_files, req.batch_size)
    items: List[ScanItem] = []
    ok = 0
    failed = 0

    def _do_transcribe(p: str) -> Dict[str, Any]:
        """Transcribe a single audio file using ASR."""
        return asr_transcribe(
            audio_path=p,
            model=req.model,
            backend=req.backend,
            device=req.device,
            language=req.language,
            beam_size=req.beam_size,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            chunk_length_s=req.chunk_length_s,
        )

    def _do_analyze(p: str) -> Dict[str, Any]:
        """Analyze sentiment of conversation segments in an audio file."""
        # Transcribe the audio file first
        tr = _do_transcribe(p)

        # Extract text segments from transcription
        segments = tr.get("segments", []) or []
        texts = [s.get("text", "").strip() for s in segments]

        # Fallback: single full transcript if no segments detected
        if not texts or all(not t for t in texts):
            texts = [
                " ".join([
                    s.get("text", "").strip()
                    for s in segments if s.get("text")
                ]).strip()
            ] if segments else []

        # Run sentiment analysis on each segment using the 'call' profile
        results, meta = analyze_smart(
            texts,
            profile="call",
            model_name=req.sentiment_model,
            device="auto",
            batch_size=16,
            normalize=True,
            return_all_scores=True,
            max_length=None,
            clean=True,
        )

        # Optional lexicon blending for improved sentiment accuracy
        use_lex = (
            req.lexicon_file is not None
            and req.lexicon_weight
            and req.lexicon_weight > 0.0
        )
        if use_lex:
            try:
                # Load lexicon for blending
                lex = load_lexicon(req.lexicon_file)
                blended_results = []
                for t, inner in zip(texts, results):
                    # Extract scores from model results
                    scores = {
                        e.get("label"): float(e.get("score", 0.0))
                        for e in inner
                    }
                    # Ensure all sentiment labels are present
                    for k in ["negativ", "neutral", "positiv"]:
                        scores.setdefault(k, 0.0)
                    # Get lexicon-based scalar score
                    s_scalar = score_text(t, lex)
                    # Convert scalar score to distribution
                    ln, le, lp = scalar_to_dist(s_scalar)
                    # Blend model and lexicon distributions
                    scores = blend_distributions(
                        scores,
                        (ln, le, lp),
                        req.lexicon_weight,
                    )
                    # Package blended results
                    blended_inner = [
                        {"label": "negativ", "score": scores["negativ"]},
                        {"label": "neutral", "score": scores["neutral"]},
                        {"label": "positiv", "score": scores["positiv"]},
                    ]
                    blended_results.append(blended_inner)
                results = blended_results
            except Exception:
                # Continue with model-only results if lexicon processing fails
                pass

        # Package output similar to analyze_conversation endpoint
        seg_out: List[SegmentSentiment] = []
        for idx, (t, inner) in enumerate(zip(texts, results)):
            # Create scores map from analysis results
            scores_map = {
                e.get("label"): float(e.get("score", 0.0))
                for e in inner
            }
            # Ensure all sentiment labels are present in scores map
            for k in ["negativ", "neutral", "positiv"]:
                scores_map.setdefault(k, 0.0)
            # Select top sentiment label
            top_label = max(scores_map.items(), key=lambda kv: kv[1])[0]
            top_score = float(scores_map[top_label])
            # Extract timing information from original segments
            start = float(segments[idx].get("start", 0.0) or 0.0) \
                if idx < len(segments) else None
            end = float(segments[idx].get("end", 0.0) or 0.0) \
                if idx < len(segments) else None
            # Create segment sentiment object
            seg_out.append(SegmentSentiment(
                index=idx,
                start=start,
                end=end,
                text=t,
                label=top_label,
                score=top_score,
                negativ=scores_map.get("negativ"),
                neutral=scores_map.get("neutral"),
                positiv=scores_map.get("positiv"),
            ))
        return {
            "transcript": tr,
            "segment_sentiments": [s.model_dump() for s in seg_out],
            "meta": meta
        }

    # Process batches with optional concurrency
    for bidx, batch in enumerate(batches):
        if req.workers == 1:
            # Sequential processing when no concurrency requested
            for p in batch:
                try:
                    # Perform requested operation (transcribe or analyze)
                    data = (
                        _do_transcribe(p)
                        if req.operation == "transcribe"
                        else _do_analyze(p)
                    )
                    items.append(
                        ScanItem(file=p, ok=True, data=data, batch_index=bidx)
                    )
                    ok += 1
                    # Update state with file processing information
                    processed[p] = {
                        "mtime": os.path.getmtime(p),
                        "when": datetime.utcnow().isoformat() + "Z",
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
                    (
                        _do_transcribe(pth)
                        if req.operation == "transcribe"
                        else _do_analyze(pth)
                    ),
                )

            with ThreadPoolExecutor(max_workers=req.workers) as ex:
                # Submit all files in the batch for concurrent processing
                futs = {
                    ex.submit(_wrap, p): p for p in batch
                }
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        # Extract results from completed future
                        _, data = fut.result()
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
                            "when": datetime.utcnow().isoformat() + "Z",
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

    # Persist processing state for incremental processing in future runs
    state["processed"] = processed
    _save_state(req.state_file, state)

    # Generate timestamp for response
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return ScanProcessResponse(
        items=items,
        ok=ok,
        failed=failed,
        total=len(new_files),
        batches=len(batches),
        timestamp=now_iso,
    )
