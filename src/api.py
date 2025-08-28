from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .sentiment import analyze_smart
from .asr import transcribe as asr_transcribe
from .lexicon import load_lexicon, score_text, scalar_to_dist, blend_distributions


app = FastAPI(title="Swedish Sentiment API", version="0.2.0")


class AnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    datatype: Optional[str] = Field(None, description="Data type: post, comment, article, review, ...")
    source: Optional[str] = Field(None, description="Source: forum, magazine, news, social, ...")
    profile: Optional[str] = Field(None, description="Explicit profile name to use")
    model: Optional[str] = Field(None, description="Optional model override")
    device: Optional[str] = Field("auto", description="Device: auto, cpu, cuda, cuda:0, mps")
    batch_size: int = Field(16, ge=1, le=128)
    return_all_scores: bool = Field(False)
    max_length: Optional[int] = Field(None, ge=8, le=4096)
    clean: bool = Field(True)
    normalize: bool = Field(True)
    lexicon_file: Optional[str] = Field(None, description="Path to Swedish lexicon (CSV/TSV) with columns term|word and polarity|score|sentiment")
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0, description="Blend weight [0..1] for lexicon distribution")


class AnalyzeResponse(BaseModel):
    meta: Dict[str, Any]
    timestamp: str
    results: List[Any]


class TranscribeRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file accessible by the server/container")
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster", description="faster | transformers")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(True)
    chunk_length_s: int = Field(30, ge=5, le=60)


class TranscribeResponse(BaseModel):
    transcript: Dict[str, Any]
    timestamp: str


class AnalyzeConversationRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file accessible by the server/container")
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
    return_all_scores: bool = Field(True)
    sentiment_model: Optional[str] = Field(None, description="Optional override for sentiment model")
    lexicon_file: Optional[str] = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class SegmentSentiment(BaseModel):
    index: int
    start: Optional[float]
    end: Optional[float]
    text: str
    label: str
    score: float
    negativ: Optional[float] = None
    neutral: Optional[float] = None
    positiv: Optional[float] = None


class AnalyzeConversationResponse(BaseModel):
    transcript: Dict[str, Any]
    segment_sentiments: List[SegmentSentiment]
    meta: Dict[str, Any]
    timestamp: str


# --- Helpers for batch/scan ---
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wma", ".aac"}


def _resolve_audio_paths(audio_paths: Optional[List[str]] = None,
                         directory: Optional[str] = None,
                         pattern: Optional[str] = None,
                         recursive: bool = True) -> List[str]:
    files: List[str] = []
    # explicit paths
    for p in (audio_paths or []):
        if any(ch in p for ch in ["*", "?", "["]):
            for m in glob.glob(p, recursive=recursive):
                if os.path.isfile(m) and os.path.splitext(m)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.abspath(m))
        elif os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(os.path.join(root, fn)))
                if not recursive:
                    break
        elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in AUDIO_EXTS:
            files.append(os.path.abspath(p))
    # directory + pattern
    if directory:
        if pattern:
            pat = os.path.join(directory, pattern)
            for m in glob.glob(pat, recursive=recursive):
                if os.path.isfile(m) and os.path.splitext(m)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.abspath(m))
        else:
            for root, _, fnames in os.walk(directory):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(os.path.join(root, fn)))
                if not recursive:
                    break
    # dedup + sort
    files = sorted(dict.fromkeys(files))
    return files


def _chunk(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _load_state(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"processed": {}}
    if not os.path.isfile(path):
        return {"processed": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict) and "processed" in obj and isinstance(obj["processed"], dict):
                return obj
    except Exception:
        pass
    return {"processed": {}}


def _save_state(path: Optional[str], state: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
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
    # Optional lexicon blending
    use_lex = req.lexicon_file is not None and req.lexicon_weight and req.lexicon_weight > 0.0
    if use_lex:
        try:
            lex = load_lexicon(req.lexicon_file)
            # Blend per item
            if results and isinstance(results[0], list):
                blended_results = []
                for t, inner in zip(req.texts, results):
                    scores = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
                    for k in ["negativ", "neutral", "positiv"]:
                        scores.setdefault(k, 0.0)
                    s_scalar = score_text(t, lex)
                    ln, le, lp = scalar_to_dist(s_scalar)
                    scores = blend_distributions(scores, (ln, le, lp), req.lexicon_weight)
                    # Convert back to list of dicts preserving order
                    blended_inner = [
                        {"label": "negativ", "score": scores["negativ"]},
                        {"label": "neutral", "score": scores["neutral"]},
                        {"label": "positiv", "score": scores["positiv"]},
                    ]
                    blended_results.append(blended_inner)
                results = blended_results
            else:
                # top-1 like; approximate distribution then blend and re-pick top1
                blended_results = []
                for t, r in zip(req.texts, results):
                    label = r.get("label")
                    score = float(r.get("score", 0.0))
                    neg = 1.0 if label == "negativ" else 0.0
                    neu = 1.0 if label == "neutral" else 0.0
                    pos = 1.0 if label == "positiv" else 0.0
                    model_dist = {"negativ": neg, "neutral": neu, "positiv": pos}
                    s_scalar = score_text(t, lex)
                    ln, le, lp = scalar_to_dist(s_scalar)
                    scores = blend_distributions(model_dist, (ln, le, lp), req.lexicon_weight)
                    top_label = max(scores.items(), key=lambda kv: kv[1])[0]
                    blended_results.append({"label": top_label, "score": float(scores[top_label])})
                results = blended_results
        except Exception:
            pass
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return AnalyzeResponse(meta=meta, timestamp=now_iso, results=results)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """Run ASR on an audio file and return normalized transcript."""
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
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return TranscribeResponse(transcript=tr, timestamp=now_iso)


@app.post("/analyze_conversation", response_model=AnalyzeConversationResponse)
async def analyze_conversation(req: AnalyzeConversationRequest) -> AnalyzeConversationResponse:
    """Transcribe a call and run sentiment per segment using the 'call' profile."""
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
    segments = tr.get("segments", []) or []
    texts = [s.get("text", "").strip() for s in segments]
    # Fallback: single full transcript if no segments detected
    if not texts or all(not t for t in texts):
        texts = [" ".join([s.get("text", "").strip() for s in segments if s.get("text")]).strip()] if segments else []
        if not texts:
            full = ""
            tr_texts = []
        else:
            tr_texts = texts
    else:
        tr_texts = texts

    # Run sentiment using 'call' profile
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
    # Lexicon blending (if requested)
    use_lex = req.lexicon_file is not None and req.lexicon_weight and req.lexicon_weight > 0.0
    if use_lex:
        try:
            lex = load_lexicon(req.lexicon_file)
            blended_results = []
            for t, inner in zip(tr_texts, results):
                scores = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
                for k in ["negativ", "neutral", "positiv"]:
                    scores.setdefault(k, 0.0)
                s_scalar = score_text(t, lex)
                ln, le, lp = scalar_to_dist(s_scalar)
                scores = blend_distributions(scores, (ln, le, lp), req.lexicon_weight)
                blended_inner = [
                    {"label": "negativ", "score": scores["negativ"]},
                    {"label": "neutral", "score": scores["neutral"]},
                    {"label": "positiv", "score": scores["positiv"]},
                ]
                blended_results.append(blended_inner)
            results = blended_results
        except Exception:
            pass

    # Map results back to segments
    seg_out: List[SegmentSentiment] = []
    for idx, (t, inner) in enumerate(zip(tr_texts, results)):
        scores_map = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
        for k in ["negativ", "neutral", "positiv"]:
            scores_map.setdefault(k, 0.0)
        top_label = max(scores_map.items(), key=lambda kv: kv[1])[0]
        top_score = float(scores_map[top_label])
        start = None
        end = None
        if idx < len(segments):
            start = float(segments[idx].get("start", 0.0) or 0.0) if isinstance(segments[idx].get("start"), (int, float)) else None
            end = float(segments[idx].get("end", 0.0) or 0.0) if isinstance(segments[idx].get("end"), (int, float)) else None
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

    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return AnalyzeConversationResponse(transcript=tr, segment_sentiments=seg_out, meta=meta, timestamp=now_iso)


# --- Batch endpoints ---

class BatchTranscribeRequest(BaseModel):
    audio_paths: Optional[List[str]] = None
    directory: Optional[str] = None
    glob: Optional[str] = Field(None, description="Glob pattern within directory, e.g. **/*.wav")
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
    total: int
    timestamp: str


@app.post("/batch_transcribe", response_model=BatchTranscribeResponse)
async def batch_transcribe(req: BatchTranscribeRequest) -> BatchTranscribeResponse:
    files = _resolve_audio_paths(req.audio_paths, req.directory, req.glob, req.recursive)
    if req.limit:
        files = files[: req.limit]
    items: List[BatchTranscribeItem] = []
    ok = 0
    failed = 0

    def _worker(p: str) -> Tuple[str, Dict[str, Any]]:
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

    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return BatchTranscribeResponse(items=items, ok=ok, failed=failed, total=len(files), timestamp=now_iso)


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


@app.post("/batch_analyze_conversation", response_model=BatchAnalyzeConversationResponse)
async def batch_analyze_conversation(req: BatchAnalyzeConversationRequest) -> BatchAnalyzeConversationResponse:
    files = _resolve_audio_paths(req.audio_paths, req.directory, req.glob, req.recursive)
    if req.limit:
        files = files[: req.limit]
    items: List[BatchAnalyzeConversationItem] = []
    ok = 0
    failed = 0

    def _worker(p: str) -> Tuple[str, Dict[str, Any], List[SegmentSentiment], Dict[str, Any]]:
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
        segments = tr.get("segments", []) or []
        texts = [s.get("text", "").strip() for s in segments]
        if not texts or all(not t for t in texts):
            texts = [" ".join([s.get("text", "").strip() for s in segments if s.get("text")]).strip()] if segments else []
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
        # Lexicon blending
        use_lex = req.lexicon_file is not None and req.lexicon_weight and req.lexicon_weight > 0.0
        if use_lex:
            try:
                lex = load_lexicon(req.lexicon_file)
                blended_results = []
                for t, inner in zip(texts, results):
                    scores = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
                    for k in ["negativ", "neutral", "positiv"]:
                        scores.setdefault(k, 0.0)
                    s_scalar = score_text(t, lex)
                    ln, le, lp = scalar_to_dist(s_scalar)
                    scores = blend_distributions(scores, (ln, le, lp), req.lexicon_weight)
                    blended_inner = [
                        {"label": "negativ", "score": scores["negativ"]},
                        {"label": "neutral", "score": scores["neutral"]},
                        {"label": "positiv", "score": scores["positiv"]},
                    ]
                    blended_results.append(blended_inner)
                results = blended_results
            except Exception:
                pass
        seg_out: List[SegmentSentiment] = []
        for idx, (t, inner) in enumerate(zip(texts, results)):
            scores_map = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
            for k in ["negativ", "neutral", "positiv"]:
                scores_map.setdefault(k, 0.0)
            top_label = max(scores_map.items(), key=lambda kv: kv[1])[0]
            top_score = float(scores_map[top_label])
            start = float(segments[idx].get("start", 0.0) or 0.0) if idx < len(segments) else None
            end = float(segments[idx].get("end", 0.0) or 0.0) if idx < len(segments) else None
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
        return p, tr, seg_out, meta

    if req.workers == 1:
        for p in files:
            try:
                _, tr, segs_out, meta = _worker(p)
                items.append(BatchAnalyzeConversationItem(file=p, transcript=tr, segment_sentiments=segs_out, meta=meta))
                ok += 1
            except Exception as e:
                items.append(BatchAnalyzeConversationItem(file=p, error=str(e)))
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=req.workers) as ex:
            futs = {ex.submit(_worker, p): p for p in files}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    _, tr, segs_out, meta = fut.result()
                    items.append(BatchAnalyzeConversationItem(file=p, transcript=tr, segment_sentiments=segs_out, meta=meta))
                    ok += 1
                except Exception as e:
                    items.append(BatchAnalyzeConversationItem(file=p, error=str(e)))
                    failed += 1

    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return BatchAnalyzeConversationResponse(items=items, ok=ok, failed=failed, total=len(files), timestamp=now_iso)


# --- Directory scan + small-batch processing ---

class ScanProcessRequest(BaseModel):
    directory: str = Field(..., description="Directory to scan")
    pattern: Optional[str] = Field(None, description="Glob pattern relative to directory (e.g., **/*.wav)")
    recursive: bool = True
    batch_size: int = Field(4, ge=1, le=64)
    max_files: Optional[int] = Field(None, ge=1)
    state_file: Optional[str] = Field(None, description="Optional JSON file to track processed files")
    workers: int = Field(1, ge=1, le=8, description="Parallel workers per batch")
    operation: str = Field("transcribe", description="transcribe | analyze_conversation")
    # ASR params
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(False)
    chunk_length_s: int = Field(30, ge=5, le=60)
    # Sentiment (only for analyze_conversation)
    sentiment_model: Optional[str] = Field(None)
    lexicon_file: Optional[str] = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class ScanItem(BaseModel):
    file: str
    ok: bool
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None  # transcript or analysis payload
    batch_index: int


class ScanProcessResponse(BaseModel):
    items: List[ScanItem]
    ok: int
    failed: int
    total: int
    batches: int
    timestamp: str


@app.post("/scan_process", response_model=ScanProcessResponse)
async def scan_process(req: ScanProcessRequest) -> ScanProcessResponse:
    # discover files
    files = _resolve_audio_paths(directory=req.directory, pattern=req.pattern, recursive=req.recursive)
    # load state and filter new/changed files
    state = _load_state(req.state_file)
    processed = state.get("processed", {})
    new_files: List[str] = []
    for p in files:
        try:
            mtime = os.path.getmtime(p)
        except Exception:
            continue
        info = processed.get(p)
        if not info or not isinstance(info, dict) or float(info.get("mtime", 0.0)) < float(mtime):
            new_files.append(p)
    if req.max_files:
        new_files = new_files[: req.max_files]

    batches = _chunk(new_files, req.batch_size)
    items: List[ScanItem] = []
    ok = 0
    failed = 0

    def _do_transcribe(p: str) -> Dict[str, Any]:
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
        tr = _do_transcribe(p)
        segments = tr.get("segments", []) or []
        texts = [s.get("text", "").strip() for s in segments]
        if not texts or all(not t for t in texts):
            texts = [" ".join([s.get("text", "").strip() for s in segments if s.get("text")]).strip()] if segments else []
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
        # lexicon
        use_lex = req.lexicon_file is not None and req.lexicon_weight and req.lexicon_weight > 0.0
        if use_lex:
            try:
                lex = load_lexicon(req.lexicon_file)
                blended_results = []
                for t, inner in zip(texts, results):
                    scores = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
                    for k in ["negativ", "neutral", "positiv"]:
                        scores.setdefault(k, 0.0)
                    s_scalar = score_text(t, lex)
                    ln, le, lp = scalar_to_dist(s_scalar)
                    scores = blend_distributions(scores, (ln, le, lp), req.lexicon_weight)
                    blended_inner = [
                        {"label": "negativ", "score": scores["negativ"]},
                        {"label": "neutral", "score": scores["neutral"]},
                        {"label": "positiv", "score": scores["positiv"]},
                    ]
                    blended_results.append(blended_inner)
                results = blended_results
            except Exception:
                pass
        # package output similar to analyze_conversation
        seg_out: List[SegmentSentiment] = []
        for idx, (t, inner) in enumerate(zip(texts, results)):
            scores_map = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
            for k in ["negativ", "neutral", "positiv"]:
                scores_map.setdefault(k, 0.0)
            top_label = max(scores_map.items(), key=lambda kv: kv[1])[0]
            top_score = float(scores_map[top_label])
            start = float(segments[idx].get("start", 0.0) or 0.0) if idx < len(segments) else None
            end = float(segments[idx].get("end", 0.0) or 0.0) if idx < len(segments) else None
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
        return {"transcript": tr, "segment_sentiments": [s.model_dump() for s in seg_out], "meta": meta}

    # process batches
    for bidx, batch in enumerate(batches):
        if req.workers == 1:
            for p in batch:
                try:
                    data = _do_transcribe(p) if req.operation == "transcribe" else _do_analyze(p)
                    items.append(ScanItem(file=p, ok=True, data=data, batch_index=bidx))
                    ok += 1
                    # update state
                    processed[p] = {"mtime": os.path.getmtime(p), "when": datetime.utcnow().isoformat() + "Z"}
                except Exception as e:
                    items.append(ScanItem(file=p, ok=False, error=str(e), batch_index=bidx))
                    failed += 1
        else:
            def _wrap(pth: str):
                return (pth, (_do_transcribe(pth) if req.operation == "transcribe" else _do_analyze(pth)))
            with ThreadPoolExecutor(max_workers=req.workers) as ex:
                futs = {ex.submit(_wrap, p): p for p in batch}
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        _, data = fut.result()
                        items.append(ScanItem(file=p, ok=True, data=data, batch_index=bidx))
                        ok += 1
                        processed[p] = {"mtime": os.path.getmtime(p), "when": datetime.utcnow().isoformat() + "Z"}
                    except Exception as e:
                        items.append(ScanItem(file=p, ok=False, error=str(e), batch_index=bidx))
                        failed += 1

    # persist state
    state["processed"] = processed
    _save_state(req.state_file, state)

    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return ScanProcessResponse(items=items, ok=ok, failed=failed, total=len(new_files), batches=len(batches), timestamp=now_iso)
