from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime

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
