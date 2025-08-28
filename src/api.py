from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .sentiment import analyze_smart
from .lexicon import load_lexicon, score_text, scalar_to_dist, blend_distributions


app = FastAPI(title="Swedish Sentiment API", version="0.1.0")


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
