from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .sentiment import analyze_smart


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
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return AnalyzeResponse(meta=meta, timestamp=now_iso, results=results)
