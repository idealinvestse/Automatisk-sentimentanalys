"""Ingest router for YouTube and other data sources.

Provides endpoints to download audio from YouTube, list ingested files,
and trigger downstream transcription/analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, HttpUrl, Field

# Import from the new data_ingestion module
try:
    from ...data_ingestion.youtube_downloader import YouTubeAudioDownloader, DownloadResult
except ImportError:
    # Fallback for development
    from src.data_ingestion.youtube_downloader import YouTubeAudioDownloader, DownloadResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest/youtube", tags=["Data Ingestion - YouTube"])


# --- Pydantic Schemas (can be moved to schemas.py later) ---

class YouTubeDownloadRequest(BaseModel):
    url: HttpUrl = Field(..., description="YouTube video or playlist URL")
    playlist: bool = Field(False, description="Download entire playlist")
    convert_to_wav: bool = Field(True, description="Convert to 16kHz mono WAV for ASR")
    sample_rate: int = Field(16000, ge=8000, le=48000)
    channels: int = Field(1, ge=1, le=2)
    auto_transcribe: bool = Field(False, description="Automatically start transcription after download")
    auto_analyze: bool = Field(False, description="Automatically run full analysis pipeline after transcription")


class DownloadResponse(BaseModel):
    success: bool
    file_path: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None


class IngestedFile(BaseModel):
    file_path: str
    title: str
    duration_seconds: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


# --- Helper to run pipeline after download (placeholder - integrate with existing pipeline) ---

def _run_pipeline_after_download(wav_path: Path, auto_analyze: bool = False):
    """Background task to trigger transcription and optional analysis."""
    logger.info(f"Triggering pipeline for downloaded file: {wav_path}")
    # TODO: Integrate with existing CallAnalysisPipeline or call /transcribe and /analyze endpoints
    # For now, just log. In full implementation:
    # from src.pipeline import CallAnalysisPipeline
    # pipeline = CallAnalysisPipeline()
    # result = pipeline.run(str(wav_path), ...)
    print(f"[INGEST] Pipeline triggered for {wav_path} (analyze={auto_analyze})")


@router.post("/download", response_model=DownloadResponse, summary="Download audio from YouTube")
async def download_youtube(
    request: YouTubeDownloadRequest,
    background_tasks: BackgroundTasks,
):
    """Download best audio from a YouTube URL (or playlist).

    Converts to project-standard 16kHz mono WAV by default.
    Optionally triggers transcription/analysis in background.
    """
    try:
        downloader = YouTubeAudioDownloader()
        result = await downloader.adownload(
            str(request.url),
            playlist=request.playlist,
            convert_to_wav=request.convert_to_wav,
            sample_rate=request.sample_rate,
            channels=request.channels,
        )

        if isinstance(result, list):
            # Playlist case - return summary
            successful = [r for r in result if r.success]
            return DownloadResponse(
                success=len(successful) > 0,
                message=f"Downloaded {len(successful)}/{len(result)} videos from playlist",
                metadata={"count": len(successful)},
            )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Download failed")

        file_path = result.file_path

        if request.auto_transcribe or request.auto_analyze:
            background_tasks.add_task(
                _run_pipeline_after_download,
                file_path,
                auto_analyze=request.auto_analyze,
            )

        return DownloadResponse(
            success=True,
            file_path=str(file_path) if file_path else None,
            metadata=result.metadata,
            message="Download successful" + (" + pipeline queued" if (request.auto_transcribe or request.auto_analyze) else ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("YouTube download failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/list", response_model=List[IngestedFile], summary="List previously ingested YouTube files")
async def list_ingested(
    limit: int = Query(50, ge=1, le=500),
):
    """List recently downloaded YouTube files with metadata."""
    base = Path("data/ingested/youtube")
    if not base.exists():
        return []

    files = []
    for json_file in sorted(base.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(json_file, encoding="utf-8") as f:
                meta = json.load(f)
            files.append(
                IngestedFile(
                    file_path=meta.get("file_path", ""),
                    title=meta.get("title", json_file.stem),
                    duration_seconds=meta.get("duration_seconds"),
                    metadata=meta,
                )
            )
        except Exception:
            continue
    return files


@router.delete("/{youtube_id}", summary="Delete an ingested file and its metadata")
async def delete_ingested(youtube_id: str):
    """Delete WAV/JSON pair by YouTube ID."""
    base = Path("data/ingested/youtube")
    deleted = []
    for pattern in [f"*[{youtube_id}]*.wav", f"*[{youtube_id}]*.json", f"*[{youtube_id}]*.m4a", f"*[{youtube_id}]*.webm"]:
        for f in base.glob(pattern):
            f.unlink(missing_ok=True)
            deleted.append(str(f))
    if not deleted:
        raise HTTPException(status_code=404, detail="No files found for this ID")
    return {"deleted": deleted}
