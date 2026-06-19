"""Tests for YouTube data ingestion (Fas 6).

Mocks yt-dlp and ffmpeg to avoid real network/downloads during tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_ingestion.youtube_downloader import YouTubeAudioDownloader, DownloadResult


@pytest.fixture
def mock_yt_dlp(monkeypatch):
    """Mock yt_dlp.YoutubeDL to return fake info without downloading."""
    mock_ydl = MagicMock()
    mock_info = {
        "id": "fake123",
        "title": "Test Video Title",
        "uploader": "Test Channel",
        "duration": 120,
        "upload_date": "20260101",
        "ext": "m4a",
    }
    mock_ydl.extract_info.return_value = mock_info
    mock_ydl.__enter__.return_value = mock_ydl
    mock_ydl.__exit__.return_value = False

    with patch("src.data_ingestion.youtube_downloader.yt_dlp.YoutubeDL", return_value=mock_ydl):
        yield mock_ydl


@pytest.fixture
def mock_ffmpeg(monkeypatch):
    """Mock subprocess.run for ffmpeg so no real conversion happens."""
    with patch("src.data_ingestion.youtube_downloader.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        yield mock_run


def test_download_single_video(mock_yt_dlp, mock_ffmpeg, tmp_path):
    """Test basic single video download and WAV conversion."""
    downloader = YouTubeAudioDownloader(output_base=tmp_path)

    # Create a fake downloaded file that the code expects
    fake_audio = tmp_path / "_temp_yt_dlp" / "Test Video Title [fake123].m4a"
    fake_audio.parent.mkdir(parents=True, exist_ok=True)
    fake_audio.write_bytes(b"fake audio data")

    result = downloader.download("https://youtube.com/watch?v=fake123")

    assert isinstance(result, DownloadResult)
    assert result.success is True
    assert result.file_path is not None
    assert result.file_path.suffix == ".wav"
    assert "youtube_id" in result.metadata
    assert result.metadata["title"] == "Test Video Title"


def test_download_with_playlist_flag(mock_yt_dlp, mock_ffmpeg, tmp_path):
    """Test that playlist=True is handled (even if mocked)."""
    downloader = YouTubeAudioDownloader(output_base=tmp_path)
    fake_audio = tmp_path / "_temp_yt_dlp" / "Test Video Title [fake123].m4a"
    fake_audio.parent.mkdir(parents=True, exist_ok=True)
    fake_audio.write_bytes(b"fake audio data")

    result = downloader.download("https://youtube.com/playlist?list=fake", playlist=True)

    # In mocked case it should still return a result (playlist logic is there)
    assert isinstance(result, (DownloadResult, list))


def test_metadata_saved(mock_yt_dlp, mock_ffmpeg, tmp_path):
    """Ensure metadata JSON is written next to the audio file."""
    downloader = YouTubeAudioDownloader(output_base=tmp_path)
    fake_audio = tmp_path / "_temp_yt_dlp" / "Test Video Title [fake123].m4a"
    fake_audio.parent.mkdir(parents=True, exist_ok=True)
    fake_audio.write_bytes(b"fake audio data")

    result = downloader.download("https://youtube.com/watch?v=fake123")

    meta_file = result.file_path.with_suffix(".json")
    assert meta_file.exists()

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["source_url"].startswith("https://youtube.com")
    assert meta["project"] == "Automatisk-sentimentanalys"


def test_error_handling(monkeypatch, tmp_path):
    """Test graceful failure when yt-dlp raises error."""
    def raise_download_error(*args, **kwargs):
        from yt_dlp.utils import DownloadError
        raise DownloadError("Simulated download failure")

    mock_ydl = MagicMock(extract_info=raise_download_error)
    mock_ydl.__enter__.return_value = mock_ydl
    mock_ydl.__exit__.return_value = False

    monkeypatch.setattr(
        "src.data_ingestion.youtube_downloader.yt_dlp.YoutubeDL",
        lambda *a, **k: mock_ydl,
    )

    downloader = YouTubeAudioDownloader(output_base=tmp_path)
    result = downloader.download("https://youtube.com/watch?v=bad")

    assert result.success is False
    assert result.error is not None
