"""Tests for POST /upload endpoint (Fas 6 hardening)."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile

from src.api import app
from src.api.app import create_app


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    from src.api.settings import get_api_settings
    get_api_settings.cache_clear()


def _create_upload_file(filename: str, content: bytes = b"fake audio data") -> UploadFile:
    """Create a mock UploadFile for testing."""
    file = UploadFile(filename=filename)
    file._file = io.BytesIO(content)
    return file


def test_upload_rejects_large_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /upload returns 413 for files exceeding MAX_UPLOAD_SIZE_MB."""
    monkeypatch.setenv("API_MEDIA_ROOT", "/tmp/test_uploads")
    monkeypatch.setenv("API_MAX_UPLOAD_SIZE_MB", "1")
    from src.api.settings import get_api_settings
    get_api_settings.cache_clear()

    client = TestClient(create_app())
    
    # Create a 2 MB file (exceeds 1 MB limit)
    large_content = b"x" * (2 * 1024 * 1024)
    
    with patch("src.api.routers.transcription.resolve_and_validate_audio_paths") as mock_validate:
        mock_validate.return_value = ["/tmp/test_uploads/uploads/test.wav"]
        
        response = client.post(
            "/upload",
            files={"file": ("test.wav", io.BytesIO(large_content), "audio/wav")}
        )
    
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_upload_accepts_file_within_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /upload accepts files within MAX_UPLOAD_SIZE_MB."""
    monkeypatch.setenv("API_MEDIA_ROOT", "/tmp/test_uploads")
    monkeypatch.setenv("API_MAX_UPLOAD_SIZE_MB", "10")
    from src.api.settings import get_api_settings
    get_api_settings.cache_clear()

    client = TestClient(create_app())
    
    # Create a 1 MB file (within 10 MB limit)
    content = b"x" * (1 * 1024 * 1024)
    
    with patch("src.api.routers.transcription.resolve_and_validate_audio_paths") as mock_validate:
        mock_validate.return_value = ["/tmp/test_uploads/uploads/test.wav"]
        
        response = client.post(
            "/upload",
            files={"file": ("test.wav", io.BytesIO(content), "audio/wav")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "audio_path" in data
    assert data["size_bytes"] == len(content)


def test_upload_rejects_unsupported_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /upload returns 400 for unsupported file formats."""
    monkeypatch.setenv("API_MEDIA_ROOT", "/tmp/test_uploads")
    from src.api.settings import get_api_settings
    get_api_settings.cache_clear()

    client = TestClient(create_app())
    
    response = client.post(
        "/upload",
        files={"file": ("test.exe", io.BytesIO(b"fake"), "application/octet-stream")}
    )
    
    assert response.status_code == 400
    assert "unsupported file format" in response.json()["detail"].lower()


def test_upload_requires_media_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /upload returns 500 when API_MEDIA_ROOT is not configured."""
    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    from src.api.settings import get_api_settings
    get_api_settings.cache_clear()

    client = TestClient(create_app())
    
    response = client.post(
        "/upload",
        files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")}
    )
    
    assert response.status_code == 500
    assert "media_root not set" in response.json()["detail"].lower()


def test_upload_uses_uuid4_in_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /upload generates unique filenames with uuid4 prefix."""
    monkeypatch.setenv("API_MEDIA_ROOT", "/tmp/test_uploads")
    from src.api.settings import get_api_settings
    get_api_settings.cache_clear()

    client = TestClient(create_app())
    
    content = b"fake audio"
    
    with patch("src.api.routers.transcription.resolve_and_validate_audio_paths") as mock_validate:
        mock_validate.return_value = ["/tmp/test_uploads/uploads/test.wav"]
        
        response = client.post(
            "/upload",
            files={"file": ("test.wav", io.BytesIO(content), "audio/wav")}
        )
    
    assert response.status_code == 200
    # The returned audio_path should have a uuid-like prefix (12 hex chars)
    audio_path = response.json()["audio_path"]
    # Extract filename from path
    filename = audio_path.split("/")[-1]
    # Should start with 12 hex chars followed by underscore
    import re
    assert re.match(r"^[a-f0-9]{12}_", filename)
