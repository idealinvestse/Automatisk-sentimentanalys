"""Tests for POST /upload endpoint (Fas 6 hardening)."""

from __future__ import annotations

import io
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

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

    with patch("src.api.routers.transcription.validate_audio_path") as mock_validate:
        mock_validate.return_value = "/tmp/test_uploads/uploads/test.wav"

        response = client.post(
            "/upload", files={"file": ("test.wav", io.BytesIO(large_content), "audio/wav")}
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

    with patch("src.api.routers.transcription.validate_audio_path") as mock_validate:
        mock_validate.return_value = "/tmp/test_uploads/uploads/test.wav"

        response = client.post(
            "/upload", files={"file": ("test.wav", io.BytesIO(content), "audio/wav")}
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
        "/upload", files={"file": ("test.exe", io.BytesIO(b"fake"), "application/octet-stream")}
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
        "/upload", files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")}
    )

    assert response.status_code == 500
    assert "media_root not set" in response.json()["detail"].lower()


def test_upload_uses_uuid4_in_filename(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """POST /upload generates unique filenames with uuid4 prefix."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media_root))
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()

    client = TestClient(create_app())
    content = b"fake audio"

    # Echo the saved path so the uuid prefix is not discarded by a hardcoded mock.
    with patch(
        "src.api.routers.transcription.validate_audio_path",
        side_effect=lambda path: path,
    ):
        response = client.post(
            "/upload",
            files={"file": ("test.wav", io.BytesIO(content), "audio/wav")},
        )

    assert response.status_code == 200
    audio_path = response.json()["audio_path"]
    filename = Path(audio_path).name
    assert re.match(r"^[a-f0-9]{12}_test\.wav$", filename)
    assert (media_root / "uploads" / filename).is_file()


def test_upload_requires_api_key_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media_root))
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()

    client = TestClient(create_app())
    denied = client.post(
        "/upload",
        files={"file": ("test.wav", io.BytesIO(b"x"), "audio/wav")},
    )
    assert denied.status_code == 401

    with patch(
        "src.api.routers.transcription.validate_audio_path",
        side_effect=lambda path: path,
    ):
        ok = client.post(
            "/upload",
            headers={"X-API-Key": "secret"},
            files={"file": ("test.wav", io.BytesIO(b"x"), "audio/wav")},
        )
    assert ok.status_code == 200


def test_upload_sanitizes_path_traversal_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media_root))
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()

    client = TestClient(create_app())
    with patch(
        "src.api.routers.transcription.validate_audio_path",
        side_effect=lambda path: path,
    ):
        response = client.post(
            "/upload",
            files={"file": ("../../escape.wav", io.BytesIO(b"payload"), "audio/wav")},
        )

    assert response.status_code == 200
    audio_path = Path(response.json()["audio_path"])
    assert audio_path.parent == media_root / "uploads"
    assert ".." not in audio_path.name
    assert audio_path.is_file()
    # Nothing written outside uploads/
    assert not (tmp_path / "escape.wav").exists()
    assert list((media_root / "uploads").iterdir())


def test_upload_save_failure_uses_public_error_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media_root))
    from src.api.error_responses import PUBLIC_ERROR_DETAIL
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()

    client = TestClient(create_app())
    with patch("builtins.open", side_effect=OSError("disk full: /secret/path")):
        response = client.post(
            "/upload",
            files={"file": ("test.wav", io.BytesIO(b"x"), "audio/wav")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == PUBLIC_ERROR_DETAIL
    assert "disk full" not in response.json()["detail"].lower()
    assert "/secret" not in response.json()["detail"]


def test_upload_end_to_end_validates_under_media_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """POST /upload succeeds with real path validation (no mocks)."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media_root))
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/upload",
        files={"file": ("call.wav", io.BytesIO(b"RIFF....data"), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert Path(data["audio_path"]).is_file()
    assert Path(data["audio_path"]).parent == media_root / "uploads"
    assert data["filename"] == "call.wav"


def test_upload_missing_file_returns_422(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """POST /upload without multipart file field is a validation error (422)."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media_root))
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post("/upload")
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_upload_rejects_empty_filename_extension(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media_root))
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()

    client = TestClient(create_app())
    response = client.post(
        "/upload",
        files={"file": ("noext", io.BytesIO(b"x"), "application/octet-stream")},
    )
    assert response.status_code == 400
