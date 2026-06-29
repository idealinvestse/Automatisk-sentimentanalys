"""Tests for launcher API dependency checks and host resolution."""

from __future__ import annotations

from unittest.mock import patch

from launcher.api_deps import check_api_dependencies, missing_api_modules
from launcher.process_util import resolve_connect_host


def test_resolve_connect_host_maps_bind_all() -> None:
    assert resolve_connect_host("0.0.0.0") == "127.0.0.1"
    assert resolve_connect_host("::") == "127.0.0.1"
    assert resolve_connect_host("127.0.0.1") == "127.0.0.1"


def test_missing_api_modules_empty_when_installed() -> None:
    assert missing_api_modules() == []


def test_check_api_dependencies_returns_none_when_ok() -> None:
    assert check_api_dependencies() is None


def test_check_api_dependencies_reports_missing_modules() -> None:
    with (
        patch("launcher.api_deps.missing_api_modules", return_value=["uvicorn"]),
        patch("launcher.api_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 1
        err = check_api_dependencies()
    assert err is not None
    assert "uvicorn" in err
    assert "provision" in err


def test_server_entrypoint_import() -> None:
    from src.api.server import app

    assert app.title == "Swedish Sentiment API"
