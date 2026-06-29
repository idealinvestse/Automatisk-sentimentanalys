"""Tests for launcher dashboard dependency checks."""

from __future__ import annotations

from unittest.mock import patch

from launcher.dashboard_deps import check_dashboard_dependencies, missing_dashboard_modules


def test_missing_dashboard_modules_empty_when_installed() -> None:
    assert missing_dashboard_modules() == []


def test_check_dashboard_dependencies_returns_none_when_ok() -> None:
    assert check_dashboard_dependencies() is None


def test_check_dashboard_dependencies_reports_missing_modules() -> None:
    with (
        patch("launcher.dashboard_deps.missing_dashboard_modules", return_value=["nicegui"]),
        patch("launcher.dashboard_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 1
        err = check_dashboard_dependencies()
    assert err is not None
    assert "nicegui" in err
    assert "provision" in err
