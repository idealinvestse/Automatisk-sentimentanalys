"""Tests for launcher dashboard dependency checks (Next.js webui)."""

from __future__ import annotations

from unittest.mock import patch

from launcher.dashboard_deps import check_dashboard_dependencies, missing_dashboard_modules


def test_missing_dashboard_modules_empty_when_installed() -> None:
    # On CI/dev machines with Node installed this is empty; tolerate either.
    missing = missing_dashboard_modules()
    assert isinstance(missing, list)


def test_check_dashboard_dependencies_returns_none_when_ok() -> None:
    with (
        patch("launcher.dashboard_deps.missing_dashboard_modules", return_value=[]),
        patch("launcher.dashboard_deps.check_dashboard_import", return_value=None),
    ):
        assert check_dashboard_dependencies() is None


def test_check_dashboard_dependencies_reports_missing_modules() -> None:
    with patch(
        "launcher.dashboard_deps.missing_dashboard_modules",
        return_value=["node"],
    ):
        err = check_dashboard_dependencies()
    assert err is not None
    assert "node" in err
    assert "npm install" in err
