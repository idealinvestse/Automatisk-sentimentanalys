"""Tests for launcher dashboard dependency checks (Next.js webui)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from launcher.dashboard_deps import (
    check_dashboard_dependencies,
    check_webui_package,
    missing_dashboard_modules,
)


def test_missing_dashboard_modules_empty_when_installed() -> None:
    # On CI/dev machines with Node installed this is empty; tolerate either.
    missing = missing_dashboard_modules()
    assert isinstance(missing, list)


def test_check_dashboard_dependencies_returns_none_when_ok() -> None:
    with (
        patch("launcher.dashboard_deps.missing_dashboard_modules", return_value=[]),
        patch("launcher.dashboard_deps.check_webui_package", return_value=None),
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


def test_check_webui_package_missing_package_json(tmp_path: Path) -> None:
    err = check_webui_package(cwd=tmp_path)
    assert err is not None
    assert "package.json" in err


def test_check_webui_package_missing_node_modules(tmp_path: Path) -> None:
    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "package.json").write_text("{}", encoding="utf-8")
    err = check_webui_package(cwd=tmp_path)
    assert err is not None
    assert "node_modules" in err
    assert "npm install" in err


def test_check_webui_package_ok(tmp_path: Path) -> None:
    webui = tmp_path / "webui"
    (webui / "node_modules").mkdir(parents=True)
    (webui / "package.json").write_text("{}", encoding="utf-8")
    assert check_webui_package(cwd=tmp_path) is None
