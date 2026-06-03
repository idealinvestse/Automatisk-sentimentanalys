"""Tests for preflight / doctor checks."""

from __future__ import annotations

from src.install.config_schema import UserConfig
from src.install.preflight import run_preflight


def test_preflight_python_ok() -> None:
    report = run_preflight(UserConfig(), require_torch=False)
    names = [c.name for c in report.checks]
    assert "python_version" in names
    assert any(c.name == "python_version" and c.ok for c in report.checks)


def test_preflight_optional_openrouter() -> None:
    cfg = UserConfig(llm={"enabled": False})
    report = run_preflight(cfg, require_openrouter=False)
    or_check = next(c for c in report.checks if c.name == "openrouter_key")
    assert or_check.ok
