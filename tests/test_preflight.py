"""Tests for preflight / doctor checks."""

from __future__ import annotations

from src.install.config_schema import InstallProfile, UserConfig
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


def test_preflight_api_checks_when_enabled() -> None:
    cfg = UserConfig(services={"api_enabled": True})
    report = run_preflight(cfg, require_torch=False)
    names = [c.name for c in report.checks]
    assert "import_fastapi" in names
    assert "import_uvicorn" in names
    assert "import_src_api" in names
    assert any(c.name == "import_src_api" and c.ok for c in report.checks)


def test_preflight_skips_api_checks_when_disabled() -> None:
    cfg = UserConfig(services={"api_enabled": False})
    report = run_preflight(cfg, require_torch=False)
    names = [c.name for c in report.checks]
    assert "import_fastapi" not in names
    assert "import_uvicorn" not in names


def test_preflight_semantic_checks_for_non_minimal_profile() -> None:
    cfg = UserConfig(install_profile=InstallProfile.full)
    report = run_preflight(cfg, require_torch=False)
    names = [c.name for c in report.checks]
    assert "semantic_sentence_transformers" in names
    assert "semantic_faiss" in names
    assert "semantic_hdbscan" in names
    for check in report.checks:
        if check.name.startswith("semantic_"):
            assert check.ok


def test_preflight_skips_semantic_checks_for_minimal_profile() -> None:
    cfg = UserConfig(install_profile=InstallProfile.minimal)
    report = run_preflight(cfg, require_torch=False)
    names = [c.name for c in report.checks]
    assert not any(name.startswith("semantic_") for name in names)
