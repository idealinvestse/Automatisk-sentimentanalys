"""ASR setup helpers for the Windows launcher (GUI + CLI)."""

from __future__ import annotations

from collections.abc import Callable

from src.install.asr_assets import (
    DEFAULT_PREFETCH_BACKENDS,
    AsrAssetReport,
    AsrStatus,
    collect_asr_status,
    ensure_asr_assets,
)
from src.install.config_schema import UserConfig

from .env_builder import resolve_python

ProgressCallback = Callable[[str], None] | None


def asr_status_for_config(cfg: UserConfig) -> AsrStatus:
    """ASR package/model status using paths from user configuration."""
    return collect_asr_status(
        model=cfg.asr.model,
        hf_home=cfg.resolved_hf_home(),
    )


def run_asr_setup(
    cfg: UserConfig,
    *,
    backends: list[str] | None = None,
    install_packages: bool = True,
    download_models: bool = True,
    progress: ProgressCallback = None,
) -> AsrAssetReport:
    """Install ASR pip packages and/or download transcription models."""
    root = cfg.resolved_app_root()
    python = resolve_python(cfg)
    device = cfg.device if cfg.device != "auto" else "cpu"
    return ensure_asr_assets(
        root,
        python=python,
        backends=backends or list(DEFAULT_PREFETCH_BACKENDS),
        model=cfg.asr.model,
        device=device,
        language=cfg.asr.language,
        revision=cfg.asr.revision,
        hf_home=cfg.resolved_hf_home(),
        install_packages=install_packages,
        download_models=download_models,
        progress=progress,
    )


def format_asr_report_lines(report: AsrAssetReport) -> list[tuple[bool, str]]:
    """Format report steps for launcher activity log."""
    lines: list[tuple[bool, str]] = []
    for step in report.steps:
        msg = f"{step.name}: {step.message}"
        if step.detail:
            msg += f" ({step.detail})"
        lines.append((step.ok, msg))
    return lines
