"""Shared helpers for the Typer CLI."""

from __future__ import annotations

import logging
import os

from rich.console import Console

from .core.audio import resolve_audio_paths as _core_resolve_audio
from .core.logging_config import configure_logging

console = Console()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    configure_logging()
    logging.getLogger().setLevel(getattr(logging, str(level).upper(), logging.INFO))


def resolve_audio_paths(inputs: list[str]) -> list[str]:
    return _core_resolve_audio(audio_paths=inputs)


def parse_asr_hotwords(
    hotwords: str | None,
    language: str,
    *,
    auto_load: bool = True,
) -> list[str] | None:
    parsed: list[str] | None = None
    if hotwords:
        parsed = [w.strip() for w in hotwords.replace(",", " ").split() if w.strip()]
    if parsed or not auto_load:
        return parsed
    if not language.lower().startswith("sv"):
        return None
    default_hw_path = os.path.join("configs", "callcenter_hotwords.txt")
    if not os.path.exists(default_hw_path):
        return None
    try:
        with open(default_hw_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if lines:
            console.print(
                f"[cyan]Auto-loaded {len(lines)} hotwords from {default_hw_path}[/cyan]"
            )
            return lines
    except Exception:
        pass
    return None
