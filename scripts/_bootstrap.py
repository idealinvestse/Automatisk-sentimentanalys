"""Shared bootstrap for CLI scripts."""

from __future__ import annotations

import logging

from src.core.logging_config import configure_logging


def bootstrap_script(*, verbose: bool = False, level: str | None = None) -> None:
    """Initialize logging for standalone scripts."""
    configure_logging()
    root = logging.getLogger()
    if verbose:
        root.setLevel(logging.DEBUG)
    elif level:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
