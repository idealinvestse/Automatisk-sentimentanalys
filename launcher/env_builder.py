"""Build environment for child processes (API, CLI, Streamlit)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.install.config_schema import UserConfig
from src.install.paths_util import augment_path
from src.install.secrets_win import apply_secrets_to_env
from src.install.user_config import config_to_env


def resolve_python(cfg: UserConfig) -> Path:
    root = cfg.resolved_app_root()
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        return venv_py
    override = os.environ.get("SENTIMENT_PYTHON", "").strip()
    if override:
        return Path(override)
    return Path(sys.executable)


def build_child_env(cfg: UserConfig) -> dict[str, str]:
    """Full environment dict for subprocess.Popen."""
    apply_secrets_to_env()
    env = os.environ.copy()
    env.update(config_to_env(cfg))

    env["PATH"] = augment_path(cfg, env.get("PATH", ""))
    env["PYTHONPATH"] = str(cfg.resolved_app_root())

    return env


def working_directory(cfg: UserConfig) -> Path:
    return cfg.resolved_app_root()
