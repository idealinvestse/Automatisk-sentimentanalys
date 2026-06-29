"""Build environment for child processes (API, CLI, Streamlit)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.install.config_schema import UserConfig
from src.install.paths_util import augment_path, resolve_ffmpeg
from src.install.secrets_win import apply_secrets_to_env
from src.install.user_config import config_to_env, load_user_config


def detect_app_root() -> Path:
    """Resolve application root from env or launcher package location."""
    override = os.environ.get("SENTIMENT_APP_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def _windows_user_env_var(name: str) -> str | None:
    """Read a persisted user environment variable from the Windows registry."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    text = str(value).strip()
    return text or None


def bootstrap_launcher_env(app_root: Path | None = None) -> Path:
    """Apply runtime env for launcher and child processes (Windows-friendly)."""
    root = (app_root or detect_app_root()).resolve()
    os.environ["SENTIMENT_APP_ROOT"] = str(root)
    os.environ["PYTHONPATH"] = str(root)

    if not os.environ.get("FFMPEG_PATH", "").strip() and (
        persisted := _windows_user_env_var("FFMPEG_PATH")
    ):
        os.environ["FFMPEG_PATH"] = persisted

    cfg = load_user_config(root)
    os.environ["PATH"] = augment_path(cfg, os.environ.get("PATH", ""))
    if ffmpeg := resolve_ffmpeg(cfg):
        os.environ.setdefault("FFMPEG_PATH", ffmpeg)
    return root


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
    apply_secrets_to_env(cfg.resolved_app_root())
    env = os.environ.copy()
    env.update(config_to_env(cfg))

    env["PATH"] = augment_path(cfg, env.get("PATH", ""))
    env["PYTHONPATH"] = str(cfg.resolved_app_root())
    if ffmpeg := resolve_ffmpeg(cfg):
        env.setdefault("FFMPEG_PATH", ffmpeg)

    return env


def working_directory(cfg: UserConfig) -> Path:
    return cfg.resolved_app_root()
