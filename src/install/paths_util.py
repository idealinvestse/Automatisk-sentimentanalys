"""Shared path resolution for config and tooling (ffmpeg, user config)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config_schema import UserConfig

_CONFIG_ENV = "SENTIMENT_USER_CONFIG"


def portable_user_config_path(app_root: Path) -> Path:
    return app_root.resolve() / "user_data" / "user_config.yaml"


def roaming_user_config_path() -> Path:
    return Path.home() / "AppData" / "Roaming" / "Sentimentanalys" / "user_config.yaml"


def resolve_user_config_path(
    app_root: Path,
    *,
    portable: bool | None = None,
) -> Path:
    """Pick user_config.yaml location (portable vs roaming vs override)."""
    override = os.environ.get(_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()

    root = app_root.resolve()
    local = portable_user_config_path(root)

    if portable is True:
        return local
    if portable is False:
        return roaming_user_config_path()

    # Auto-detect: portable bundle ships user_data/user_config.yaml
    if local.is_file():
        return local
    if os.environ.get("SENTIMENT_PORTABLE", "").strip().lower() in ("1", "true", "yes"):
        return local
    return roaming_user_config_path()


def _ffmpeg_exe_name() -> str:
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def augment_path(cfg: UserConfig, base_path: str | None = None) -> str:
    """PATH with bundled ffmpeg and venv Scripts (matches build_child_env)."""
    root = cfg.resolved_app_root()
    parts: list[str] = []
    ffmpeg_override = os.environ.get("FFMPEG_PATH", "").strip()
    if ffmpeg_override:
        ff_dir = str(Path(ffmpeg_override).expanduser().resolve().parent)
        parts.append(ff_dir)
    ffmpeg_bin = root / "tools" / "ffmpeg" / "bin"
    if ffmpeg_bin.is_dir():
        parts.append(str(ffmpeg_bin))
    scripts = root / ".venv" / "Scripts"
    if scripts.is_dir():
        parts.append(str(scripts))
    if base_path:
        parts.append(base_path)
    elif (existing := os.environ.get("PATH", "")):
        parts.append(existing)
    return os.pathsep.join(parts)


def resolve_ffmpeg(cfg: UserConfig) -> str | None:
    """Return path to ffmpeg executable (env override, bundle, then PATH)."""
    override = os.environ.get("FFMPEG_PATH", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())

    root = cfg.resolved_app_root()
    bundled = root / "tools" / "ffmpeg" / "bin" / _ffmpeg_exe_name()
    if bundled.is_file():
        return str(bundled)
    path = augment_path(cfg)
    return shutil.which(_ffmpeg_exe_name().removesuffix(".exe"), path=path)
