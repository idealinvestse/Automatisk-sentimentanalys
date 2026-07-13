"""Shared path resolution for config and tooling (ffmpeg, user config)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config_schema import UserConfig

_CONFIG_ENV = "SENTIMENT_USER_CONFIG"
_LEGACY_DASHBOARD_PORTS = frozenset({8080, 8501})


def looks_like_app_root(path: Path) -> bool:
    """True when *path* contains a shippable project tree (pyproject + launcher/src)."""
    root = path.resolve()
    if not (root / "pyproject.toml").is_file():
        return False
    return (root / "launcher").is_dir() or (root / "src").is_dir()


def find_app_root_near(path: Path) -> Path | None:
    """Resolve project root at *path* or one nested child (workspace layout)."""
    root = path.resolve()
    if looks_like_app_root(root):
        return root
    if not root.is_dir():
        return None
    nested_named = root / "Automatisk-sentimentanalys"
    if looks_like_app_root(nested_named):
        return nested_named
    candidates: list[Path] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_dir() and looks_like_app_root(child):
            candidates.append(child)
    if not candidates:
        return None
    with_launcher = [c for c in candidates if (c / "launcher").is_dir()]
    return sorted(with_launcher or candidates, key=lambda p: p.name.lower())[0]


def heal_app_root(configured: Path, *, preferred: Path | None = None) -> Path:
    """Return a usable app root, preferring *preferred* when *configured* is wrong."""
    configured = configured.resolve()
    if looks_like_app_root(configured):
        return configured
    if preferred is not None:
        preferred = preferred.resolve()
        if looks_like_app_root(preferred):
            return preferred
        found_preferred = find_app_root_near(preferred)
        if found_preferred is not None:
            return found_preferred
    found = find_app_root_near(configured)
    if found is not None:
        return found
    return preferred.resolve() if preferred is not None else configured


def migrate_legacy_dashboard_settings(cfg: UserConfig) -> bool:
    """Normalize Streamlit/NiceGUI-era dashboard port/ui. Returns True if changed."""
    changed = False
    if cfg.services.dashboard_port in _LEGACY_DASHBOARD_PORTS:
        cfg.services.dashboard_port = 3000
        changed = True
    if cfg.services.dashboard_ui != "webui":
        cfg.services.dashboard_ui = "webui"
        changed = True
    return changed


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
    elif existing := os.environ.get("PATH", ""):
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
