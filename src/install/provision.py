"""Download and install runtime dependencies (venv, pip, ffmpeg)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen

from .config_schema import InstallProfile, UserConfig
from .paths_util import resolve_ffmpeg
from .user_config import load_user_config, save_user_config

ProgressCallback = Callable[[str], None] | None

_FFMPEG_WIN64_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip"
)
_USER_AGENT = "Sentimentanalys-provision/1.0"


@dataclass
class ProvisionStep:
    name: str
    ok: bool
    message: str
    detail: str = ""


@dataclass
class ProvisionReport:
    steps: list[ProvisionStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)

    def add(self, name: str, ok: bool, message: str, detail: str = "") -> None:
        self.steps.append(ProvisionStep(name=name, ok=ok, message=message, detail=detail))


def requirements_for_profile(profile: InstallProfile) -> list[str]:
    """Requirement files to install for a given profile."""
    install = "requirements-install.txt"
    mapping: dict[InstallProfile, list[str]] = {
        InstallProfile.minimal: ["requirements-min.txt", install],
        InstallProfile.cli: [
            "requirements-min.txt",
            "requirements-cli.txt",
            "requirements-api.txt",
            "requirements-dashboard-nicegui.txt",
            install,
        ],
        InstallProfile.api: [
            "requirements-min.txt",
            "requirements-api.txt",
            install,
        ],
        InstallProfile.full: [
            "requirements-min.txt",
            "requirements-cli.txt",
            "requirements-api.txt",
            "requirements-dashboard-nicegui.txt",
            "requirements.txt",
            "requirements-desktop.txt",
            install,
        ],
        InstallProfile.dev: [
            "requirements-min.txt",
            "requirements-cli.txt",
            "requirements-api.txt",
            "requirements-dashboard-nicegui.txt",
            "requirements.txt",
            "requirements-desktop.txt",
            install,
            "requirements-dev.txt",
        ],
    }
    return mapping[profile]


def venv_python_path(root: Path) -> Path:
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def resolve_bootstrap_python(root: Path) -> Path:
    """Pick interpreter used to create or run the project venv."""
    venv_py = venv_python_path(root)
    if venv_py.is_file():
        return venv_py
    override = os.environ.get("SENTIMENT_PYTHON", "").strip()
    if override:
        return Path(override)
    return Path(sys.executable)


def ensure_venv(root: Path, *, python: Path | None = None) -> Path:
    """Create .venv when missing and return the venv python executable."""
    venv_py = venv_python_path(root)
    if venv_py.is_file():
        return venv_py

    creator = python or Path(sys.executable)
    subprocess.run(
        [str(creator), "-m", "venv", str(root / ".venv")],
        check=True,
        cwd=str(root),
    )
    if not venv_py.is_file():
        raise RuntimeError(f"Virtual environment was not created at {venv_py}")
    return venv_py


def _run_pip(python: Path, root: Path, args: list[str]) -> None:
    subprocess.run(
        [str(python), "-m", "pip", *args],
        check=True,
        cwd=str(root),
    )


def install_requirements(root: Path, python: Path, profile: InstallProfile) -> list[str]:
    """Install pip requirement bundles for profile. Returns installed file names."""
    _run_pip(python, root, ["install", "-U", "pip", "wheel"])
    installed: list[str] = []
    for req_file in requirements_for_profile(profile):
        path = root / req_file
        if not path.is_file():
            continue
        _run_pip(python, root, ["install", "-r", str(path)])
        installed.append(req_file)
    return installed


def bundled_ffmpeg_path(root: Path) -> Path:
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    return root / "tools" / "ffmpeg" / "bin" / name


def _download_file(url: str, dest: Path, *, timeout: float = 300.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_ffmpeg_binaries(zip_path: Path, dest_bin: Path) -> Path:
    dest_bin.mkdir(parents=True, exist_ok=True)
    wanted = {"ffmpeg.exe", "ffprobe.exe"}
    found: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            base = Path(name).name.lower()
            if base not in wanted:
                continue
            target = dest_bin / base
            with archive.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            found.add(base)

    ffmpeg_exe = dest_bin / "ffmpeg.exe"
    if not ffmpeg_exe.is_file():
        raise RuntimeError("ffmpeg.exe not found in downloaded archive")
    if "ffprobe.exe" not in found:
        raise RuntimeError("ffprobe.exe not found in downloaded archive")
    return ffmpeg_exe


def ensure_ffmpeg(root: Path, cfg: UserConfig) -> str | None:
    """Download bundled ffmpeg on Windows when no executable is available."""
    existing = resolve_ffmpeg(cfg)
    if existing:
        return existing

    if sys.platform != "win32":
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg via your package manager or set FFMPEG_PATH."
        )

    dest_bin = root / "tools" / "ffmpeg" / "bin"
    with tempfile.TemporaryDirectory(prefix="sentiment-ffmpeg-") as tmp:
        zip_path = Path(tmp) / "ffmpeg.zip"
        _download_file(_FFMPEG_WIN64_URL, zip_path)
        ffmpeg_exe = _extract_ffmpeg_binaries(zip_path, dest_bin)

    os.environ["FFMPEG_PATH"] = str(ffmpeg_exe)
    os.environ["PATH"] = os.pathsep.join([str(dest_bin), os.environ.get("PATH", "")])
    return str(ffmpeg_exe)


def ensure_user_config(root: Path) -> UserConfig:
    """Ensure user_config.yaml exists with app_root set."""
    cfg = load_user_config(root, create_if_missing=True)
    if not cfg.paths.app_root:
        cfg.paths.app_root = str(root.resolve())
        save_user_config(cfg)
    return cfg


def run_provision(
    cfg: UserConfig,
    profile: InstallProfile,
    *,
    ensure_virtualenv: bool = True,
    install_packages: bool = True,
    download_ffmpeg: bool = True,
    init_config: bool = True,
    progress: ProgressCallback = None,
) -> ProvisionReport:
    """Install venv, pip packages, ffmpeg, and optional user config."""
    report = ProvisionReport()
    root = cfg.resolved_app_root()

    def log(message: str) -> None:
        if progress:
            progress(message)

    if init_config:
        log("Creating user configuration if missing")
        try:
            cfg = ensure_user_config(root)
            report.add("config", True, "User configuration ready")
        except Exception as exc:
            report.add("config", False, "Failed to create user configuration", str(exc))
            return report

    python = resolve_bootstrap_python(root)
    if ensure_virtualenv:
        log("Ensuring virtual environment")
        try:
            python = ensure_venv(root, python=python)
            report.add("venv", True, f"Virtual environment: {python}")
        except Exception as exc:
            report.add("venv", False, "Failed to create virtual environment", str(exc))
            return report

    if install_packages:
        log(f"Installing pip packages for profile '{profile.value}'")
        try:
            installed = install_requirements(root, python, profile)
            detail = ", ".join(installed) if installed else "no requirement files found"
            report.add("pip", True, "Python packages installed", detail)
            cfg.install_profile = profile
            save_user_config(cfg)
        except subprocess.CalledProcessError as exc:
            report.add("pip", False, "pip install failed", str(exc))
            return report
        except Exception as exc:
            report.add("pip", False, "pip install failed", str(exc))
            return report

    if download_ffmpeg:
        log("Checking ffmpeg")
        try:
            resolved = ensure_ffmpeg(root, cfg)
            if resolved:
                report.add("ffmpeg", True, "ffmpeg available", resolved)
            else:
                report.add("ffmpeg", False, "ffmpeg not found after install")
        except Exception as exc:
            report.add("ffmpeg", False, "ffmpeg install failed", str(exc))

    return report