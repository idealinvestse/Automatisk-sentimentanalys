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

from .asr_assets import ensure_asr_assets
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


def extras_for_profile(profile: InstallProfile) -> list[str]:
    """Optional dependency extras to install for a given profile (pyproject.toml)."""
    mapping: dict[InstallProfile, list[str]] = {
        InstallProfile.minimal: ["min", "install"],
        InstallProfile.cli: [
            "min",
            "cli",
            "asr",
            "api",
            "install",
        ],
        InstallProfile.api: ["min", "asr", "api", "install"],
        InstallProfile.full: [
            "min",
            "cli",
            "asr",
            "api",
            "llm",
            "training",
            "semantic",
            "install",
        ],
        InstallProfile.dev: [
            "min",
            "cli",
            "asr",
            "api",
            "llm",
            "training",
            "semantic",
            "install",
            "dev",
            "diarize",
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


_TORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
# whisperx/pyannote 3.x need torchaudio.AudioMetaData (removed in torchaudio 2.9+).
_TORCH_WHISPERX_PINS = ("torch==2.8.0", "torchaudio==2.8.0")
_PIP_ERROR_TAIL_CHARS = 1500
_TORCH_REQ_PREFIXES = ("torch", "torchaudio")
_ACCESS_DENIED_HINT = (
    "WinError 5 (Access is denied) på torch-DLL: stäng ALLA Sentimentanalys-fönster "
    "och andra Python-processer som använder .venv, starta sedan launchern igen och "
    "kör Installera/reparera — eller från PowerShell: .\\launcher.ps1 provision"
)


def _subprocess_no_window_kwargs() -> dict[str, int]:
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    return {}


def cleanup_pip_leftovers(site_packages: Path) -> list[str]:
    """Remove Windows pip leftover dirs (``~orch…``) from interrupted uninstalls."""
    if not site_packages.is_dir():
        return []
    removed: list[str] = []
    for entry in site_packages.iterdir():
        if not entry.name.startswith("~"):
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except OSError:
                continue
        if not entry.exists():
            removed.append(entry.name)
    return removed


def site_packages_for_python(python: Path) -> Path:
    """Best-effort site-packages path for a venv interpreter."""
    if sys.platform == "win32":
        return python.resolve().parent.parent / "Lib" / "site-packages"
    return (
        python.resolve().parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _format_pip_failure(args: list[str], returncode: int, detail: str) -> str:
    msg = f"pip {' '.join(args)} failed (exit {returncode}):\n{detail}"
    lower = detail.lower()
    if "access is denied" in lower or "winerror 5" in lower:
        msg = f"{msg}\n\n{_ACCESS_DENIED_HINT}"
    return msg


def _run_pip(python: Path, root: Path, args: list[str]) -> None:
    result = subprocess.run(  # type: ignore[call-overload]
        [str(python), "-m", "pip", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        **_subprocess_no_window_kwargs(),
    )
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip()
    if len(detail) > _PIP_ERROR_TAIL_CHARS:
        detail = detail[-_PIP_ERROR_TAIL_CHARS:]
    raise RuntimeError(_format_pip_failure(args, result.returncode, detail))


def _nvidia_smi_available() -> bool:
    try:
        result = subprocess.run(  # type: ignore[call-overload]
            ["nvidia-smi"],
            capture_output=True,
            check=False,
            **_subprocess_no_window_kwargs(),
        )
    except OSError:
        return False
    return bool(result.returncode == 0)


def probe_cuda_torch(python: Path) -> dict[str, str | bool] | None:
    """Return torch/torchaudio versions when CUDA torch is whisperx-compatible."""
    script = (
        "import json,torch,torchaudio;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'torchaudio': torchaudio.__version__,"
        "'cuda': bool(torch.cuda.is_available()),"
        "'audiometadata': hasattr(torchaudio,'AudioMetaData')"
        "}))"
    )
    result = subprocess.run(  # type: ignore[call-overload]
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        check=False,
        **_subprocess_no_window_kwargs(),
    )
    if result.returncode != 0:
        return None
    try:
        import json

        data = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError, TypeError):
        return None
    if not isinstance(data, dict) or not data.get("cuda"):
        return None
    # torchaudio>=2.9 breaks whisperx/pyannote 3.x (no AudioMetaData).
    if not data.get("audiometadata"):
        return None
    torch_ver = str(data.get("torch", ""))
    return {
        "torch": torch_ver,
        "torchaudio": str(data.get("torchaudio", "")),
        "cuda": True,
    }


def _is_torch_requirement(req: str) -> bool:
    name = req.strip().lower()
    for prefix in _TORCH_REQ_PREFIXES:
        if (
            name == prefix
            or name.startswith(f"{prefix}=")
            or name.startswith(f"{prefix}[")
            or name.startswith(f"{prefix}~")
            or name.startswith(f"{prefix}>")
            or name.startswith(f"{prefix}<")
            or name.startswith(f"{prefix}!")
        ):
            return True
    return False


def extra_requirements_from_pyproject(root: Path, extras: list[str]) -> list[str]:
    """Flatten optional-dependency requirements for *extras* from pyproject.toml."""
    import tomllib

    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    optional = data.get("project", {}).get("optional-dependencies", {})
    seen: set[str] = set()
    out: list[str] = []
    for extra in extras:
        for req in optional.get(extra, []):
            key = req.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def ensure_cuda_torch(
    root: Path,
    python: Path,
    *,
    progress: ProgressCallback = None,
) -> str | None:
    """Reinstall torch/torchaudio from the CUDA wheel index when an NVIDIA GPU is present.

    Plain ``pip install -e '.[asr]'`` often pulls CPU wheels (or downgrades a
    previous ``+cu*`` build via whisperx pins). Restore a CUDA build afterwards.
    Skips when CUDA torch already imports cleanly (avoids WinError 5 on locked DLLs).
    """
    index = os.environ.get("SENTIMENT_TORCH_INDEX", "").strip() or _TORCH_CU128_INDEX
    if not _nvidia_smi_available() and not os.environ.get("SENTIMENT_TORCH_INDEX", "").strip():
        return None
    existing = probe_cuda_torch(python)
    if existing:
        if progress:
            progress(f"CUDA torch already OK ({existing['torch']}); skipping reinstall")
        return f"already:{existing['torch']}"
    if progress:
        progress(
            f"Installing CUDA torch/torchaudio ({', '.join(_TORCH_WHISPERX_PINS)}) from {index}"
        )
    cleanup_pip_leftovers(site_packages_for_python(python))
    _run_pip(
        python,
        root,
        ["install", "--upgrade", *_TORCH_WHISPERX_PINS, "--index-url", index],
    )
    return index


def install_requirements(root: Path, python: Path, profile: InstallProfile) -> list[str]:
    """Install optional dependency extras for profile via editable pyproject.toml install."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"Missing pyproject.toml at {pyproject}")

    leftovers = cleanup_pip_leftovers(site_packages_for_python(python))
    extras = extras_for_profile(profile)
    _run_pip(python, root, ["install", "-U", "pip", "wheel"])

    index = ""
    if _nvidia_smi_available() or os.environ.get("SENTIMENT_TORCH_INDEX", "").strip():
        index = os.environ.get("SENTIMENT_TORCH_INDEX", "").strip() or _TORCH_CU128_INDEX

    # When CUDA torch already works, avoid pip replacing the locked Windows .pyd
    # (whisperx pins torch~=2.8 and would otherwise force a reinstall → WinError 5).
    cuda_torch = probe_cuda_torch(python)
    if cuda_torch:
        _run_pip(python, root, ["install", "-e", ".", "--no-deps"])
        reqs = [
            r
            for r in extra_requirements_from_pyproject(root, extras)
            if not _is_torch_requirement(r)
        ]
        if reqs:
            args = ["install", *reqs]
            if index:
                args.extend(["--extra-index-url", index])
            _run_pip(python, root, args)
    else:
        extras_args = ["install", "-e", f".[{','.join(extras)}]"]
        if index:
            extras_args.extend(["--extra-index-url", index])
        _run_pip(python, root, extras_args)

    if leftovers:
        cleanup_pip_leftovers(site_packages_for_python(python))
    return extras


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


def ensure_webui_deps(root: Path, *, progress: ProgressCallback = None) -> str:
    """Ensure Node/npm are available and install webui dependencies when missing."""
    missing: list[str] = []
    if shutil.which("node") is None:
        missing.append("node")
    if shutil.which("npm") is None:
        missing.append("npm")
    if missing:
        raise RuntimeError(
            f"Dashboard-beroenden saknas ({', '.join(missing)}). "
            "Installera Node.js (inkl. npm), sedan: cd webui && npm install"
        )

    webui_dir = root / "webui"
    package_json = webui_dir / "package.json"
    if not package_json.is_file():
        raise RuntimeError(f"webui/package.json saknas under {root}")

    node_modules = webui_dir / "node_modules"
    if node_modules.is_dir():
        return f"webui deps already present ({node_modules})"

    npm = shutil.which("npm")
    assert npm is not None
    if progress:
        progress("Installing webui npm dependencies")
    lockfile = webui_dir / "package-lock.json"
    cmd = [npm, "ci"] if lockfile.is_file() else [npm, "install"]
    subprocess.run(cmd, check=True, cwd=str(webui_dir))
    return f"Installed webui dependencies via {' '.join(cmd)}"


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
    download_asr: bool = True,
    install_webui: bool = True,
    init_config: bool = True,
    progress: ProgressCallback = None,
) -> ProvisionReport:
    """Install venv, pip packages, ffmpeg, ASR assets, webui npm deps, and optional user config."""
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
            detail = ", ".join(installed) if installed else "no extras configured"
            report.add("pip", True, "Python packages installed", detail)
            cfg.install_profile = profile
            save_user_config(cfg)
        except Exception as exc:
            report.add("pip", False, "pip install failed", str(exc))
            return report

        try:
            cuda_index = ensure_cuda_torch(root, python, progress=log)
            if cuda_index:
                report.add("torch_cuda", True, "CUDA torch/torchaudio installed", cuda_index)
            else:
                report.add(
                    "torch_cuda",
                    True,
                    "CUDA torch hoppades över (ingen NVIDIA GPU / SENTIMENT_TORCH_INDEX)",
                )
        except Exception as exc:
            report.add("torch_cuda", False, "CUDA torch install failed", str(exc))
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

    if download_asr and install_packages:
        log("Installing ASR packages and downloading transcription models")
        try:
            asr_report = ensure_asr_assets(
                root,
                python=python,
                backends=["faster", "whisperx", "transformers"],
                model=cfg.asr.model,
                device=cfg.device if cfg.device != "auto" else "cpu",
                language=cfg.asr.language,
                revision=cfg.asr.revision,
                hf_home=cfg.resolved_hf_home(),
                install_packages=True,
                download_models=True,
                progress=log,
            )
            for step in asr_report.steps:
                report.add(f"asr_{step.name}", step.ok, step.message, step.detail)
        except Exception as exc:
            report.add("asr_assets", False, "ASR setup failed", str(exc))

    if install_webui and cfg.services.dashboard_enabled:
        log("Checking Node.js and installing webui dependencies")
        try:
            detail = ensure_webui_deps(root, progress=log)
            report.add("webui", True, "Webui dependencies ready", detail)
        except Exception as exc:
            report.add("webui", False, "Webui setup failed", str(exc))

    return report
