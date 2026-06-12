"""Installation health checks (doctor)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config_schema import UserConfig
from .paths_util import resolve_ffmpeg
from .secrets_win import secret_status


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    detail: str = ""


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, message: str, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, ok=ok, message=message, detail=detail))


def _check_python(report: PreflightReport) -> None:
    ver = sys.version_info
    ok = ver >= (3, 11)
    report.add(
        "python_version",
        ok,
        f"Python {ver.major}.{ver.minor}.{ver.micro}",
        "Requires Python 3.11+" if not ok else "",
    )


def _check_import(report: PreflightReport, mod: str) -> None:
    ok = importlib.util.find_spec(mod) is not None
    report.add(
        f"import_{mod.replace('.', '_')}",
        ok,
        f"Module {mod}",
        "pip install missing dependency" if not ok else "",
    )


def _check_ffmpeg(report: PreflightReport, cfg: UserConfig) -> None:
    resolved = resolve_ffmpeg(cfg)
    ok = resolved is not None
    report.add(
        "ffmpeg",
        ok,
        "ffmpeg available" if ok else "ffmpeg not found (required for ASR preprocess)",
        resolved or "",
    )


def _check_torch_cuda(report: PreflightReport) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            report.add("cuda", True, f"CUDA available: {name}")
        else:
            report.add("cuda", True, "CUDA not available (CPU mode OK)", "No NVIDIA GPU or driver")
    except ImportError:
        report.add("import_torch", False, "torch not installed")


def _check_disk(report: PreflightReport, path: Path, min_gb: float = 2.0) -> None:
    label = str(path).replace(":", "").replace("\\", "_").replace("/", "_")[:40]
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        ok = free_gb >= min_gb
        report.add(
            f"disk_{label}",
            ok,
            f"Free space at {path}: {free_gb:.1f} GB",
            f"Need at least {min_gb} GB for models/cache" if not ok else "",
        )
    except OSError as e:
        report.add(f"disk_{label}", False, f"Cannot check disk at {path}", str(e))


def _check_writable(report: PreflightReport, path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        report.add(f"writable_{label}", True, f"Writable: {path}")
    except OSError as e:
        report.add(f"writable_{label}", False, f"Not writable: {path}", str(e))


def _check_api_deps(report: PreflightReport, cfg: UserConfig) -> None:
    if not cfg.services.api_enabled:
        return
    for mod in ("fastapi", "uvicorn"):
        _check_import(report, mod)
    try:
        import src.api  # noqa: F401
        report.add("import_src_api", True, "API application importable")
    except Exception as exc:
        report.add(
            "import_src_api",
            False,
            "API application import failed",
            str(exc),
        )


def _check_secrets(report: PreflightReport, cfg: UserConfig, *, require_openrouter: bool) -> None:
    status = secret_status(cfg.resolved_app_root())
    or_ok = status["openrouter"]["configured"]
    report.add(
        "openrouter_key",
        or_ok or not require_openrouter,
        "OPENROUTER_API_KEY set" if or_ok else "OPENROUTER_API_KEY missing (LLM disabled)",
        str(status["openrouter"]["source"]),
    )
    hf_ok = status["huggingface"]["configured"]
    report.add(
        "hf_token",
        True,
        "HF_TOKEN set (diarization)" if hf_ok else "HF_TOKEN optional (pyannote fallback)",
        str(status["huggingface"]["source"]),
    )


def run_preflight(
    cfg: UserConfig | None = None,
    *,
    require_torch: bool = True,
    require_openrouter: bool | None = None,
) -> PreflightReport:
    cfg = cfg or UserConfig()
    app_root = cfg.resolved_app_root()
    report = PreflightReport()

    _check_python(report)
    if require_torch:
        _check_import(report, "torch")
        _check_import(report, "transformers")
    _check_import(report, "typer")
    _check_import(report, "yaml")
    _check_ffmpeg(report, cfg)
    if require_torch:
        _check_torch_cuda(report)
    _check_disk(report, cfg.resolved_hf_home())
    _check_writable(report, cfg.resolved_hf_home(), "hf_cache")
    _check_writable(report, app_root / cfg.paths.outputs, "outputs")
    _check_writable(report, cfg.resolved_logs_dir(), "logs")

    need_or = require_openrouter if require_openrouter is not None else cfg.llm.enabled
    _check_secrets(report, cfg, require_openrouter=need_or)
    _check_api_deps(report, cfg)

    venv_py = app_root / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        report.add("venv", True, f"Bundled venv: {venv_py}")
    elif (app_root / ".venv").is_dir():
        report.add("venv", True, "Virtual environment present")
    else:
        report.add("venv", True, "Using current interpreter (dev mode)", sys.executable)

    return report
