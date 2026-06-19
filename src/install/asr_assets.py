"""Install ASR Python packages and pre-download transcription model weights."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..transcription.base import resolve_model_name, resolve_model_name_for_backend

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None] | None

ASR_PIP_PACKAGES = (
    "faster-whisper>=1.0.0",
    "whisperx>=3.1.1",
    "huggingface-hub>=0.23.0",
)

DEFAULT_PREFETCH_BACKENDS = ("faster", "whisperx", "transformers")


@dataclass
class AsrAssetStep:
    name: str
    ok: bool
    message: str
    detail: str = ""


@dataclass
class AsrAssetReport:
    steps: list[AsrAssetStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)

    def add(self, name: str, ok: bool, message: str, detail: str = "") -> None:
        self.steps.append(AsrAssetStep(name=name, ok=ok, message=message, detail=detail))


@dataclass(frozen=True)
class AsrStatus:
    """Snapshot of installed ASR packages and local model cache."""

    faster_whisper_installed: bool
    whisperx_installed: bool
    huggingface_hub_installed: bool
    model_name: str
    hf_cache_dir: str
    kb_model_cached: bool

    @property
    def packages_ready(self) -> bool:
        return self.faster_whisper_installed and self.whisperx_installed

    @property
    def ready(self) -> bool:
        return self.packages_ready and self.kb_model_cached

    def summary(self) -> str:
        fw = "faster-whisper ✓" if self.faster_whisper_installed else "faster-whisper ✗"
        wx = "whisperx ✓" if self.whisperx_installed else "whisperx ✗"
        model = "modell ✓" if self.kb_model_cached else "modell ej cachelagrad"
        return f"{fw} | {wx} | {model}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "faster_whisper_installed": self.faster_whisper_installed,
            "whisperx_installed": self.whisperx_installed,
            "huggingface_hub_installed": self.huggingface_hub_installed,
            "model_name": self.model_name,
            "hf_cache_dir": self.hf_cache_dir,
            "kb_model_cached": self.kb_model_cached,
            "packages_ready": self.packages_ready,
            "ready": self.ready,
            "summary": self.summary(),
        }


def hf_repo_cached(repo_id: str, hf_home: Path) -> bool:
    """Return True if a Hugging Face repo has at least one cached snapshot."""
    slug = "models--" + repo_id.replace("/", "--")
    snapshots = hf_home / "hub" / slug / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        return any(snapshots.iterdir())
    except OSError:
        return False


def collect_asr_status(
    *,
    model: str = "kb-whisper-large",
    hf_home: Path | None = None,
) -> AsrStatus:
    """Inspect ASR package installation and default model cache."""
    repo_id = resolve_model_name(model)
    cache_root = hf_home or Path(os.environ.get("HF_HOME", "cache/hf"))
    return AsrStatus(
        faster_whisper_installed=is_module_installed("faster_whisper"),
        whisperx_installed=is_module_installed("whisperx"),
        huggingface_hub_installed=is_module_installed("huggingface_hub"),
        model_name=repo_id,
        hf_cache_dir=str(cache_root.resolve()),
        kb_model_cached=hf_repo_cached(repo_id, cache_root),
    )


def _log(progress: ProgressCallback, message: str) -> None:
    logger.info(message)
    if progress:
        progress(message)


def is_module_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def configure_hf_cache(hf_home: Path | None) -> None:
    """Point Hugging Face caches at the project/user cache directory."""
    if not hf_home:
        return
    hf_home.mkdir(parents=True, exist_ok=True)
    path_str = str(hf_home.resolve())
    os.environ.setdefault("HF_HOME", path_str)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))


def install_asr_packages(
    root: Path,
    python: Path | None = None,
    *,
    progress: ProgressCallback = None,
) -> AsrAssetReport:
    """Ensure faster-whisper, whisperx and huggingface-hub are installed via pip."""
    report = AsrAssetReport()
    interpreter = python or Path(sys.executable)

    missing = [pkg for mod, pkg in (
        ("faster_whisper", "faster-whisper>=1.0.0"),
        ("whisperx", "whisperx>=3.1.1"),
        ("huggingface_hub", "huggingface-hub>=0.23.0"),
    ) if not is_module_installed(mod)]

    if not missing:
        report.add("asr_packages", True, "ASR-paket redan installerade")
        return report

    to_install = list(ASR_PIP_PACKAGES)
    _log(progress, f"Installerar ASR-paket: {', '.join(to_install)}")
    try:
        subprocess.run(
            [str(interpreter), "-m", "pip", "install", *to_install],
            check=True,
            cwd=str(root),
        )
        report.add("asr_packages", True, "ASR-paket installerade", ", ".join(to_install))
    except subprocess.CalledProcessError as exc:
        report.add("asr_packages", False, "pip install ASR misslyckades", str(exc))
    return report


def _download_faster_whisper(
    *,
    model_name: str,
    device: str,
    revision: str | None,
    progress: ProgressCallback,
) -> None:
    from faster_whisper import WhisperModel

    from ..core.device import normalize_device_for_asr

    load_name = resolve_model_name_for_backend(model_name, "faster")
    dev_kind, cuda_idx = normalize_device_for_asr(device)
    compute_type = (
        "float16" if dev_kind == "cuda" else ("int8" if dev_kind == "cpu" else "float32")
    )
    model_kwargs: dict[str, Any] = {
        "device": dev_kind,
        "compute_type": compute_type,
    }
    if cuda_idx is not None:
        model_kwargs["device_index"] = cuda_idx
    if revision:
        model_kwargs["revision"] = revision

    _log(progress, f"Laddar ner faster-whisper-modell: {load_name}")
    model = WhisperModel(load_name, **model_kwargs)
    del model


def _download_transformers(
    *,
    model_name: str,
    revision: str | None,
    progress: ProgressCallback,
) -> None:
    from huggingface_hub import snapshot_download

    repo_id = resolve_model_name(model_name)
    _log(progress, f"Laddar ner Hugging Face-modell: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_files_only=False,
    )


def _download_whisperx(
    *,
    model_name: str,
    device: str,
    language: str,
    progress: ProgressCallback,
) -> None:
    import whisperx

    from ..core.device import normalize_device_for_asr

    load_name = resolve_model_name_for_backend(model_name, "whisperx")
    lower = load_name.lower()
    if "kb-whisper" in lower or "kblab" in lower:
        load_name = "large-v3"

    dev_kind, cuda_idx = normalize_device_for_asr(device)
    if dev_kind == "cuda":
        device_str = f"cuda:{cuda_idx or 0}"
    elif dev_kind == "mps":
        device_str = "mps"
    else:
        device_str = "cpu"
    compute_type = "float16" if dev_kind == "cuda" else ("int8" if dev_kind == "cpu" else "float32")

    _log(progress, f"Laddar ner WhisperX ASR-modell: {load_name}")
    model = whisperx.load_model(load_name, device_str, compute_type=compute_type)
    del model

    _log(progress, f"Laddar ner WhisperX align-modell för språk: {language}")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device_str)
    del align_model, metadata


def download_asr_models(
    *,
    backends: list[str] | None = None,
    model: str = "kb-whisper-large",
    device: str = "cpu",
    language: str = "sv",
    revision: str | None = "strict",
    hf_home: Path | None = None,
    progress: ProgressCallback = None,
) -> AsrAssetReport:
    """Pre-download model weights for the selected ASR backends."""
    report = AsrAssetReport()
    configure_hf_cache(hf_home)

    selected = [b.strip().lower() for b in (backends or list(DEFAULT_PREFETCH_BACKENDS)) if b.strip()]
    if not selected:
        report.add("models", False, "Inga backends angivna")
        return report

    for backend in selected:
        step_name = f"model_{backend}"
        try:
            if backend == "faster":
                if not is_module_installed("faster_whisper"):
                    raise ImportError("faster-whisper ej installerat")
                _download_faster_whisper(
                    model_name=model,
                    device=device,
                    revision=revision,
                    progress=progress,
                )
            elif backend == "transformers":
                if not is_module_installed("transformers"):
                    raise ImportError("transformers ej installerat")
                _download_transformers(
                    model_name=model,
                    revision=revision,
                    progress=progress,
                )
            elif backend == "whisperx":
                if not is_module_installed("whisperx"):
                    raise ImportError("whisperx ej installerat")
                _download_whisperx(
                    model_name=model,
                    device=device,
                    language=language,
                    progress=progress,
                )
            else:
                raise ValueError(f"Okänd backend: {backend}")
            report.add(step_name, True, f"Modell hämtad ({backend})")
        except Exception as exc:
            logger.warning("ASR model download failed for %s: %s", backend, exc)
            report.add(step_name, False, f"Modellnedladdning misslyckades ({backend})", str(exc))

    return report


def ensure_asr_assets(
    root: Path,
    *,
    python: Path | None = None,
    backends: list[str] | None = None,
    model: str = "kb-whisper-large",
    device: str = "cpu",
    language: str = "sv",
    revision: str | None = "strict",
    hf_home: Path | None = None,
    install_packages: bool = True,
    download_models: bool = True,
    progress: ProgressCallback = None,
) -> AsrAssetReport:
    """Install ASR packages and optionally pre-download models."""
    combined = AsrAssetReport()
    interpreter = python or Path(sys.executable)

    if install_packages:
        pkg_report = install_asr_packages(root, interpreter, progress=progress)
        combined.steps.extend(pkg_report.steps)
        if not pkg_report.ok:
            return combined

    if download_models:
        model_report = download_asr_models(
            backends=backends,
            model=model,
            device=device,
            language=language,
            revision=revision,
            hf_home=hf_home,
            progress=progress,
        )
        combined.steps.extend(model_report.steps)

    return combined


def download_asr_cli(
    backend: list[str] | None = None,
    model: str = "kb-whisper-large",
    device: str = "cpu",
    language: str = "sv",
    revision: str | None = "strict",
    skip_packages: bool = False,
    skip_models: bool = False,
) -> None:
    """Console entry point: sentimentanalys-download-asr."""
    import typer
    from rich.console import Console

    from .config_schema import UserConfig

    console = Console()
    cfg = UserConfig()
    report = ensure_asr_assets(
        cfg.resolved_app_root(),
        backends=backend or list(DEFAULT_PREFETCH_BACKENDS),
        model=model,
        device=device,
        language=language,
        revision=revision,
        hf_home=cfg.resolved_hf_home(),
        install_packages=not skip_packages,
        download_models=not skip_models,
        progress=lambda msg: console.print(f"[dim]… {msg}[/dim]"),
    )
    for step in report.steps:
        color = "green" if step.ok else "red"
        detail = f" ({step.detail})" if step.detail else ""
        console.print(f"[{color}]{step.name}[/]: {step.message}{detail}")
    raise typer.Exit(code=0 if report.ok else 1)


def main() -> None:
    import typer
    from typing import Annotated

    def _entry(
        backend: Annotated[
            list[str],
            typer.Option(help="faster | whisperx | transformers"),
        ] = list(DEFAULT_PREFETCH_BACKENDS),
        model: Annotated[str, typer.Option("--model", "-m")] = "kb-whisper-large",
        device: Annotated[str, typer.Option("--device")] = "cpu",
        language: Annotated[str, typer.Option("--language", "-l")] = "sv",
        revision: Annotated[str | None, typer.Option("--revision")] = "strict",
        skip_packages: Annotated[bool, typer.Option("--skip-packages")] = False,
        skip_models: Annotated[bool, typer.Option("--skip-models")] = False,
    ) -> None:
        download_asr_cli(
            backend=backend,
            model=model,
            device=device,
            language=language,
            revision=revision,
            skip_packages=skip_packages,
            skip_models=skip_models,
        )

    typer.run(_entry)


if __name__ == "__main__":
    main()