"""Tests for ASR package install and model prefetch helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from src.install.asr_assets import (
    AsrAssetReport,
    collect_asr_status,
    configure_hf_cache,
    download_asr_models,
    ensure_asr_assets,
    ensure_torchaudio_audiometadata,
    hf_repo_cached,
    install_asr_packages,
    _format_download_error,
)


def test_ensure_torchaudio_audiometadata_noop_when_present() -> None:
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        pytest.skip("torchaudio lacks AudioMetaData in this env")
    ensure_torchaudio_audiometadata()
    assert hasattr(torchaudio, "AudioMetaData")


def test_ensure_torchaudio_audiometadata_restores_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    ensure_torchaudio_audiometadata()
    assert hasattr(fake, "AudioMetaData")


def test_format_download_error_hints_audiometadata() -> None:
    msg = _format_download_error(
        AttributeError("module 'torchaudio' has no attribute 'AudioMetaData'")
    )
    assert "torchaudio<2.9" in msg
    assert "cu128" in msg


def test_ensure_torch_load_weights_compat_sets_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from src.install.asr_assets import ensure_torch_load_weights_compat

    calls: list[dict] = []

    def fake_load(*args: object, **kwargs: object) -> str:
        calls.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(torch, "load", fake_load)
    ensure_torch_load_weights_compat()
    assert torch.load("x.pt") == "ok"
    assert calls[0].get("weights_only") is False
    # second call is idempotent; explicit True is overridden
    ensure_torch_load_weights_compat()
    torch.load("y.pt", weights_only=True)
    assert calls[1].get("weights_only") is False


def test_configure_hf_cache_sets_env(tmp_path):
    configure_hf_cache(tmp_path)
    import os

    assert os.environ["HF_HOME"] == str(tmp_path.resolve())


def test_install_asr_packages_skips_when_present(tmp_path):
    with patch("src.install.asr_assets.is_module_installed", return_value=True):
        report = install_asr_packages(tmp_path)
    assert report.ok
    assert report.steps[0].name == "asr_packages"


def test_install_asr_packages_runs_pip_when_missing(tmp_path):
    with (
        patch("src.install.asr_assets.is_module_installed", return_value=False),
        patch("src.install.asr_assets.subprocess.run") as mock_run,
    ):
        report = install_asr_packages(tmp_path)
    assert report.ok
    mock_run.assert_called_once()


def test_is_module_installed_checks_other_python(tmp_path: Path) -> None:
    from src.install.asr_assets import is_module_installed

    other = tmp_path / "python.exe"
    other.write_text("", encoding="utf-8")
    with patch("src.install.asr_assets.subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 0})()
        assert is_module_installed("faster_whisper", other) is True
    assert mock_run.call_args.args[0][0] == str(other)


def test_is_module_installed_treats_pythonw_as_same_venv(tmp_path: Path, monkeypatch) -> None:
    """Launcher starts via pythonw.exe; resolve_python returns python.exe — must not spawn."""
    import sys

    from src.install.asr_assets import is_module_installed

    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    py = scripts / "python.exe"
    pyw = scripts / "pythonw.exe"
    py.write_text("", encoding="utf-8")
    pyw.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(pyw))
    with patch("src.install.asr_assets.subprocess.run") as mock_run:
        with patch("src.install.asr_assets.importlib.util.find_spec", return_value=object()):
            assert is_module_installed("faster_whisper", py) is True
    mock_run.assert_not_called()


def test_is_module_installed_hides_console_on_windows(tmp_path: Path) -> None:
    import subprocess

    from src.install.asr_assets import is_module_installed

    other = tmp_path / "other" / "python.exe"
    other.parent.mkdir()
    other.write_text("", encoding="utf-8")
    with patch("src.install.asr_assets.subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 0})()
        assert is_module_installed("faster_whisper", other) is True
    kwargs = mock_run.call_args.kwargs
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        assert kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW
    else:
        assert kwargs.get("creationflags", 0) == 0


@patch("src.install.asr_assets._ensure_interpreter_site_packages")
@patch("src.install.asr_assets._download_faster_whisper")
@patch("src.install.asr_assets.is_module_installed", return_value=True)
def test_download_asr_models_faster(_mock_installed, mock_fw, _mock_site, tmp_path):
    mock_fw.return_value = None
    report = download_asr_models(backends=["faster"], hf_home=tmp_path)
    assert report.ok
    mock_fw.assert_called_once()


@patch("src.install.asr_assets._ensure_interpreter_site_packages")
@patch("src.install.asr_assets._download_whisperx")
@patch("src.install.asr_assets.is_module_installed", return_value=True)
def test_download_asr_models_whisperx(_mock_installed, mock_wx, _mock_site, tmp_path):
    mock_wx.return_value = None
    report = download_asr_models(backends=["whisperx"], language="sv", hf_home=tmp_path)
    assert report.ok
    mock_wx.assert_called_once()


@patch("src.install.asr_assets._ensure_interpreter_site_packages")
@patch("src.install.asr_assets.is_module_installed", return_value=False)
def test_download_asr_models_reports_missing_package(_mock_installed, _mock_site, tmp_path):
    report = download_asr_models(backends=["whisperx"], hf_home=tmp_path)
    assert not report.ok
    assert "whisperx" in report.steps[0].detail


def test_configure_hf_cache_disables_xet(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    configure_hf_cache(tmp_path)
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


@patch("src.install.asr_assets.download_asr_models")
@patch("src.install.asr_assets.install_asr_packages")
def test_ensure_asr_assets_passes_python(mock_install, mock_download, tmp_path):
    mock_install.return_value = AsrAssetReport()
    mock_install.return_value.add("asr_packages", True, "ok")
    mock_download.return_value = AsrAssetReport()
    mock_download.return_value.add("model_faster", True, "ok")
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")

    ensure_asr_assets(tmp_path, python=py, install_packages=True, download_models=True)
    assert mock_download.call_args.kwargs["python"] == py


def test_hf_repo_cached_detects_snapshot(tmp_path):
    repo = tmp_path / "hub" / "models--KBLab--kb-whisper-large" / "snapshots" / "abc"
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}", encoding="utf-8")
    assert hf_repo_cached("KBLab/kb-whisper-large", tmp_path)


def test_collect_asr_status_summary():
    status = collect_asr_status(model="kb-whisper-large", hf_home=Path("cache/hf"))
    assert "faster-whisper" in status.summary()
    assert status.model_name == "KBLab/kb-whisper-large"


@patch("src.install.asr_assets.download_asr_models")
@patch("src.install.asr_assets.install_asr_packages")
def test_ensure_asr_assets_combines_reports(mock_install, mock_download, tmp_path):
    mock_install.return_value = AsrAssetReport()
    mock_install.return_value.add("asr_packages", True, "ok")
    mock_download.return_value = AsrAssetReport()
    mock_download.return_value.add("model_faster", True, "ok")

    report = ensure_asr_assets(tmp_path, install_packages=True, download_models=True)
    assert report.ok
    assert len(report.steps) == 2
