"""Tests for ASR package install and model prefetch helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.install.asr_assets import (
    AsrAssetReport,
    collect_asr_status,
    configure_hf_cache,
    download_asr_models,
    ensure_asr_assets,
    hf_repo_cached,
    install_asr_packages,
)


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


@patch("src.install.asr_assets._download_faster_whisper")
@patch("src.install.asr_assets.is_module_installed", return_value=True)
def test_download_asr_models_faster(_mock_installed, mock_fw, tmp_path):
    mock_fw.return_value = None
    report = download_asr_models(backends=["faster"], hf_home=tmp_path)
    assert report.ok
    mock_fw.assert_called_once()


@patch("src.install.asr_assets._download_whisperx")
@patch("src.install.asr_assets.is_module_installed", return_value=True)
def test_download_asr_models_whisperx(_mock_installed, mock_wx, tmp_path):
    mock_wx.return_value = None
    report = download_asr_models(backends=["whisperx"], language="sv", hf_home=tmp_path)
    assert report.ok
    mock_wx.assert_called_once()


@patch("src.install.asr_assets.is_module_installed", return_value=False)
def test_download_asr_models_reports_missing_package(_mock_installed, tmp_path):
    report = download_asr_models(backends=["whisperx"], hf_home=tmp_path)
    assert not report.ok
    assert "whisperx" in report.steps[0].detail


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
