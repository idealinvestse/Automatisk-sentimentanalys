"""Tests for launcher provisioning (venv, pip, ffmpeg)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.install.config_schema import InstallProfile, UserConfig
from src.install.provision import (
    _extract_ffmpeg_binaries,
    ensure_ffmpeg,
    requirements_for_profile,
    run_provision,
)


def test_requirements_for_profile_cli_includes_dashboard() -> None:
    files = requirements_for_profile(InstallProfile.cli)
    assert "requirements-dashboard-nicegui.txt" in files
    assert "requirements-api.txt" in files


def test_extract_ffmpeg_binaries_from_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "ffmpeg.zip"
    dest_bin = tmp_path / "bin"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("ffmpeg-master/bin/ffmpeg.exe", b"ffmpeg")
        archive.writestr("ffmpeg-master/bin/ffprobe.exe", b"ffprobe")

    ffmpeg_exe = _extract_ffmpeg_binaries(zip_path, dest_bin)
    assert ffmpeg_exe == dest_bin / "ffmpeg.exe"
    assert (dest_bin / "ffprobe.exe").is_file()


def test_ensure_ffmpeg_skips_when_already_available(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "existing" / "ffmpeg.exe"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"")
    monkeypatch.setenv("FFMPEG_PATH", str(fake))
    cfg = UserConfig(paths={"app_root": str(tmp_path)})

    with patch("src.install.provision._download_file") as mock_download:
        resolved = ensure_ffmpeg(tmp_path, cfg)

    assert resolved == str(fake.resolve())
    mock_download.assert_not_called()


def test_run_provision_reports_pip_failure(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "install_defaults.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "requirements-min.txt").write_text("not-a-real-package-name-xyz\n", encoding="utf-8")
    cfg = UserConfig(paths={"app_root": str(tmp_path)}, install_profile=InstallProfile.minimal)

    report = run_provision(
        cfg,
        InstallProfile.minimal,
        ensure_virtualenv=False,
        download_ffmpeg=False,
        init_config=True,
    )

    assert not report.ok
    assert any(step.name == "pip" and not step.ok for step in report.steps)


def test_run_provision_downloads_ffmpeg_when_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "install_defaults.yaml").write_text("version: 1\n", encoding="utf-8")
    cfg = UserConfig(paths={"app_root": str(tmp_path)})

    zip_path = tmp_path / "ffmpeg.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("ffmpeg-master/bin/ffmpeg.exe", b"ffmpeg")
        archive.writestr("ffmpeg-master/bin/ffprobe.exe", b"ffprobe")

    def fake_download(url: str, dest: Path, *, timeout: float = 300.0) -> None:
        dest.write_bytes(zip_path.read_bytes())

    with patch("src.install.provision._download_file", side_effect=fake_download):
        report = run_provision(
            cfg,
            InstallProfile.cli,
            ensure_virtualenv=False,
            install_packages=False,
            init_config=True,
        )

    ffmpeg_step = next(step for step in report.steps if step.name == "ffmpeg")
    assert ffmpeg_step.ok
    assert (tmp_path / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe").is_file()