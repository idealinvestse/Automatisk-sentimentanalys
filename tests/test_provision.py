"""Tests for launcher provisioning (venv, pip, ffmpeg)."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.install.config_schema import InstallProfile, UserConfig
from src.install.provision import (
    _extract_ffmpeg_binaries,
    ensure_ffmpeg,
    extras_for_profile,
    install_requirements,
    run_provision,
    venv_python_path,
)


def test_extras_for_profile_cli_includes_dashboard() -> None:
    extras = extras_for_profile(InstallProfile.cli)
    assert "dashboard-nicegui" in extras
    assert "api" in extras
    assert "install" in extras


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


def test_install_requirements_fails_without_pyproject(tmp_path: Path) -> None:
    with (
        patch("src.install.provision._run_pip"),
        pytest.raises(FileNotFoundError, match="pyproject.toml"),
    ):
        install_requirements(tmp_path, venv_python_path(tmp_path), InstallProfile.cli)


def test_install_requirements_uses_editable_extras(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    python = venv_python_path(tmp_path)

    with patch("src.install.provision._run_pip") as mock_pip:
        installed = install_requirements(tmp_path, python, InstallProfile.minimal)

    assert installed == ["min", "install"]
    assert mock_pip.call_args_list[-1][0][2] == ["install", "-e", ".[min,install]"]


def test_run_provision_reports_pip_failure(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "install_defaults.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    cfg = UserConfig(paths={"app_root": str(tmp_path)}, install_profile=InstallProfile.minimal)

    with patch("src.install.provision._run_pip", side_effect=OSError("pip failed")):
        report = run_provision(
            cfg,
            InstallProfile.minimal,
            ensure_virtualenv=False,
            download_ffmpeg=False,
            init_config=True,
        )

    assert not report.ok
    assert any(step.name == "pip" and not step.ok for step in report.steps)


@pytest.mark.skipif(sys.platform != "win32", reason="Bundled ffmpeg download is Windows-only")
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

    with (
        patch("src.install.provision._download_file", side_effect=fake_download),
        patch("src.install.provision.resolve_ffmpeg", return_value=None),
    ):
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


def test_venv_python_path_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # venv_python_path branches on sys.platform, not the host OS running the
    # test, so the Linux layout must be simulated explicitly to be verified
    # on any platform (e.g. Windows dev machines / CI runners).
    monkeypatch.setattr("src.install.provision.sys.platform", "linux")
    assert venv_python_path(tmp_path).name == "python"


def test_bundled_ffmpeg_path_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.install.provision import bundled_ffmpeg_path

    # bundled_ffmpeg_path branches on os.name, not the host OS running the
    # test; simulate POSIX explicitly so this assertion holds on Windows too.
    monkeypatch.setattr("src.install.provision.os.name", "posix")
    assert bundled_ffmpeg_path(tmp_path).as_posix().endswith("tools/ffmpeg/bin/ffmpeg")


def test_resolve_bootstrap_python_prefers_venv(tmp_path: Path) -> None:
    from src.install.provision import resolve_bootstrap_python

    venv_py = venv_python_path(tmp_path)
    venv_py.parent.mkdir(parents=True, exist_ok=True)
    venv_py.write_text("", encoding="utf-8")
    assert resolve_bootstrap_python(tmp_path) == venv_py


def test_resolve_bootstrap_python_env_override(tmp_path: Path, monkeypatch) -> None:
    from src.install.provision import resolve_bootstrap_python

    override = tmp_path / "custom-python"
    override.write_text("", encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_PYTHON", str(override))
    assert resolve_bootstrap_python(tmp_path) == override


def test_ensure_venv_returns_existing(tmp_path: Path) -> None:
    from src.install.provision import ensure_venv

    venv_py = venv_python_path(tmp_path)
    venv_py.parent.mkdir(parents=True, exist_ok=True)
    venv_py.write_text("", encoding="utf-8")
    assert ensure_venv(tmp_path) == venv_py


def test_ensure_ffmpeg_non_windows_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    # ensure_ffmpeg branches on sys.platform, not the host OS running the
    # test; simulate non-Windows explicitly so this assertion holds when the
    # suite is run on Windows too.
    monkeypatch.setattr("src.install.provision.sys.platform", "linux")
    cfg = UserConfig(paths={"app_root": str(tmp_path)})

    with (
        patch("src.install.provision.resolve_ffmpeg", return_value=None),
        pytest.raises(RuntimeError, match="ffmpeg not found"),
    ):
        ensure_ffmpeg(tmp_path, cfg)


def test_ensure_user_config_sets_app_root(tmp_path: Path, monkeypatch) -> None:
    from src.install.provision import ensure_user_config

    config_path = tmp_path / "user_config.yaml"
    monkeypatch.setenv("SENTIMENT_USER_CONFIG", str(config_path))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "install_defaults.yaml").write_text("version: 1\n", encoding="utf-8")
    cfg = ensure_user_config(tmp_path)
    assert cfg.paths.app_root == str(tmp_path.resolve())


def test_extras_for_profile_dev_includes_diarize() -> None:
    extras = extras_for_profile(InstallProfile.dev)
    assert "dev" in extras
    assert "diarize" in extras


def test_provision_report_ok_property() -> None:
    from src.install.provision import ProvisionReport

    report = ProvisionReport()
    report.add("step1", True, "ok")
    report.add("step2", False, "fail")
    assert report.ok is False
