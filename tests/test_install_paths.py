"""Tests for portable vs roaming config path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from src.install.config_schema import UserConfig
from src.install.paths_util import (
    _ffmpeg_exe_name,
    portable_user_config_path,
    resolve_ffmpeg,
    resolve_user_config_path,
)
from src.install.user_config import load_user_config, save_user_config


def test_resolve_portable_when_user_data_exists(tmp_path: Path) -> None:
    portable = portable_user_config_path(tmp_path)
    portable.parent.mkdir(parents=True)
    portable.write_text("portable_mode: true\nsentiment_profile: forum\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "install_defaults.yaml").write_text(
        "portable_mode: false\nsentiment_profile: default\n", encoding="utf-8"
    )

    path = resolve_user_config_path(tmp_path)
    assert path == portable

    cfg = load_user_config(tmp_path)
    assert cfg.portable_mode is True
    assert cfg.sentiment_profile == "forum"


def test_save_uses_portable_path_when_configured(tmp_path: Path) -> None:
    cfg_path = portable_user_config_path(tmp_path)
    cfg_path.parent.mkdir(parents=True)
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "install_defaults.yaml").write_text("version: 1\n", encoding="utf-8")

    cfg = UserConfig(portable_mode=True, paths={"app_root": str(tmp_path)})
    saved = save_user_config(cfg)
    assert saved == cfg_path
    assert yaml.safe_load(saved.read_text(encoding="utf-8"))["portable_mode"] is True


def test_resolve_ffmpeg_env_override(tmp_path: Path, monkeypatch) -> None:
    fake_ffmpeg = tmp_path / "custom" / "ffmpeg.exe"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_bytes(b"")
    monkeypatch.setenv("FFMPEG_PATH", str(fake_ffmpeg))
    cfg = UserConfig(paths={"app_root": str(tmp_path)})
    assert resolve_ffmpeg(cfg) == str(fake_ffmpeg.resolve())


def test_resolve_ffmpeg_bundled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    bundled = tmp_path / "tools" / "ffmpeg" / "bin" / _ffmpeg_exe_name()
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"")
    cfg = UserConfig(paths={"app_root": str(tmp_path)})
    with patch("src.install.paths_util.shutil.which", return_value=None):
        assert resolve_ffmpeg(cfg) == str(bundled)
