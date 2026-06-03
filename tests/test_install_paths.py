"""Tests for portable vs roaming config path resolution."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.install.paths_util import portable_user_config_path, resolve_user_config_path
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

    from src.install.config_schema import UserConfig

    cfg = UserConfig(portable_mode=True, paths={"app_root": str(tmp_path)})
    saved = save_user_config(cfg)
    assert saved == cfg_path
    assert yaml.safe_load(saved.read_text(encoding="utf-8"))["portable_mode"] is True
