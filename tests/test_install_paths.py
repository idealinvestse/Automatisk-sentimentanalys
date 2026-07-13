"""Tests for portable vs roaming config path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from src.install.config_schema import UserConfig
from src.install.paths_util import (
    _ffmpeg_exe_name,
    find_app_root_near,
    heal_app_root,
    looks_like_app_root,
    migrate_legacy_dashboard_settings,
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


def _make_project_tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "launcher").mkdir()
    (root / "src").mkdir()
    return root


def test_looks_like_app_root(tmp_path: Path) -> None:
    project = _make_project_tree(tmp_path / "app")
    assert looks_like_app_root(project)
    assert not looks_like_app_root(tmp_path)


def test_find_app_root_near_nested_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = _make_project_tree(workspace / "Automatisk-sentimentanalys")
    assert find_app_root_near(workspace) == project.resolve()


def test_heal_app_root_prefers_preferred(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = _make_project_tree(workspace / "Automatisk-sentimentanalys")
    healed = heal_app_root(workspace, preferred=project)
    assert healed == project.resolve()


def test_load_user_config_heals_wrong_app_root_and_legacy_dashboard(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = _make_project_tree(workspace / "Automatisk-sentimentanalys")
    (project / "configs").mkdir()
    (project / "configs" / "install_defaults.yaml").write_text(
        "version: 1\nservices:\n  dashboard_port: 3000\n  dashboard_ui: webui\n",
        encoding="utf-8",
    )

    user_cfg = tmp_path / "user_config.yaml"
    user_cfg.write_text(
        "\n".join(
            [
                "version: 1",
                "paths:",
                f"  app_root: {workspace.as_posix()}",
                "services:",
                "  dashboard_port: 8501",
                "  dashboard_ui: nicegui",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SENTIMENT_USER_CONFIG", str(user_cfg))

    cfg = load_user_config(project)
    assert cfg.resolved_app_root() == project.resolve()
    assert cfg.services.dashboard_port == 3000
    assert cfg.services.dashboard_ui == "webui"
    saved = yaml.safe_load(user_cfg.read_text(encoding="utf-8"))
    assert Path(saved["paths"]["app_root"]).resolve() == project.resolve()
    assert saved["services"]["dashboard_port"] == 3000


def test_migrate_legacy_dashboard_settings() -> None:
    cfg = UserConfig(services={"dashboard_port": 8080, "dashboard_ui": "webui"})
    assert migrate_legacy_dashboard_settings(cfg) is True
    assert cfg.services.dashboard_port == 3000
