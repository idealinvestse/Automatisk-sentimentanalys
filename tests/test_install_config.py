"""Tests for install configuration layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.install.config_schema import InstallProfile, UserConfig
from src.install.user_config import (
    config_to_env,
    load_user_config,
    merge_configs,
    save_user_config,
)


def test_user_config_defaults() -> None:
    cfg = UserConfig()
    assert cfg.sentiment_profile == "callcenter"
    assert cfg.services.api_port == 8000


def test_merge_configs_deep() -> None:
    merged = merge_configs(
        {"services": {"api_port": 8000}, "llm": {"enabled": False}},
        {"services": {"api_port": 9000}},
    )
    assert merged.services.api_port == 9000
    assert merged.llm.enabled is False


def test_save_and_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_USER_CONFIG", str(tmp_path / "user_config.yaml"))
    defaults = tmp_path / "configs" / "install_defaults.yaml"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(yaml.safe_dump({"sentiment_profile": "forum"}), encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_INSTALL_DEFAULTS", str(defaults))
    monkeypatch.chdir(tmp_path)

    cfg = UserConfig(sentiment_profile="news", paths={"app_root": str(tmp_path)})
    save_user_config(cfg)
    loaded = load_user_config(tmp_path)
    assert loaded.sentiment_profile == "news"


def test_config_to_env() -> None:
    cfg = UserConfig(paths={"app_root": "C:/app", "hf_cache": "cache/hf"})
    env = config_to_env(cfg)
    assert "HF_HOME" in env
    assert env["HF_HOME"].endswith("cache\\hf") or env["HF_HOME"].endswith("cache/hf")


def test_resolve_data_path_relative_under_app_root(tmp_path: Path) -> None:
    cfg = UserConfig(paths={"app_root": str(tmp_path), "hf_cache": "cache/hf"})
    assert cfg.resolved_hf_home() == (tmp_path / "cache" / "hf").resolve()


def test_resolve_data_path_under_data_root(tmp_path: Path) -> None:
    data = tmp_path / "E_drive" / "SentimentData"
    data.mkdir(parents=True)
    cfg = UserConfig(
        paths={
            "app_root": str(tmp_path / "app"),
            "data_root": str(data),
            "hf_cache": "cache/hf",
            "llm_cache": "cache/llm",
            "outputs": "outputs",
        }
    )
    assert cfg.resolved_data_root() == data.resolve()
    assert cfg.resolved_hf_home() == (data / "cache" / "hf").resolve()
    assert cfg.resolved_llm_cache() == (data / "cache" / "llm").resolve()
    assert cfg.resolved_outputs() == (data / "outputs").resolve()


def test_resolve_data_path_absolute_overrides_data_root(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    abs_hf = tmp_path / "other" / "models"
    abs_hf.mkdir(parents=True)
    cfg = UserConfig(
        paths={
            "app_root": str(tmp_path),
            "data_root": str(data),
            "hf_cache": str(abs_hf),
        }
    )
    assert cfg.resolved_hf_home() == abs_hf.resolve()


def test_config_to_env_includes_data_root(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    cfg = UserConfig(paths={"app_root": str(tmp_path), "data_root": str(data)})
    env = config_to_env(cfg)
    assert env["SENTIMENT_DATA_ROOT"] == str(data.resolve())
    assert env["HF_HOME"] == str((data / "cache" / "hf").resolve())
    assert "SENTIMENT_LLM_CACHE" in env


def test_install_profile_enum() -> None:
    assert InstallProfile.full.value == "full"
