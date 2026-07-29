"""Tests for launcher settings_service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from launcher.settings_service import (
    build_draft,
    export_bundle,
    import_bundle,
    restart_hints,
    save_draft,
    save_secret_permanent,
    validate_draft,
)
from src.install.config_schema import InstallProfile, UserConfig


@pytest.fixture
def config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SENTIMENT_USER_CONFIG", str(tmp_path / "user_config.yaml"))
    defaults = tmp_path / "configs" / "install_defaults.yaml"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(yaml.safe_dump({"sentiment_profile": "callcenter"}), encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_INSTALL_DEFAULTS", str(defaults))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_build_draft_is_deep_copy() -> None:
    cfg = UserConfig(sentiment_profile="forum")
    draft = build_draft(cfg)
    draft.sentiment_profile = "news"
    assert cfg.sentiment_profile == "forum"


def test_validate_rejects_same_ports() -> None:
    cfg = UserConfig()
    cfg.services.api_port = 8000
    cfg.services.dashboard_port = 8000
    issues = validate_draft(cfg, check_ports=False)
    assert any(i.field == "services" for i in issues)


def test_save_roundtrip(config_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.install.user_config import load_user_config

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = UserConfig(
        sentiment_profile="news",
        paths={"app_root": str(config_env)},
        services={"api_port": 9001, "dashboard_port": 9002},
    )
    result = save_draft(cfg, check_ports=False)
    assert result.path.is_file()
    loaded = load_user_config(config_env)
    assert loaded.sentiment_profile == "news"
    assert loaded.services.api_port == 9001


def test_export_import_bundle(config_env: Path) -> None:
    cfg = UserConfig(sentiment_profile="magazine", paths={"app_root": str(config_env)})
    path = config_env / "export.json"
    export_bundle(cfg, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["sentiment_profile"] == "magazine"
    loaded = import_bundle(path, config_env)
    assert loaded.sentiment_profile == "magazine"


def test_restart_hints_on_port_change() -> None:
    before = UserConfig()
    after = before.model_copy(deep=True)
    after.services.api_port = 9000
    hints = restart_hints(before, after)
    assert "api" in hints
    assert "dashboard" in hints


def test_config_to_env_sets_next_public_api_base_url() -> None:
    from src.install.user_config import config_to_env

    cfg = UserConfig(
        services={"api_host": "127.0.0.1", "api_port": 9001},
        runtime={"dashboard": {"api_base_url": "http://127.0.0.1:9001"}},
    )
    env = config_to_env(cfg)
    assert env["SENTIMENT_API_BASE_URL"] == "http://127.0.0.1:9001"
    assert env["NEXT_PUBLIC_API_BASE_URL"] == "http://127.0.0.1:9001"


def test_config_to_env_derives_api_base_url_when_empty() -> None:
    from src.install.user_config import config_to_env

    cfg = UserConfig(
        services={"api_host": "0.0.0.0", "api_port": 8123},
        runtime={"dashboard": {"api_base_url": ""}},
    )
    env = config_to_env(cfg)
    assert env["NEXT_PUBLIC_API_BASE_URL"] == "http://127.0.0.1:8123"
    assert env["SENTIMENT_API_BASE_URL"] == "http://127.0.0.1:8123"


def test_save_draft_syncs_api_base_url_when_port_changes(
    config_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.install.user_config import load_user_config

    baseline = UserConfig(
        paths={"app_root": str(config_env)},
        services={"api_host": "127.0.0.1", "api_port": 8000},
        runtime={"dashboard": {"api_base_url": "http://127.0.0.1:8000"}},
    )
    draft = baseline.model_copy(deep=True)
    draft.services.api_port = 9000
    result = save_draft(draft, baseline=baseline, check_ports=False)
    assert result.path.is_file()
    loaded = load_user_config(config_env)
    assert loaded.runtime.dashboard.api_base_url == "http://127.0.0.1:9000"
    assert "dashboard" in result.restart_services


def test_save_draft_preserves_custom_api_base_url_on_port_change(
    config_env: Path,
) -> None:
    from src.install.user_config import load_user_config

    baseline = UserConfig(
        paths={"app_root": str(config_env)},
        services={"api_host": "127.0.0.1", "api_port": 8000},
        runtime={"dashboard": {"api_base_url": "https://api.example.com"}},
    )
    draft = baseline.model_copy(deep=True)
    draft.services.api_port = 9000
    save_draft(draft, baseline=baseline, check_ports=False)
    loaded = load_user_config(config_env)
    assert loaded.runtime.dashboard.api_base_url == "https://api.example.com"


def test_config_to_env_mirrors_api_key_to_next_public() -> None:
    from src.install.user_config import config_to_env

    cfg = UserConfig(runtime={"api": {"api_key": "pilot-secret"}})
    env = config_to_env(cfg)
    assert env["SENTIMENT_API_KEY"] == "pilot-secret"
    assert env["NEXT_PUBLIC_API_KEY"] == "pilot-secret"


def test_config_to_env_omits_next_public_api_key_when_unset() -> None:
    from src.install.user_config import config_to_env

    cfg = UserConfig(runtime={"api": {"api_key": ""}})
    env = config_to_env(cfg)
    assert "SENTIMENT_API_KEY" not in env
    assert "NEXT_PUBLIC_API_KEY" not in env


def test_config_to_env_defaults_cors_for_local_dashboard() -> None:
    from src.install.user_config import config_to_env

    cfg = UserConfig(
        services={"dashboard_enabled": True, "dashboard_port": 3100},
        runtime={"api": {"cors_origins": ""}},
    )
    env = config_to_env(cfg)
    origins = {o.strip() for o in env["API_CORS_ORIGINS"].split(",") if o.strip()}
    assert origins == {"http://localhost:3100", "http://127.0.0.1:3100"}


def test_config_to_env_preserves_explicit_cors_origins() -> None:
    from src.install.user_config import config_to_env

    cfg = UserConfig(
        services={"dashboard_enabled": True, "dashboard_port": 3100},
        runtime={"api": {"cors_origins": "https://app.example.com"}},
    )
    env = config_to_env(cfg)
    assert env["API_CORS_ORIGINS"] == "https://app.example.com"


def test_save_draft_syncs_cors_when_dashboard_port_changes(
    config_env: Path,
) -> None:
    from src.install.user_config import load_user_config

    baseline = UserConfig(
        paths={"app_root": str(config_env)},
        services={"dashboard_port": 3000, "api_port": 8000},
        runtime={
            "api": {"cors_origins": "http://localhost:3000,http://127.0.0.1:3000"},
            "dashboard": {"api_base_url": "http://127.0.0.1:8000"},
        },
    )
    draft = baseline.model_copy(deep=True)
    draft.services.dashboard_port = 3200
    save_draft(draft, baseline=baseline, check_ports=False)
    loaded = load_user_config(config_env)
    origins = {o.strip() for o in loaded.runtime.api.cors_origins.split(",") if o.strip()}
    assert origins == {"http://localhost:3200", "http://127.0.0.1:3200"}


def test_restart_hints_on_profile_change() -> None:
    before = UserConfig(install_profile=InstallProfile.cli)
    after = before.model_copy(deep=True)
    after.install_profile = InstallProfile.full
    hints = restart_hints(before, after)
    assert "api" in hints and "dashboard" in hints


def test_save_secret_permanent_writes_user_file(
    config_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.install.secrets_win import get_secret, user_secret_file

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = user_secret_file("openrouter", config_env)
    msg = save_secret_permanent("openrouter", "sk-or-test-key", config_env)
    assert path.is_file()
    assert get_secret("openrouter", config_env) == "sk-or-test-key"
    assert "Sparad i:" in msg
    assert "openrouter.key" in msg


def test_config_to_env_includes_runtime(config_env: Path) -> None:
    from src.install.user_config import config_to_env, save_user_config

    cfg = UserConfig(
        paths={"app_root": str(config_env)},
        runtime={
            "api": {"api_key": "secret-key", "cors_origins": "http://localhost"},
            "alerting": {"webhook_enabled": True, "webhook_url": "https://example.com/hook"},
            "dashboard": {"dev_mode": True},
        },
    )
    save_user_config(cfg)
    env = config_to_env(cfg)
    assert env["SENTIMENT_API_KEY"] == "secret-key"
    assert env["API_CORS_ORIGINS"] == "http://localhost"
    assert env["ALERT_WEBHOOK_URL"] == "https://example.com/hook"
    assert env["SENTIMENT_DEV_MODE"] == "1"


def test_config_to_env_dashboard_storage_secret(config_env: Path) -> None:
    from src.install.user_config import config_to_env

    cfg = UserConfig(
        paths={"app_root": str(config_env)},
        runtime={"dashboard": {"storage_secret": "dash-secret"}},
    )
    env = config_to_env(cfg)
    assert env["DASHBOARD_STORAGE_SECRET"] == "dash-secret"
    assert "NICEGUI_STORAGE_SECRET" not in env


def test_build_child_env_defaults_api_media_root(
    config_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launcher ensures uploads work even when media_root is empty in config."""
    from launcher.env_builder import build_child_env

    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    cfg = UserConfig(paths={"app_root": str(config_env)}, runtime={"api": {"media_root": ""}})
    env = build_child_env(cfg)
    media = Path(env["API_MEDIA_ROOT"])
    assert media == config_env / "media"
    assert media.is_dir()
