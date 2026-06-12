"""Tests for launcher status snapshot."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from launcher.status_snapshot import (
    ServiceState,
    check_api_health,
    collect_snapshot,
    service_status_text,
)
from src.install.config_schema import UserConfig


@pytest.fixture
def cfg(tmp_path: Path) -> UserConfig:
    return UserConfig(
        paths={"app_root": str(tmp_path)},
        portable_mode=True,
        services={"api_host": "127.0.0.1", "api_port": 8765, "dashboard_port": 9501},
    )


def test_check_api_health_resolves_bind_all_host() -> None:
    seen_url: list[str] = []

    class FakeResp:
        status = 200

        def read(self) -> bytes:
            return b'{"status":"ok"}'

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def fake_urlopen(url: str, timeout: float = 0.5) -> FakeResp:
        seen_url.append(url)
        return FakeResp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert check_api_health("0.0.0.0", 8765) is True
    assert seen_url == ["http://127.0.0.1:8765/health"]


def test_check_api_health_ok() -> None:
    class FakeResp:
        status = 200

        def read(self) -> bytes:
            return b'{"status":"ok"}'

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        assert check_api_health("127.0.0.1", 8765) is True


def test_collect_snapshot_stopped(cfg: UserConfig) -> None:
    with (
        patch("launcher.status_snapshot.is_port_open", return_value=False),
        patch("launcher.status_snapshot.get_pid_info", return_value=None),
        patch("launcher.status_snapshot.secret_status", return_value={}),
    ):
        snap = collect_snapshot(cfg, launcher_root=cfg.resolved_app_root())
    assert snap.api.state == ServiceState.STOPPED
    assert snap.dashboard.state == ServiceState.STOPPED
    assert snap.system.app_root == cfg.resolved_app_root()


def test_collect_snapshot_api_running(cfg: UserConfig) -> None:
    from launcher.pid_store import PidRecord

    record = PidRecord(name="api", pid=12345, command=[], pid_file=Path("x.json"))
    with (
        patch("launcher.status_snapshot.is_port_open", return_value=True),
        patch("launcher.status_snapshot.is_process_running", return_value=True),
        patch("launcher.status_snapshot.get_pid_info", return_value=record),
        patch("launcher.status_snapshot.check_api_health", return_value=True),
        patch("launcher.status_snapshot.secret_status", return_value={}),
    ):
        snap = collect_snapshot(cfg, launcher_root=cfg.resolved_app_root())
    assert snap.api.state == ServiceState.RUNNING
    assert snap.api.health_ok is True


def test_service_status_text(cfg: UserConfig) -> None:
    with patch("launcher.status_snapshot.collect_snapshot") as mock_collect:
        from launcher.status_snapshot import (
            LauncherSnapshot,
            ServiceSnapshot,
            SystemSnapshot,
        )

        api = ServiceSnapshot(
            name="api",
            state=ServiceState.RUNNING,
            host="127.0.0.1",
            port=8000,
            url="http://127.0.0.1:8000",
            pid=1,
            process_alive=True,
            port_open=True,
            health_ok=True,
            pid_file=None,
            stdout_log=Path("a.log"),
            stderr_log=Path("a.err"),
            last_error_tail="",
        )
        dash = ServiceSnapshot(
            name="dashboard",
            state=ServiceState.STOPPED,
            host="127.0.0.1",
            port=8501,
            url="http://localhost:8501",
            pid=None,
            process_alive=False,
            port_open=False,
            health_ok=None,
            pid_file=None,
            stdout_log=Path("d.log"),
            stderr_log=Path("d.err"),
            last_error_tail="",
        )
        sys_snap = SystemSnapshot(
            launcher_root=cfg.resolved_app_root(),
            app_root=cfg.resolved_app_root(),
            config_path=Path("c.yaml"),
            user_data_dir=cfg.resolved_user_data_dir(),
            python_exe=Path("python"),
            venv_ok=False,
            install_profile="cli",
            sentiment_profile="default",
            device="cpu",
            llm_enabled=False,
            openrouter_configured=False,
            huggingface_configured=False,
            api_version="0.4.0",
        )
        mock_collect.return_value = LauncherSnapshot(
            api=api,
            dashboard=dash,
            system=sys_snap,
            collected_at="now",
        )
        text = service_status_text(cfg, "api")
        assert "Kör" in text
        assert "pid 1" in text
