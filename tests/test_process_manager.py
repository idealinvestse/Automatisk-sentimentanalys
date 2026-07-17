"""Tests for launcher.process_manager orchestration (mocked subprocess)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from launcher.process_manager import (
    restart_named_services,
    start_api,
    start_dashboard,
    stop_service,
    _wait_for_service,
)
from src.install.config_schema import UserConfig


@pytest.fixture
def cfg(tmp_path: Path) -> UserConfig:
    return UserConfig(
        paths={"app_root": str(tmp_path)},
        portable_mode=True,
        services={
            "api_host": "127.0.0.1",
            "api_port": 8765,
            "dashboard_port": 3456,
            "dashboard_ui": "webui",
        },
        runtime={"dashboard": {"dev_mode": True, "api_base_url": "http://127.0.0.1:8765"}},
    )


def _port_free_then_open() -> object:
    """Preflight sees free port; wait loop then sees it open."""
    calls = {"n": 0}

    def is_open(*_a: object, **_k: object) -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    return is_open


def test_start_api_spawns_uvicorn_and_tracks_pid(cfg: UserConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 4242

    def fake_popen(c, name, cmd, *, extra_env=None):
        captured["name"] = name
        captured["cmd"] = cmd
        captured["extra_env"] = extra_env
        return FakeProc()

    monkeypatch.setattr("launcher.process_manager.stop_service", lambda *a, **k: False)
    monkeypatch.setattr("launcher.process_manager.check_api_dependencies", lambda **kw: None)
    monkeypatch.setattr("launcher.process_manager.resolve_python", lambda c: Path("python.exe"))
    monkeypatch.setattr("launcher.process_manager._popen_service", fake_popen)
    monkeypatch.setattr("launcher.process_manager.is_port_open", _port_free_then_open())
    monkeypatch.setattr("launcher.process_manager.is_process_running", lambda pid: True)
    monkeypatch.setattr(
        "launcher.process_manager.port_owned_by_pid_tree",
        lambda *a, **k: True,
    )
    monkeypatch.setattr("launcher.status_snapshot.check_api_health", lambda *a, **k: True)

    info = start_api(cfg)
    assert info.pid == 4242
    assert info.name == "api"
    assert captured["name"] == "api"
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:4] == ["python.exe", "-m", "uvicorn", "src.api:app"]
    assert "--port" in cmd and "8765" in cmd
    assert info.pid_file.is_file()


def test_start_dashboard_passes_port_env(cfg: UserConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 5151

    def fake_popen(c, name, cmd, *, extra_env=None):
        captured["cmd"] = cmd
        captured["extra_env"] = extra_env
        return FakeProc()

    monkeypatch.setattr("launcher.process_manager.stop_service", lambda *a, **k: False)
    monkeypatch.setattr("launcher.process_manager.check_dashboard_dependencies", lambda **kw: None)
    monkeypatch.setattr("launcher.process_manager.resolve_python", lambda c: Path("python.exe"))
    monkeypatch.setattr("launcher.process_manager._popen_service", fake_popen)
    monkeypatch.setattr("launcher.process_manager.is_port_open", _port_free_then_open())
    monkeypatch.setattr("launcher.process_manager.is_process_running", lambda pid: True)
    monkeypatch.setattr(
        "launcher.process_manager.port_owned_by_pid_tree",
        lambda *a, **k: True,
    )

    info = start_dashboard(cfg)
    assert info.pid == 5151
    assert captured["cmd"] == ["python.exe", "-m", "launcher.dashboard_spawn"]
    env = captured["extra_env"]
    assert isinstance(env, dict)
    assert env["PORT"] == "3456"
    assert env["WEBUI_PORT"] == "3456"
    assert env["DASHBOARD_UI"] == "webui"


def test_start_dashboard_fails_when_port_already_occupied(
    cfg: UserConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("launcher.process_manager.stop_service", lambda *a, **k: False)
    monkeypatch.setattr("launcher.process_manager.check_dashboard_dependencies", lambda **kw: None)
    monkeypatch.setattr("launcher.process_manager.resolve_python", lambda c: Path("python.exe"))
    monkeypatch.setattr("launcher.process_manager.is_port_open", lambda *a, **k: True)
    monkeypatch.setattr(
        "launcher.process_manager.describe_port_occupant",
        lambda *a, **k: "LM Studio.exe (pid 19468)",
    )
    popen = MagicMock()
    monkeypatch.setattr("launcher.process_manager._popen_service", popen)

    with pytest.raises(RuntimeError, match=r"3456.*already in use|redan upptagen"):
        start_dashboard(cfg)
    popen.assert_not_called()


def test_start_api_fails_when_port_already_occupied(
    cfg: UserConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("launcher.process_manager.stop_service", lambda *a, **k: False)
    monkeypatch.setattr("launcher.process_manager.check_api_dependencies", lambda **kw: None)
    monkeypatch.setattr("launcher.process_manager.resolve_python", lambda c: Path("python.exe"))
    monkeypatch.setattr("launcher.process_manager.is_port_open", lambda *a, **k: True)
    monkeypatch.setattr(
        "launcher.process_manager.describe_port_occupant",
        lambda *a, **k: "python.exe (pid 10876)",
    )
    popen = MagicMock()
    monkeypatch.setattr("launcher.process_manager._popen_service", popen)

    with pytest.raises(RuntimeError, match=r"8765.*already in use|redan upptagen"):
        start_api(cfg)
    popen.assert_not_called()


def test_wait_for_service_rejects_foreign_port_while_our_process_alive(
    cfg: UserConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Port open before our bind must not count as success (LM Studio case)."""
    proc = MagicMock()
    proc.pid = 88
    monkeypatch.setattr("launcher.process_manager.is_port_open", lambda *a, **k: True)
    monkeypatch.setattr("launcher.process_manager.is_process_running", lambda pid: True)
    monkeypatch.setattr(
        "launcher.process_manager.port_owned_by_pid_tree",
        lambda host, port, root_pid, **k: False,
    )
    monkeypatch.setattr("launcher.process_manager.stop_service", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match=r"not owned by|ägs inte av"):
        _wait_for_service(cfg, "dashboard", proc, timeout_sec=1.0)


def test_stop_service_kills_tracked_pid_on_windows(
    cfg: UserConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from launcher.pid_store import get_pid_info, save_pid

    save_pid(cfg, "api", 9999, ["python", "-m", "uvicorn"])
    monkeypatch.setattr("launcher.process_manager.is_process_running", lambda pid: True)
    monkeypatch.setattr("launcher.process_manager.sys.platform", "win32")
    with patch("launcher.process_manager.subprocess.run") as mock_run:
        assert stop_service(cfg, "api") is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[:3] == ["taskkill", "/PID", "9999"]
    assert get_pid_info(cfg, "api") is None


def test_wait_for_service_raises_when_process_exits(cfg: UserConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = MagicMock()
    proc.pid = 77
    monkeypatch.setattr("launcher.process_manager.is_port_open", lambda *a, **k: False)
    monkeypatch.setattr("launcher.process_manager.is_process_running", lambda pid: False)
    with pytest.raises(RuntimeError, match="exited before listening"):
        _wait_for_service(cfg, "api", proc, timeout_sec=1.0)


def test_restart_named_services_starts_requested(
    cfg: UserConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []
    monkeypatch.setattr(
        "launcher.process_manager.start_api",
        lambda c, log=None: started.append("api") or MagicMock(pid=1),
    )
    monkeypatch.setattr(
        "launcher.process_manager.start_dashboard",
        lambda c, log=None: started.append("dashboard") or MagicMock(pid=2),
    )
    restart_named_services(cfg, ["api", "dashboard"])
    assert started == ["api", "dashboard"]
