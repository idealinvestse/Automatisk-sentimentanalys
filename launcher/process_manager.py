"""Start/stop API and NiceGUI dashboard child processes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.install.config_schema import UserConfig

from .api_deps import check_api_dependencies
from .dashboard_deps import check_dashboard_dependencies
from .env_builder import build_child_env, resolve_python, working_directory
from .event_log import EventLog
from .pid_store import clear_pid_file, get_pid_info, save_pid, service_log_paths
from .process_util import is_port_open, is_process_running, resolve_connect_host

_START_TIMEOUT_SEC = 30.0
_LOG_TAIL_CHARS = 1500
_TICK_INTERVAL_SEC = 0.25


@dataclass
class ProcessInfo:
    name: str
    pid: int
    command: list[str]
    pid_file: Path


def _read_log_tail(path: Path, max_chars: int = _LOG_TAIL_CHARS) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:].strip()


def _windows_creationflags() -> int:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags |= subprocess.CREATE_NO_WINDOW  # type: ignore[operator]
    return flags


def _popen_service(
    cfg: UserConfig,
    name: str,
    cmd: list[str],
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Start a detached child; redirect stdio so pythonw parents do not kill it."""
    out_path, err_path = service_log_paths(cfg, name)
    env = build_child_env(cfg)
    if extra_env:
        env.update(extra_env)
    with (
        out_path.open("a", encoding="utf-8") as stdout,
        err_path.open("a", encoding="utf-8") as stderr,
    ):
        return subprocess.Popen(
            cmd,
            cwd=str(working_directory(cfg)),
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=_windows_creationflags() if sys.platform == "win32" else 0,
        )


def _service_endpoint(cfg: UserConfig, name: str) -> tuple[str, int] | None:
    if name == "api":
        return cfg.services.api_host, cfg.services.api_port
    if name == "dashboard":
        return "127.0.0.1", cfg.services.dashboard_port
    return None


def _wait_for_service(
    cfg: UserConfig,
    name: str,
    proc: subprocess.Popen[bytes],
    *,
    log: EventLog | None = None,
    on_tick: Callable[[float, bool, bool], None] | None = None,
) -> None:
    endpoint = _service_endpoint(cfg, name)
    phase = f"{name}.wait_port"
    if endpoint is None:
        time.sleep(0.5)
        if not is_process_running(proc.pid):
            raise RuntimeError(f"{name} process exited immediately (pid {proc.pid})")
        return

    host, port = endpoint
    connect_host = resolve_connect_host(host)
    if log:
        log.info(
            f"Waiting for {connect_host}:{port} (bind {host}, timeout {_START_TIMEOUT_SEC:.0f}s)",
            phase=phase,
        )

    deadline = time.monotonic() + _START_TIMEOUT_SEC
    last_logged_sec = -1.0
    while time.monotonic() < deadline:
        elapsed = _START_TIMEOUT_SEC - (deadline - time.monotonic())
        port_open = is_port_open(host, port, timeout=0.2)
        alive = is_process_running(proc.pid)
        if on_tick:
            on_tick(elapsed, port_open, alive)
        if log and int(elapsed) > int(last_logged_sec):
            last_logged_sec = elapsed
            log.info(
                f"elapsed {elapsed:.1f}s · port={'open' if port_open else 'closed'} · "
                f"process={'alive' if alive else 'dead'}",
                phase=phase,
            )
        if port_open:
            if log:
                log.info(f"Port {host}:{port} is accepting connections", phase=phase)
            if name == "api":
                from .status_snapshot import check_api_health

                if log:
                    log.info("Checking GET /health", phase=f"{name}.health")
                if check_api_health(connect_host, port):
                    if log:
                        log.info("Health check OK", phase=f"{name}.health")
                elif log:
                    log.warn("Port open but /health did not return ok", phase=f"{name}.health")
            return
        if not alive:
            _, err_path = service_log_paths(cfg, name)
            detail = _read_log_tail(err_path) or _read_log_tail(service_log_paths(cfg, name)[0])
            if log:
                log.error(
                    f"Process exited before port was open (pid {proc.pid})",
                    phase=phase,
                )
                if detail:
                    log.error(detail[:500], phase=phase)
            raise RuntimeError(
                f"{name} exited before listening on {host}:{port}. "
                f"See {err_path}" + (f":\n{detail}" if detail else "")
            )
        time.sleep(_TICK_INTERVAL_SEC)

    stop_service(cfg, name, log=log)
    _, err_path = service_log_paths(cfg, name)
    if log:
        log.error(f"Timeout after {_START_TIMEOUT_SEC:.0f}s waiting for {host}:{port}", phase=phase)
    raise RuntimeError(
        f"{name} did not listen on {host}:{port} within {_START_TIMEOUT_SEC:.0f}s. See {err_path}"
    )


def stop_service(cfg: UserConfig, name: str, *, log: EventLog | None = None) -> bool:
    phase = f"{name}.stop"
    info = get_pid_info(cfg, name)
    if not info:
        if log:
            log.info("No tracked process", phase=phase)
        return False
    if log:
        log.phase(phase, f"Stopping pid {info.pid}")
    if not is_process_running(info.pid):
        clear_pid_file(cfg, name)
        if log:
            log.info("Process already stopped; cleared pid file", phase=phase)
        return True
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(info.pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        else:
            import os

            os.kill(info.pid, signal.SIGTERM)
    except OSError:
        pass
    clear_pid_file(cfg, name)
    if log:
        log.info("Stop complete", phase=phase)
    return True


def start_api(cfg: UserConfig, *, log: EventLog | None = None) -> ProcessInfo:
    if log:
        log.phase("api.start", "Starting API")
    stop_service(cfg, "api", log=log)
    py = resolve_python(cfg)
    child_env = build_child_env(cfg)
    root = working_directory(cfg)
    dep_error = check_api_dependencies(python=py, env=child_env, cwd=root)
    if dep_error:
        if log:
            log.error(dep_error, phase="api.start")
        raise RuntimeError(dep_error)
    cmd = [
        str(py),
        "-m",
        "uvicorn",
        "src.api:app",
        "--host",
        cfg.services.api_host,
        "--port",
        str(cfg.services.api_port),
    ]
    if log:
        log.info(f"Command: {' '.join(cmd)}", phase="api.start")
        log.info(f"Working directory: {root}", phase="api.start")
    proc = _popen_service(cfg, "api", cmd)
    pid_path = save_pid(cfg, "api", proc.pid, cmd)
    if log:
        log.info(f"Spawned pid {proc.pid} (tracked in {pid_path})", phase="api.start")
    _wait_for_service(cfg, "api", proc, log=log)
    if log:
        log.info(
            f"API ready at http://{cfg.services.api_host}:{cfg.services.api_port}",
            phase="api.start",
        )
    return ProcessInfo(name="api", pid=proc.pid, command=cmd, pid_file=pid_path)


def start_dashboard(cfg: UserConfig, *, log: EventLog | None = None) -> ProcessInfo:
    if log:
        log.phase("dashboard.start", "Starting Dashboard")
    stop_service(cfg, "dashboard", log=log)
    py = resolve_python(cfg)
    child_env = build_child_env(cfg)
    root = working_directory(cfg)
    dep_error = check_dashboard_dependencies(python=py, env=child_env, cwd=root)
    if dep_error:
        if log:
            log.error(dep_error, phase="dashboard.start")
        raise RuntimeError(dep_error)
    dashboard_ui = getattr(cfg.services, "dashboard_ui", "nicegui") or "nicegui"
    if dashboard_ui == "streamlit":
        if log:
            log.warn(
                "DASHBOARD_UI=streamlit är avvecklad. Startar NiceGUI istället.",
                phase="dashboard.start",
            )
        dashboard_ui = "nicegui"
    extra_env = {
        "DASHBOARD_UI": dashboard_ui,
        "NICEGUI_PORT": str(cfg.services.dashboard_port),
    }
    cmd = [str(py), "-m", "app.dashboard_launcher"]
    if log:
        log.info(f"Command: {' '.join(cmd)} (ui={dashboard_ui})", phase="dashboard.start")
    proc = _popen_service(cfg, "dashboard", cmd, extra_env=extra_env)
    pid_path = save_pid(cfg, "dashboard", proc.pid, cmd)
    if log:
        log.info(f"Spawned pid {proc.pid}", phase="dashboard.start")
    _wait_for_service(cfg, "dashboard", proc, log=log)
    if log:
        log.info(
            f"Dashboard ready at http://localhost:{cfg.services.dashboard_port}",
            phase="dashboard.start",
        )
    return ProcessInfo(name="dashboard", pid=proc.pid, command=cmd, pid_file=pid_path)


def service_status(cfg: UserConfig, name: str) -> str:
    from .status_snapshot import service_status_text

    return service_status_text(cfg, name)
