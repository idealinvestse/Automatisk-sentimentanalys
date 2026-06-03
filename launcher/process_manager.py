"""Start/stop API and Streamlit child processes."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.install.config_schema import UserConfig

from .env_builder import build_child_env, resolve_python, working_directory
from .process_util import is_port_open, is_process_running

_START_TIMEOUT_SEC = 30.0
_LOG_TAIL_CHARS = 1500


@dataclass
class ProcessInfo:
    name: str
    pid: int
    command: list[str]


def _run_dir(cfg: UserConfig) -> Path:
    d = cfg.resolved_user_data_dir() / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pid_file(cfg: UserConfig, name: str) -> Path:
    return _run_dir(cfg) / f"{name}.json"


def _service_log_paths(cfg: UserConfig, name: str) -> tuple[Path, Path]:
    log_dir = cfg.resolved_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}.log", log_dir / f"{name}.err.log"


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


def _popen_service(cfg: UserConfig, name: str, cmd: list[str]) -> subprocess.Popen[bytes]:
    """Start a detached child; redirect stdio so pythonw parents do not kill it."""
    out_path, err_path = _service_log_paths(cfg, name)
    # Append so repeated starts keep history; child keeps handles after parent closes (Windows).
    with (
        out_path.open("a", encoding="utf-8") as stdout,
        err_path.open("a", encoding="utf-8") as stderr,
    ):
        return subprocess.Popen(
            cmd,
            cwd=str(working_directory(cfg)),
            env=build_child_env(cfg),
            stdout=stdout,
            stderr=stderr,
            creationflags=_windows_creationflags() if sys.platform == "win32" else 0,
        )


def _save_pid(cfg: UserConfig, name: str, pid: int, command: list[str]) -> None:
    path = _pid_file(cfg, name)
    path.write_text(
        json.dumps({"pid": pid, "command": command}, indent=2),
        encoding="utf-8",
    )


def _load_pid(cfg: UserConfig, name: str) -> ProcessInfo | None:
    path = _pid_file(cfg, name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProcessInfo(name=name, pid=int(data["pid"]), command=list(data.get("command", [])))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _service_endpoint(cfg: UserConfig, name: str) -> tuple[str, int] | None:
    if name == "api":
        return cfg.services.api_host, cfg.services.api_port
    if name == "dashboard":
        return "127.0.0.1", cfg.services.dashboard_port
    return None


def _wait_for_service(cfg: UserConfig, name: str, proc: subprocess.Popen[bytes]) -> None:
    endpoint = _service_endpoint(cfg, name)
    if endpoint is None:
        time.sleep(0.5)
        if not is_process_running(proc.pid):
            raise RuntimeError(f"{name} process exited immediately (pid {proc.pid})")
        return

    host, port = endpoint
    deadline = time.monotonic() + _START_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if is_port_open(host, port):
            return
        if not is_process_running(proc.pid):
            _, err_path = _service_log_paths(cfg, name)
            detail = _read_log_tail(err_path) or _read_log_tail(_service_log_paths(cfg, name)[0])
            raise RuntimeError(
                f"{name} exited before listening on {host}:{port}. "
                f"See {err_path}" + (f":\n{detail}" if detail else "")
            )
        time.sleep(0.25)

    stop_service(cfg, name)
    _, err_path = _service_log_paths(cfg, name)
    raise RuntimeError(
        f"{name} did not listen on {host}:{port} within {_START_TIMEOUT_SEC:.0f}s. See {err_path}"
    )


def stop_service(cfg: UserConfig, name: str) -> bool:
    info = _load_pid(cfg, name)
    if not info:
        return False
    if not is_process_running(info.pid):
        _pid_file(cfg, name).unlink(missing_ok=True)
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
    _pid_file(cfg, name).unlink(missing_ok=True)
    return True


def start_api(cfg: UserConfig) -> ProcessInfo:
    stop_service(cfg, "api")
    py = resolve_python(cfg)
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
    proc = _popen_service(cfg, "api", cmd)
    _save_pid(cfg, "api", proc.pid, cmd)
    _wait_for_service(cfg, "api", proc)
    return ProcessInfo(name="api", pid=proc.pid, command=cmd)


def start_dashboard(cfg: UserConfig) -> ProcessInfo:
    stop_service(cfg, "dashboard")
    py = resolve_python(cfg)
    cmd = [
        str(py),
        "-m",
        "streamlit",
        "run",
        "app/dashboard.py",
        "--server.port",
        str(cfg.services.dashboard_port),
        "--server.headless",
        "true",
    ]
    proc = _popen_service(cfg, "dashboard", cmd)
    _save_pid(cfg, "dashboard", proc.pid, cmd)
    _wait_for_service(cfg, "dashboard", proc)
    return ProcessInfo(name="dashboard", pid=proc.pid, command=cmd)


def service_status(cfg: UserConfig, name: str) -> str:
    endpoint = _service_endpoint(cfg, name)
    listening = is_port_open(*endpoint) if endpoint else False
    info = _load_pid(cfg, name)
    if listening:
        if info and is_process_running(info.pid):
            return f"running (pid {info.pid})"
        return "running"
    if info:
        if is_process_running(info.pid):
            return "starting"
        _pid_file(cfg, name).unlink(missing_ok=True)
    return "stopped"