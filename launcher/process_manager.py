"""Start/stop API and Streamlit child processes."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.install.config_schema import UserConfig

from .env_builder import build_child_env, resolve_python, working_directory
from .process_util import is_process_running


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
    proc = subprocess.Popen(
        cmd,
        cwd=str(working_directory(cfg)),
        env=build_child_env(cfg),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    _save_pid(cfg, "api", proc.pid, cmd)
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
    proc = subprocess.Popen(
        cmd,
        cwd=str(working_directory(cfg)),
        env=build_child_env(cfg),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    _save_pid(cfg, "dashboard", proc.pid, cmd)
    return ProcessInfo(name="dashboard", pid=proc.pid, command=cmd)


def service_status(cfg: UserConfig, name: str) -> str:
    info = _load_pid(cfg, name)
    if not info:
        return "stopped"
    if is_process_running(info.pid):
        return f"running (pid {info.pid})"
    _pid_file(cfg, name).unlink(missing_ok=True)
    return "stopped"
