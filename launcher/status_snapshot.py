"""Structured launcher status for GUI and CLI."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from src.install.config_schema import UserConfig
from src.install.secrets_win import secret_status
from src.install.user_config import default_user_config_path

from .env_builder import resolve_python
from .pid_store import get_pid_info, service_log_paths
from .process_util import is_port_open, is_process_running

API_VERSION = "0.4.0"
_HEALTH_TIMEOUT_SEC = 0.5


class ServiceState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    LISTENING = "listening"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass(frozen=True)
class ServiceSnapshot:
    name: str
    state: ServiceState
    host: str
    port: int
    url: str
    pid: int | None
    process_alive: bool
    port_open: bool
    health_ok: bool | None
    pid_file: Path | None
    stdout_log: Path
    stderr_log: Path
    last_error_tail: str

    @property
    def state_label(self) -> str:
        labels = {
            ServiceState.STOPPED: "Stoppad",
            ServiceState.STARTING: "Startar",
            ServiceState.LISTENING: "Lyssnar",
            ServiceState.RUNNING: "Kör",
            ServiceState.DEGRADED: "Degraderad",
            ServiceState.ERROR: "Fel",
        }
        return labels.get(self.state, self.state.value)

    def summary_line(self) -> str:
        parts = [self.state_label]
        if self.pid is not None:
            parts.append(f"pid {self.pid}")
        if self.port_open:
            parts.append("port OK")
        if self.health_ok is True:
            parts.append("health OK")
        elif self.health_ok is False:
            parts.append("health FAIL")
        return " · ".join(parts)


@dataclass(frozen=True)
class SystemSnapshot:
    launcher_root: Path
    app_root: Path
    config_path: Path
    user_data_dir: Path
    python_exe: Path
    venv_ok: bool
    install_profile: str
    sentiment_profile: str
    device: str
    llm_enabled: bool
    openrouter_configured: bool
    huggingface_configured: bool
    api_version: str


@dataclass(frozen=True)
class LauncherSnapshot:
    api: ServiceSnapshot
    dashboard: ServiceSnapshot
    system: SystemSnapshot
    collected_at: str


def _read_log_tail(path: Path, max_chars: int = 400) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()


def check_api_health(host: str, port: int, *, timeout_sec: float = _HEALTH_TIMEOUT_SEC) -> bool:
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8"))
            return isinstance(body, dict) and body.get("status") in ("ok", "healthy")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def _service_endpoint(cfg: UserConfig, name: str) -> tuple[str, int]:
    if name == "api":
        return cfg.services.api_host, cfg.services.api_port
    return "127.0.0.1", cfg.services.dashboard_port


def _build_service_snapshot(cfg: UserConfig, name: str) -> ServiceSnapshot:
    host, port = _service_endpoint(cfg, name)
    url = f"http://{host}:{port}" if name == "api" else f"http://localhost:{port}"
    out_log, err_log = service_log_paths(cfg, name)
    port_open = is_port_open(host, port, timeout=0.35)
    info = get_pid_info(cfg, name)
    pid = info.pid if info else None
    process_alive = is_process_running(pid) if pid else False
    pid_path = info.pid_file if info else None

    health_ok: bool | None = None
    if name == "api" and port_open:
        health_ok = check_api_health(host, port)

    state = ServiceState.STOPPED
    if port_open and process_alive and (health_ok is not False or name != "api"):
        state = ServiceState.RUNNING if (name != "api" or health_ok is True) else ServiceState.LISTENING
    elif port_open and (health_ok is True or name != "api"):
        state = ServiceState.RUNNING if health_ok is not False else ServiceState.LISTENING
    elif port_open:
        state = ServiceState.DEGRADED
    elif process_alive:
        state = ServiceState.STARTING
    elif pid is not None and not process_alive:
        state = ServiceState.ERROR

    err_tail = _read_log_tail(err_log)

    return ServiceSnapshot(
        name=name,
        state=state,
        host=host,
        port=port,
        url=url,
        pid=pid,
        process_alive=process_alive,
        port_open=port_open,
        health_ok=health_ok,
        pid_file=pid_path,
        stdout_log=out_log,
        stderr_log=err_log,
        last_error_tail=err_tail,
    )


def collect_snapshot(
    cfg: UserConfig,
    *,
    launcher_root: Path | None = None,
) -> LauncherSnapshot:
    root = launcher_root or Path(os.environ.get("SENTIMENT_APP_ROOT", Path.cwd())).resolve()
    app_root = cfg.resolved_app_root()
    py = resolve_python(cfg)
    secrets = secret_status(app_root)
    collected = datetime.now(UTC).astimezone().isoformat(timespec="milliseconds")

    system = SystemSnapshot(
        launcher_root=root,
        app_root=app_root,
        config_path=default_user_config_path(cfg.portable_mode, app_root),
        user_data_dir=cfg.resolved_user_data_dir(),
        python_exe=py,
        venv_ok=(app_root / ".venv" / "Scripts" / "python.exe").is_file(),
        install_profile=cfg.install_profile.value,
        sentiment_profile=cfg.sentiment_profile,
        device=str(cfg.device),
        llm_enabled=cfg.llm.enabled,
        openrouter_configured=bool(secrets.get("openrouter", {}).get("configured")),
        huggingface_configured=bool(secrets.get("huggingface", {}).get("configured")),
        api_version=API_VERSION,
    )

    return LauncherSnapshot(
        api=_build_service_snapshot(cfg, "api"),
        dashboard=_build_service_snapshot(cfg, "dashboard"),
        system=system,
        collected_at=collected,
    )


def service_status_text(cfg: UserConfig, name: str) -> str:
    """Backward-compatible one-line status for CLI."""
    snap = collect_snapshot(cfg)
    svc = snap.api if name == "api" else snap.dashboard
    return svc.summary_line()
