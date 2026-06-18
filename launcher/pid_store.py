"""PID file tracking for launcher-managed services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.install.config_schema import UserConfig


@dataclass
class PidRecord:
    name: str
    pid: int
    command: list[str]
    pid_file: Path


def run_dir(cfg: UserConfig) -> Path:
    d = cfg.resolved_user_data_dir() / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file_path(cfg: UserConfig, name: str) -> Path:
    return run_dir(cfg) / f"{name}.json"


def service_log_paths(cfg: UserConfig, name: str) -> tuple[Path, Path]:
    log_dir = cfg.resolved_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}.log", log_dir / f"{name}.err.log"


def launcher_activity_log_path(cfg: UserConfig) -> Path:
    """Persistent launcher GUI activity log."""
    log_dir = cfg.resolved_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "launcher_activity.log"


def save_pid(cfg: UserConfig, name: str, pid: int, command: list[str]) -> Path:
    path = pid_file_path(cfg, name)
    path.write_text(
        json.dumps({"pid": pid, "command": command}, indent=2),
        encoding="utf-8",
    )
    return path


def get_pid_info(cfg: UserConfig, name: str) -> PidRecord | None:
    path = pid_file_path(cfg, name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PidRecord(
            name=name,
            pid=int(data["pid"]),
            command=list(data.get("command", [])),
            pid_file=path,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def clear_pid_file(cfg: UserConfig, name: str) -> None:
    pid_file_path(cfg, name).unlink(missing_ok=True)
