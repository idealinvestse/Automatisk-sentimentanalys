"""Launcher CLI for IT automation and doctor checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.install.config_schema import InstallProfile, UserConfig
from src.install.preflight import run_preflight
from src.install.secrets_win import delete_secret, set_secret
from src.install.user_config import load_user_config, save_user_config

from .env_builder import resolve_python, working_directory
from .process_manager import start_api, start_dashboard, stop_service
from .status_snapshot import collect_snapshot

app = typer.Typer(help="Sentimentanalys Windows launcher")
console = Console()


def _app_root_option() -> Path:
    return Path(os.environ.get("SENTIMENT_APP_ROOT", Path.cwd()))


@app.command("doctor")
def doctor_cmd(
    app_root: Path = typer.Option(None, help="Application root directory"),
    require_openrouter: bool | None = typer.Option(None, "--require-openrouter"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Run installation health checks."""
    root = app_root or _app_root_option()
    cfg = load_user_config(root)
    report = run_preflight(cfg, require_openrouter=require_openrouter)
    if json_out:
        payload = {
            "ok": report.ok,
            "checks": [
                {"name": c.name, "ok": c.ok, "message": c.message, "detail": c.detail}
                for c in report.checks
            ],
        }
        console.print_json(json.dumps(payload))
        raise typer.Exit(code=0 if report.ok else 1)

    table = Table(title="Sentimentanalys Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Message")
    for c in report.checks:
        status = "[green]OK[/green]" if c.ok else "[red]FAIL[/red]"
        table.add_row(c.name, status, c.message + (f" ({c.detail})" if c.detail else ""))
    console.print(table)
    raise typer.Exit(code=0 if report.ok else 1)


@app.command("configure")
def configure_cmd(
    app_root: Path = typer.Option(None, help="Application root"),
    profile: InstallProfile | None = typer.Option(None, "--profile"),
    sentiment_profile: str | None = typer.Option(None, "--sentiment-profile"),
    device: str | None = typer.Option(None, "--device"),
    api_port: int | None = typer.Option(None, "--api-port"),
    portable: bool | None = typer.Option(None, "--portable/--no-portable"),
    export_path: Path | None = typer.Option(None, "--export", help="Export config JSON bundle"),
    import_path: Path | None = typer.Option(
        None, "--import-bundle", help="Import config JSON bundle"
    ),
    init: bool = typer.Option(False, "--init", help="Create default user_config.yaml"),
) -> None:
    """Create or update user configuration."""
    root = app_root or _app_root_option()
    if import_path:
        data = json.loads(import_path.read_text(encoding="utf-8"))
        cfg = UserConfig.model_validate(data.get("config", data))
        if not cfg.paths.app_root:
            cfg.paths.app_root = str(root.resolve())
        save_user_config(cfg)
        console.print(f"[green]Imported config from {import_path}[/green]")
        return

    cfg = load_user_config(root, create_if_missing=init)
    if profile is not None:
        cfg.install_profile = profile
    if sentiment_profile is not None:
        cfg.sentiment_profile = sentiment_profile
    if device is not None:
        cfg.device = device  # type: ignore[assignment]
    if api_port is not None:
        cfg.services.api_port = api_port
    if portable is not None:
        cfg.portable_mode = portable
    if not cfg.paths.app_root:
        cfg.paths.app_root = str(root.resolve())

    path = save_user_config(cfg)
    console.print(f"[green]Saved {path}[/green]")

    if export_path:
        bundle = {"config": cfg.model_dump(mode="json")}
        export_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        console.print(f"[green]Exported {export_path}[/green]")


@app.command("set-secret")
def set_secret_cmd(
    kind: str = typer.Argument(..., help="openrouter or huggingface"),
    value: str | None = typer.Option(None, help="Secret value"),
    from_file: Path | None = typer.Option(None, "--from-file"),
) -> None:
    if kind not in ("openrouter", "huggingface"):
        raise typer.BadParameter("kind must be openrouter or huggingface")
    secret = value
    if from_file:
        secret = from_file.read_text(encoding="utf-8").strip().splitlines()[0]
    if not secret:
        raise typer.BadParameter("Provide --value or --from-file")
    set_secret(kind, secret)  # type: ignore[arg-type]
    console.print(f"[green]Stored {kind} secret[/green]")


@app.command("clear-secret")
def clear_secret_cmd(kind: str = typer.Argument(...)) -> None:
    delete_secret(kind)  # type: ignore[arg-type]
    console.print(f"[green]Cleared {kind}[/green]")


@app.command("start-api")
def start_api_cmd(app_root: Path | None = None) -> None:
    cfg = load_user_config(app_root or _app_root_option())
    info = start_api(cfg)
    console.print(f"API started pid={info.pid} http://{cfg.services.api_host}:{cfg.services.api_port}")


@app.command("stop-api")
def stop_api_cmd(app_root: Path | None = None) -> None:
    cfg = load_user_config(app_root or _app_root_option())
    stop_service(cfg, "api")
    console.print("API stopped")


@app.command("start-dashboard")
def start_dashboard_cmd(app_root: Path | None = None) -> None:
    cfg = load_user_config(app_root or _app_root_option())
    info = start_dashboard(cfg)
    console.print(
        f"Dashboard started pid={info.pid} http://localhost:{cfg.services.dashboard_port}"
    )


@app.command("stop-dashboard")
def stop_dashboard_cmd(app_root: Path | None = None) -> None:
    cfg = load_user_config(app_root or _app_root_option())
    stop_service(cfg, "dashboard")
    console.print("Dashboard stopped")


@app.command("status")
def status_cmd(app_root: Path | None = None) -> None:
    root = app_root or _app_root_option()
    cfg = load_user_config(root)
    snap = collect_snapshot(cfg, launcher_root=root.resolve())

    table = Table(title="Launcher status")
    table.add_column("Field")
    table.add_column("Value")
    for svc in (snap.api, snap.dashboard):
        table.add_row(f"{svc.name} state", svc.state_label)
        table.add_row(f"{svc.name} url", svc.url if svc.port_open else "—")
        table.add_row(f"{svc.name} pid", str(svc.pid) if svc.pid else "—")
        table.add_row(f"{svc.name} port", "open" if svc.port_open else "closed")
        if svc.health_ok is not None:
            table.add_row(f"{svc.name} health", "ok" if svc.health_ok else "fail")
    table.add_row("app_root", str(snap.system.app_root))
    table.add_row("launcher_root", str(snap.system.launcher_root))
    table.add_row("config", str(snap.system.config_path))
    table.add_row("python", str(snap.system.python_exe))
    table.add_row("profile", f"{snap.system.install_profile} / {snap.system.sentiment_profile}")
    table.add_row("device", snap.system.device)
    table.add_row("llm", "on" if snap.system.llm_enabled else "off")
    table.add_row("collected_at", snap.collected_at)
    console.print(table)


@app.command("open-cli")
def open_cli_cmd(app_root: Path | None = None) -> None:
    """Open PowerShell with venv activated (Windows)."""
    cfg = load_user_config(app_root or _app_root_option())
    root = working_directory(cfg)
    py = resolve_python(cfg)
    activate = root / ".venv" / "Scripts" / "Activate.ps1"
    if sys.platform == "win32":
        if activate.is_file():
            cmd = f'powershell -NoExit -Command "cd \'{root}\'; . \'{activate}\'"'
        else:
            cmd = f'powershell -NoExit -Command "cd \'{root}\'; $env:PYTHONPATH=\'{root}\'"'
        subprocess.Popen(cmd, shell=True)
    else:
        console.print(f"Run: cd {root} && {py} -m src.cli --help")


@app.command("repair")
def repair_cmd(
    profile: InstallProfile = typer.Option(InstallProfile.cli, "--profile"),
    app_root: Path | None = None,
) -> None:
    """Re-install pip dependencies for the selected profile."""
    root = app_root or _app_root_option()
    py = resolve_python(cfg := load_user_config(root))
    _install = "requirements-install.txt"
    req_files = {
        InstallProfile.minimal: ["requirements-min.txt", _install],
        InstallProfile.cli: ["requirements-min.txt", "requirements-cli.txt", _install],
        InstallProfile.api: ["requirements-min.txt", "requirements-api.txt", _install],
        InstallProfile.full: [
            "requirements-min.txt",
            "requirements-cli.txt",
            "requirements-api.txt",
            "requirements.txt",
            "requirements-desktop.txt",
            _install,
        ],
        InstallProfile.dev: [
            "requirements-min.txt",
            "requirements-cli.txt",
            "requirements-api.txt",
            "requirements.txt",
            "requirements-desktop.txt",
            _install,
            "requirements-dev.txt",
        ],
    }
    files = req_files[profile]
    for rf in files:
        path = root / rf
        if not path.is_file():
            console.print(f"[yellow]Skip missing {rf}[/yellow]")
            continue
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(path)], check=True, cwd=str(root))
    cfg.install_profile = profile
    save_user_config(cfg)
    console.print("[green]Repair complete[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
