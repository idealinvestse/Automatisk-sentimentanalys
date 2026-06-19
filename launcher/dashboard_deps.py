"""Runtime checks for dashboard startup dependencies."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_DASHBOARD_MODULES = ("nicegui", "httpx")


def missing_dashboard_modules() -> list[str]:
    """Return names of required dashboard modules that are not importable."""
    return [mod for mod in _DASHBOARD_MODULES if importlib.util.find_spec(mod) is None]


def check_dashboard_import(
    python: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str | None:
    """Verify that ``app.nicegui_dashboard.main`` can be imported in the target interpreter."""
    py = python or Path(sys.executable)
    result = subprocess.run(
        [str(py), "-c", "import app.nicegui_dashboard.main"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout or "").strip()
    return detail.splitlines()[-1] if detail else "dashboard import failed"


def check_dashboard_dependencies(
    *,
    python: Path | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str | None:
    """Return a user-facing error message when dashboard dependencies are missing."""
    py = python or Path(sys.executable)
    missing = missing_dashboard_modules()
    if missing:
        probe = subprocess.run(
            [str(py), "-c", "import nicegui, httpx"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        if probe.returncode != 0:
            mods = ", ".join(missing)
            return (
                f"Dashboard-beroenden saknas ({mods}). "
                "Kör 'Installera / Reparera allt' i launchern eller: "
                r".\launcher.ps1 provision"
            )
    import_err = check_dashboard_import(py, env=env, cwd=cwd)
    if import_err:
        return (
            f"Dashboard kunde inte laddas: {import_err}. "
            "Kör 'Installera / Reparera allt' i launchern eller: "
            r".\launcher.ps1 provision"
        )
    return None