"""Runtime checks for API startup dependencies."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_API_MODULES = ("uvicorn", "fastapi")


def missing_api_modules() -> list[str]:
    """Return names of required API modules that are not importable."""
    return [mod for mod in _API_MODULES if importlib.util.find_spec(mod) is None]


def check_api_import(
    python: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str | None:
    """Verify that ``src.api`` can be imported in the target interpreter."""
    py = python or Path(sys.executable)
    result = subprocess.run(
        [str(py), "-c", "from src.api import app; assert app.title"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout or "").strip()
    return detail.splitlines()[-1] if detail else "src.api import failed"


def check_api_dependencies(
    *,
    python: Path | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str | None:
    """Return a user-facing error message when API dependencies are missing."""
    py = python or Path(sys.executable)
    missing = missing_api_modules()
    if missing:
        # Re-check in the target venv interpreter (launcher may run under pythonw).
        probe = subprocess.run(
            [str(py), "-c", "import uvicorn, fastapi"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        if probe.returncode != 0:
            mods = ", ".join(missing)
            return (
                f"API-beroenden saknas ({mods}). "
                "Kör 'Installera / Reparera allt' i launchern eller: "
                r".\launcher.ps1 provision"
            )
    import_err = check_api_import(py, env=env, cwd=cwd)
    if import_err:
        return (
            f"API kunde inte laddas: {import_err}. "
            "Kör 'Installera / Reparera allt' i launchern eller: "
            r".\launcher.ps1 provision"
        )
    return None
