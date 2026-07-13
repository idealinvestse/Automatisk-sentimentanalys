"""Runtime checks for dashboard (Next.js webui) startup dependencies."""

from __future__ import annotations

import shutil
from pathlib import Path


def missing_dashboard_modules() -> list[str]:
    """Return names of required dashboard tools that are not available."""
    missing: list[str] = []
    if shutil.which("node") is None:
        missing.append("node")
    if shutil.which("npm") is None:
        missing.append("npm")
    return missing


def check_dashboard_import(
    python: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str | None:
    """Verify that webui/package.json exists under the working directory."""
    del python, env  # unused; signature kept for callers
    root = Path(cwd) if cwd else Path.cwd()
    package_json = root / "webui" / "package.json"
    if package_json.is_file():
        return None
    return f"webui/package.json saknas under {root}"


def check_dashboard_dependencies(
    *,
    python: Path | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str | None:
    """Return a user-facing error message when webui dependencies are missing."""
    del python, env
    missing = missing_dashboard_modules()
    if missing:
        mods = ", ".join(missing)
        return (
            f"Dashboard-beroenden saknas ({mods}). "
            "Installera Node.js (inkl. npm), sedan: cd webui && npm install"
        )
    import_err = check_dashboard_import(cwd=cwd)
    if import_err:
        return (
            f"Dashboard kunde inte laddas: {import_err}. "
            "Kontrollera att webui/ finns i repo-roten."
        )
    return None
