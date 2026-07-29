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


def check_webui_package(
    python: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str | None:
    """Verify that webui/package.json and node_modules exist under the working directory."""
    del python, env  # unused; signature kept for callers
    root = Path(cwd) if cwd else Path.cwd()
    package_json = root / "webui" / "package.json"
    if not package_json.is_file():
        return f"webui/package.json saknas under {root}"
    node_modules = root / "webui" / "node_modules"
    if not node_modules.is_dir():
        return f"webui/node_modules saknas under {root}. " "Kör: cd webui && npm install"
    return None


# Backward-compatible alias
check_dashboard_import = check_webui_package


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
    package_err = check_webui_package(cwd=cwd)
    if package_err:
        return (
            f"Dashboard kunde inte laddas: {package_err}. "
            "Kontrollera att webui/ finns i repo-roten och kör npm install."
        )
    return None
