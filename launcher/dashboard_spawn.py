"""Launch the primary Next.js web UI (webui/).

Usage:
    python -m launcher.dashboard_spawn
    # or: sentimentanalys-dashboard

Requires Node.js/npm and dependencies installed in webui/ (`npm install`).

Mode:
    SENTIMENT_DEV_MODE=1 (or runtime.dashboard.dev_mode) → `npm run dev`
    otherwise → `npm run build` (if .next missing) then `npm start`
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_dashboard_ui() -> str:
    """Return dashboard backend id from DASHBOARD_UI env (default: webui)."""
    return os.environ.get("DASHBOARD_UI", "webui").strip().lower() or "webui"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dev_mode() -> bool:
    flag = os.environ.get("SENTIMENT_DEV_MODE", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _run_npm(npm: str, args: list[str], *, cwd: Path) -> int:
    cmd = [npm, *args]
    print(f"Startar webui: {' '.join(cmd)} (cwd={cwd})")
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> None:
    """Start Next.js webui in development or production mode."""
    ui = resolve_dashboard_ui()
    if ui != "webui":
        print(
            f"Okänt DASHBOARD_UI={ui!r}. Giltigt värde: webui",
            file=sys.stderr,
        )
        raise SystemExit(2)

    npm = shutil.which("npm")
    if not npm:
        print(
            "npm hittades inte. Installera Node.js eller starta manuellt: cd webui && npm run dev",
            file=sys.stderr,
        )
        raise SystemExit(1)

    webui_dir = _repo_root() / "webui"
    if not (webui_dir / "package.json").is_file():
        print(f"webui/package.json saknas under {webui_dir}", file=sys.stderr)
        raise SystemExit(1)

    port = os.environ.get("PORT") or os.environ.get("WEBUI_PORT") or "3000"

    if _dev_mode():
        raise SystemExit(_run_npm(npm, ["run", "dev", "--", "-p", str(port)], cwd=webui_dir))

    next_dir = webui_dir / ".next"
    if not next_dir.is_dir():
        build_rc = _run_npm(npm, ["run", "build"], cwd=webui_dir)
        if build_rc != 0:
            raise SystemExit(build_rc)

    env = os.environ.copy()
    env["PORT"] = str(port)
    cmd = [npm, "run", "start", "--", "-p", str(port)]
    print(f"Startar webui (produktion): {' '.join(cmd)} (cwd={webui_dir})")
    raise SystemExit(subprocess.call(cmd, cwd=str(webui_dir), env=env))


if __name__ == "__main__":
    main()
