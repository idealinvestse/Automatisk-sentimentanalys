"""Launch the primary Next.js web UI (webui/).

Usage:
    python -m app.dashboard_launcher
    # or: sentimentanalys-dashboard

Requires Node.js/npm and dependencies installed in webui/ (`npm install`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_dashboard_ui() -> str:
    """Return dashboard backend id from DASHBOARD_UI env (default: webui)."""
    return os.environ.get("DASHBOARD_UI", "webui").strip().lower()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    """Start Next.js webui in development mode."""
    ui = resolve_dashboard_ui()
    if ui not in {"webui", "next", "nextjs"}:
        print(
            f"Okänt DASHBOARD_UI={ui!r}. Giltiga värden: webui",
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
    cmd = [npm, "run", "dev", "--", "-p", str(port)]
    print(f"Startar webui: {' '.join(cmd)} (cwd={webui_dir})")
    raise SystemExit(subprocess.call(cmd, cwd=str(webui_dir)))


if __name__ == "__main__":
    main()
