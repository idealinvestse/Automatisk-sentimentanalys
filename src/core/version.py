"""Single package version source for API metadata and Prometheus info."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE_NAME = "automatisk-sentimentanalys"
_FALLBACK_VERSION = "0.5.1"


def get_package_version() -> str:
    """Return the package version from pyproject.toml if present, or installed metadata/fallback."""
    try:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.is_file():
            import tomllib

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            ver = data.get("project", {}).get("version")
            if ver:
                return str(ver)
    except Exception:
        pass
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION
