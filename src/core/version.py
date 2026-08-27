"""Single package version source for API metadata and Prometheus info."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_PACKAGE_NAME = "automatisk-sentimentanalys"
_FALLBACK_VERSION = "0.5.1"


def get_package_version() -> str:
    """Return the installed package version, or the repo fallback."""
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION
