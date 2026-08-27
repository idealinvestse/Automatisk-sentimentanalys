"""Package version helper used by FastAPI metadata and Prometheus."""

from src.core.version import get_package_version


def test_package_version_is_nonempty() -> None:
    version = get_package_version()
    assert version
    assert version[0].isdigit()
