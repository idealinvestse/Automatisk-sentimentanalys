"""Windows installer / launcher configuration and preflight checks."""

from .config_schema import InstallProfile, UserConfig
from .preflight import PreflightReport, run_preflight
from .user_config import (
    default_user_config_path,
    load_user_config,
    merge_configs,
    save_user_config,
)

__all__ = [
    "InstallProfile",
    "UserConfig",
    "PreflightReport",
    "run_preflight",
    "default_user_config_path",
    "load_user_config",
    "merge_configs",
    "save_user_config",
]
