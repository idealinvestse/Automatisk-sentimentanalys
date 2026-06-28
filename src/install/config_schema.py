"""Pydantic schema for launcher and setup hub user configuration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class InstallProfile(str, Enum):
    """Pip requirement bundle installed in the environment."""

    minimal = "minimal"
    cli = "cli"
    api = "api"
    full = "full"
    dev = "dev"


class PathsConfig(BaseModel):
    """Filesystem paths (absolute or relative to app root)."""

    app_root: str = ""
    hf_cache: str = "cache/hf"
    llm_cache: str = "cache/llm"
    outputs: str = "outputs"
    logs: str = ""
    incoming_calls: str = ""
    state_dir: str = ""


class AsrDefaults(BaseModel):
    backend: Literal["faster", "transformers", "whisperx"] = "faster"
    model: str = "kb-whisper-large"
    revision: Literal["standard", "strict", "subtitle"] = "strict"
    language: str = "sv"
    hotwords_file: str = "configs/callcenter_hotwords.txt"
    preprocess: bool = False
    preprocess_mode: Literal["off", "basic", "callcenter"] = "off"

    @field_validator("preprocess_mode", mode="before")
    @classmethod
    def _coerce_preprocess_mode(cls, value: object) -> object:
        if isinstance(value, bool):
            return "basic" if value else "off"
        return value


class LlmConfig(BaseModel):
    enabled: bool = False
    provider: str = "openrouter"
    default_model: str = "mistralai/mistral-medium-3.5"
    cost_budget_per_call: float = 0.08
    anonymize_before_llm: bool = False
    log_external_calls: bool = True


class ServicesConfig(BaseModel):
    api_enabled: bool = True
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    dashboard_ui: Literal["nicegui"] = "nicegui"

    @field_validator("dashboard_ui", mode="before")
    @classmethod
    def _coerce_dashboard_ui(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() == "streamlit":
            return "nicegui"
        return value


class ApiRuntimeConfig(BaseModel):
    api_key: str = ""
    cors_origins: str = ""
    media_root: str = ""
    rate_limit_rpm: int = 0
    use_redis_cache: bool = False
    redis_url: str = ""
    allow_client_llm_key: bool = False


class AlertingRuntimeConfig(BaseModel):
    webhook_enabled: bool = False
    webhook_url: str = ""
    timeout_seconds: int = 10
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    retry_backoff_base: float = 1.0


class DashboardRuntimeConfig(BaseModel):
    api_base_url: str = "http://127.0.0.1:8000"
    storage_secret: str = ""
    dev_mode: bool = False


class RuntimeConfig(BaseModel):
    api: ApiRuntimeConfig = Field(default_factory=ApiRuntimeConfig)
    alerting: AlertingRuntimeConfig = Field(default_factory=AlertingRuntimeConfig)
    dashboard: DashboardRuntimeConfig = Field(default_factory=DashboardRuntimeConfig)


class UserConfig(BaseModel):
    """User-facing settings persisted outside the git tree."""

    version: int = 1
    install_profile: InstallProfile = InstallProfile.cli
    sentiment_profile: str = "callcenter"
    device: Literal["auto", "cpu", "cuda", "cuda:0", "mps"] = "auto"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    asr: AsrDefaults = Field(default_factory=AsrDefaults)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    portable_mode: bool = False

    @field_validator("sentiment_profile")
    @classmethod
    def _normalize_profile(cls, v: str) -> str:
        return v.strip().lower() if v else "default"

    def resolved_app_root(self) -> Path:
        if self.paths.app_root:
            return Path(self.paths.app_root).resolve()
        return Path.cwd().resolve()

    def resolved_user_data_dir(self) -> Path:
        if self.portable_mode:
            return self.resolved_app_root() / "user_data"
        import os

        override = os.environ.get("SENTIMENT_USER_DATA", "").strip()
        if override:
            return Path(override).expanduser()
        return Path.home() / "AppData" / "Roaming" / "Sentimentanalys"

    def resolved_logs_dir(self) -> Path:
        if self.paths.logs:
            return Path(self.paths.logs).resolve()
        return self.resolved_user_data_dir() / "logs"

    def resolved_hf_home(self) -> Path:
        root = self.resolved_app_root()
        return (root / self.paths.hf_cache).resolve()
