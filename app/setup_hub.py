"""Streamlit setup hub for Windows installer / launcher configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("SENTIMENT_APP_ROOT", str(_ROOT))

import streamlit as st

from src.install.config_schema import InstallProfile, UserConfig
from src.install.preflight import run_preflight
from src.install.secrets_win import get_secret, secret_status, set_secret
from src.install.user_config import default_user_config_path, load_user_config, save_user_config

_PROFILES = ["default", "forum", "call", "callcenter", "news", "social", "review", "magazine"]
_DEVICES = ["auto", "cpu", "cuda", "cuda:0", "mps"]
_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

st.set_page_config(page_title="Sentimentanalys Setup", layout="wide")
st.title("Sentimentanalys — Setup Hub")
st.caption("Configure paths, services, ASR defaults, and API keys.")

cfg = load_user_config(_ROOT)

with st.sidebar:
    st.header("Install profile")
    profile = st.selectbox(
        "Pip bundle",
        options=[p.value for p in InstallProfile],
        index=[p.value for p in InstallProfile].index(cfg.install_profile.value),
    )
    portable = st.checkbox("Portable mode (config in ./user_data)", value=cfg.portable_mode)

tab_general, tab_paths, tab_asr, tab_llm, tab_secrets, tab_doctor = st.tabs(
    ["General", "Paths", "ASR", "LLM / GDPR", "Secrets", "Doctor"]
)

with tab_general:
    sentiment_profile = st.selectbox(
        "Sentiment profile",
        _PROFILES,
        index=_PROFILES.index(cfg.sentiment_profile) if cfg.sentiment_profile in _PROFILES else 0,
    )
    device = st.selectbox(
        "Device",
        _DEVICES,
        index=_DEVICES.index(cfg.device) if cfg.device in _DEVICES else 0,
    )
    log_level = st.selectbox(
        "Log level",
        _LOG_LEVELS,
        index=_LOG_LEVELS.index(cfg.log_level) if cfg.log_level in _LOG_LEVELS else 1,
    )
    api_port = st.number_input("API port", min_value=1024, max_value=65535, value=cfg.services.api_port)
    dash_port = st.number_input(
        "Dashboard port", min_value=1024, max_value=65535, value=cfg.services.dashboard_port
    )

with tab_paths:
    outputs = st.text_input("Outputs directory", cfg.paths.outputs)
    hf_cache = st.text_input("Hugging Face cache (HF_HOME)", cfg.paths.hf_cache)
    incoming = st.text_input("Incoming calls watch folder (optional)", cfg.paths.incoming_calls or "")

with tab_asr:
    asr_backend = st.selectbox(
        "ASR backend",
        ["faster", "transformers", "whisperx"],
        index=["faster", "transformers", "whisperx"].index(cfg.asr.backend),
    )
    asr_model = st.text_input("ASR model", cfg.asr.model)
    revision = st.selectbox(
        "Revision",
        ["standard", "strict", "subtitle"],
        index=["standard", "strict", "subtitle"].index(cfg.asr.revision),
    )
    preprocess = st.checkbox("Preprocess audio (ffmpeg + noisereduce)", cfg.asr.preprocess)

with tab_llm:
    st.warning(
        "External LLM calls send transcript data to OpenRouter/Mistral (third party). "
        "Enable only with appropriate consent and DPA."
    )
    llm_enabled = st.checkbox("Enable Mistral/OpenRouter LLM path", cfg.llm.enabled)
    llm_model = st.text_input("Default LLM model", cfg.llm.default_model)
    anonymize = st.checkbox("Anonymize before LLM (when implemented)", cfg.llm.anonymize_before_llm)
    budget = st.number_input(
        "Cost budget per call (USD)",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.llm.cost_budget_per_call),
    )

with tab_secrets:
    st.json(secret_status())
    or_key = st.text_input("OpenRouter API key", type="password", value="")
    if st.button("Save OpenRouter key to Credential Manager") and or_key:
        set_secret("openrouter", or_key)
        st.success("Saved")
    hf_key = st.text_input("Hugging Face token (pyannote)", type="password", value="")
    if st.button("Save HF token") and hf_key:
        set_secret("huggingface", hf_key)
        st.success("Saved")
    if get_secret("openrouter"):
        st.caption("OpenRouter key is configured (not shown).")

with tab_doctor:
    if st.button("Run health check"):
        report = run_preflight(cfg, require_openrouter=cfg.llm.enabled)
        for c in report.checks:
            icon = "✅" if c.ok else "❌"
            st.write(f"{icon} **{c.name}**: {c.message}")
            if c.detail:
                st.caption(c.detail)

if st.button("Save configuration", type="primary"):
    updated = UserConfig(
        install_profile=InstallProfile(profile),
        sentiment_profile=sentiment_profile,
        device=device,  # type: ignore[arg-type]
        portable_mode=portable,
        log_level=log_level,  # type: ignore[arg-type]
        paths=cfg.paths.model_copy(
            update={
                "app_root": str(_ROOT),
                "outputs": outputs,
                "hf_cache": hf_cache,
                "incoming_calls": incoming,
            }
        ),
        asr=cfg.asr.model_copy(
            update={
                "backend": asr_backend,  # type: ignore[arg-type]
                "model": asr_model,
                "revision": revision,  # type: ignore[arg-type]
                "preprocess": preprocess,
            }
        ),
        llm=cfg.llm.model_copy(
            update={
                "enabled": llm_enabled,
                "default_model": llm_model,
                "anonymize_before_llm": anonymize,
                "cost_budget_per_call": budget,
            }
        ),
        services=cfg.services.model_copy(
            update={"api_port": int(api_port), "dashboard_port": int(dash_port)}
        ),
    )
    path = save_user_config(updated)
    st.success(f"Saved to {path}")
    st.info(f"Default path: {default_user_config_path(portable, _ROOT)}")
