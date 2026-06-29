"""Predefined transcription monitor presets (settings + strategy + paths)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_PRESET_ID = "callcenter_standard"

PRESET_IDS = (
    DEFAULT_PRESET_ID,
    "lokal_snabb",
    "api_standard",
    "api_callcenter",
    "scan_inkrementell",
    "scan_analys",
    "anpassad",
)

# Shared callcenter ASR tuning used by the default preset.
_CALLCENTER_ASR: dict[str, Any] = {
    "backend": "faster",
    "model": "kb-whisper-large",
    "device": "auto",
    "language": "sv",
    "preprocess": True,
    "diarize": False,
    "revision": "strict",
    "beam_size": 5,
    "vad": True,
    "chunk_length_s": 30,
    "word_timestamps": True,
    "use_hotwords_file": True,
    "hotwords_file": "configs/callcenter_hotwords.txt",
    "hotwords": "",
    "initial_prompt": "Svenskt callcenter-samtal mellan agent och kund.",
    "workers": 1,
    "worker_timeout": 600.0,
    "local_timeout_s": 900.0,
    "api_retries": 2,
    "api_fallback_local": True,
}

_PRESETS: dict[str, dict[str, Any]] = {
    DEFAULT_PRESET_ID: {
        "label": "Callcenter standard (rekommenderad)",
        "description": (
            "Standard för svenska callcenter: KB-Whisper strict, preprocess, hotwords från fil, "
            "VAD + chunking. API per fil med automatisk lokal fallback vid fel."
        ),
        "recommended": True,
        "use_api": True,
        "api_strategy": "transcribe",
        "pending_folder": "inputs/pending",
        "output_dir": "outputs/transcripts",
        "settings": dict(_CALLCENTER_ASR),
        "scan_config": {
            "operation": "transcribe",
            "batch_size": 4,
            "pattern": None,
            "max_files": None,
            "recursive": True,
        },
    },
    "lokal_snabb": {
        "label": "Lokal snabb",
        "description": "Lokal ASR med faster-whisper, preprocess på. Ingen backend-API.",
        "use_api": False,
        "api_strategy": "transcribe",
        "pending_folder": "inputs/pending",
        "output_dir": "outputs/transcripts",
        "settings": {
            "backend": "faster",
            "model": "kb-whisper-large",
            "device": "auto",
            "language": "sv",
            "preprocess": True,
            "diarize": False,
            "beam_size": 5,
            "vad": True,
            "chunk_length_s": 30,
            "word_timestamps": True,
            "workers": 1,
            "local_timeout_s": 600.0,
            "api_retries": 1,
        },
        "scan_config": {
            "operation": "transcribe",
            "batch_size": 4,
            "pattern": None,
            "max_files": None,
            "recursive": True,
        },
    },
    "api_standard": {
        "label": "API standard",
        "description": "Parallell batch via backend (workers=2), WebSocket-loggar.",
        "use_api": True,
        "api_strategy": "batch_transcribe",
        "pending_folder": "inputs/pending",
        "output_dir": "outputs/transcripts",
        "settings": {
            "backend": "faster",
            "model": "kb-whisper-large",
            "device": "auto",
            "language": "sv",
            "preprocess": True,
            "diarize": False,
            "beam_size": 5,
            "vad": True,
            "workers": 2,
            "worker_timeout": 300.0,
            "api_retries": 2,
        },
        "scan_config": {
            "operation": "transcribe",
            "batch_size": 4,
            "pattern": None,
            "max_files": None,
            "recursive": True,
        },
    },
    "api_callcenter": {
        "label": "API callcenter (batch)",
        "description": "Optimerad batch för svenska callcenter: strict revision, hotwords, preprocess.",
        "use_api": True,
        "api_strategy": "batch_transcribe",
        "pending_folder": "inputs/pending",
        "output_dir": "outputs/transcripts",
        "settings": {
            **dict(_CALLCENTER_ASR),
            "workers": 2,
            "worker_timeout": 600.0,
        },
        "scan_config": {
            "operation": "transcribe",
            "batch_size": 4,
            "pattern": "**/*.{wav,mp3,m4a}",
            "max_files": None,
            "recursive": True,
        },
    },
    "scan_inkrementell": {
        "label": "Scan inkrementell",
        "description": "Skannar väntemapp, hoppar över redan bearbetade filer (state_file).",
        "use_api": True,
        "api_strategy": "scan_process",
        "pending_folder": "inputs/pending",
        "output_dir": "outputs/transcripts",
        "settings": {
            "backend": "faster",
            "model": "kb-whisper-large",
            "device": "auto",
            "language": "sv",
            "preprocess": True,
            "diarize": False,
            "workers": 2,
            "worker_timeout": 300.0,
            "api_retries": 2,
        },
        "scan_config": {
            "operation": "transcribe",
            "batch_size": 4,
            "pattern": "**/*.{wav,mp3,m4a,flac,ogg}",
            "max_files": None,
            "recursive": True,
        },
    },
    "scan_analys": {
        "label": "Scan + analys",
        "description": "Inkrementell scan med analyze_conversation (transkript + sentiment).",
        "use_api": True,
        "api_strategy": "scan_process",
        "pending_folder": "inputs/pending",
        "output_dir": "outputs/transcripts",
        "settings": {
            "backend": "faster",
            "model": "kb-whisper-large",
            "device": "auto",
            "language": "sv",
            "preprocess": True,
            "diarize": False,
            "workers": 1,
            "worker_timeout": 600.0,
            "sentiment_profile": "callcenter",
            "api_retries": 2,
        },
        "scan_config": {
            "operation": "analyze_conversation",
            "batch_size": 2,
            "pattern": "**/*.{wav,mp3,m4a}",
            "max_files": 50,
            "recursive": True,
        },
    },
}


def preset_options() -> dict[str, str]:
    """Map preset id -> human label for ui.select."""
    out: dict[str, str] = {}
    for pid in PRESET_IDS:
        if pid not in _PRESETS:
            continue
        label = _PRESETS[pid]["label"]
        if _PRESETS[pid].get("recommended"):
            label = f"★ {label}"
        out[pid] = label
    return out


def preset_description(preset_id: str) -> str:
    if preset_id == "anpassad":
        return "Anpassad konfiguration (sparad lokalt)."
    return _PRESETS.get(preset_id, {}).get("description", "")


def is_recommended_preset(preset_id: str) -> bool:
    return bool(_PRESETS.get(preset_id, {}).get("recommended"))


def get_preset(preset_id: str) -> dict[str, Any] | None:
    if preset_id not in _PRESETS:
        return None
    return deepcopy(_PRESETS[preset_id])


def apply_preset(state: Any, preset_id: str) -> bool:
    """Apply preset fields onto TranscriptionState. Returns False if unknown preset."""
    preset = get_preset(preset_id)
    if preset is None:
        return False

    state.settings.clear()
    state.settings.update(_default_settings_merge(preset.get("settings", {})))
    state.api_strategy = preset.get("api_strategy", state.api_strategy)
    state.pending_folder = preset.get("pending_folder", state.pending_folder)
    state.output_dir = preset.get("output_dir", state.output_dir)
    state.scan_config.update(preset.get("scan_config", {}))
    state.status["use_api"] = bool(preset.get("use_api", False))
    state.active_preset = preset_id
    state.save()
    return True


def _default_settings_merge(preset_settings: dict[str, Any]) -> dict[str, Any]:
    """Ensure all known keys exist when applying a preset."""
    from app.nicegui_dashboard.services.transcription_service import _default_settings

    merged = _default_settings()
    merged.update(preset_settings)
    return merged


def apply_default_preset(state: Any) -> bool:
    """Apply the recommended default preset."""
    return apply_preset(state, DEFAULT_PRESET_ID)
