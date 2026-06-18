"""Predefined transcription monitor presets (settings + strategy + paths)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PRESET_IDS = (
    "lokal_snabb",
    "api_standard",
    "api_callcenter",
    "scan_inkrementell",
    "scan_analys",
    "anpassad",
)

_PRESETS: dict[str, dict[str, Any]] = {
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
            "workers": 1,
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
        "label": "API callcenter",
        "description": "Optimerad för svenska callcenter: strict revision, hotwords, preprocess.",
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
            "revision": "strict",
            "beam_size": 5,
            "vad": True,
            "hotwords": "kundtjänst, faktura, avtal, uppsägning, support",
            "initial_prompt": "Svenskt callcenter-samtal mellan agent och kund.",
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
    return {pid: _PRESETS[pid]["label"] for pid in PRESET_IDS if pid in _PRESETS}


def preset_description(preset_id: str) -> str:
    if preset_id == "anpassad":
        return "Anpassad konfiguration (sparad lokalt)."
    return _PRESETS.get(preset_id, {}).get("description", "")


def get_preset(preset_id: str) -> dict[str, Any] | None:
    if preset_id not in _PRESETS:
        return None
    return deepcopy(_PRESETS[preset_id])


def apply_preset(state: Any, preset_id: str) -> bool:
    """Apply preset fields onto TranscriptionState. Returns False if unknown preset."""
    preset = get_preset(preset_id)
    if preset is None:
        return False

    state.settings.update(preset.get("settings", {}))
    state.api_strategy = preset.get("api_strategy", state.api_strategy)
    state.pending_folder = preset.get("pending_folder", state.pending_folder)
    state.output_dir = preset.get("output_dir", state.output_dir)
    state.scan_config.update(preset.get("scan_config", {}))
    state.status["use_api"] = bool(preset.get("use_api", False))
    state.active_preset = preset_id
    state.save()
    return True