"""Tests for transcription monitor presets."""

from __future__ import annotations

from app.nicegui_dashboard.services.transcription_presets import (
    DEFAULT_PRESET_ID,
    apply_default_preset,
    apply_preset,
    get_preset,
    is_recommended_preset,
    preset_options,
)
from app.nicegui_dashboard.services.transcription_service import (
    TranscriptionState,
    create_transcription_state,
)


def test_default_preset_is_callcenter_standard() -> None:
    assert DEFAULT_PRESET_ID == "callcenter_standard"
    preset = get_preset(DEFAULT_PRESET_ID)
    assert preset is not None
    assert preset.get("recommended") is True
    assert preset["api_strategy"] == "transcribe"
    assert preset["settings"]["revision"] == "strict"
    assert preset["settings"]["use_hotwords_file"] is True
    assert preset["settings"]["api_fallback_local"] is True


def test_preset_options_marks_recommended() -> None:
    options = preset_options()
    assert "★" in options[DEFAULT_PRESET_ID]
    assert is_recommended_preset(DEFAULT_PRESET_ID)


def test_get_preset_api_callcenter() -> None:
    preset = get_preset("api_callcenter")
    assert preset is not None
    assert preset["use_api"] is True
    assert preset["api_strategy"] == "batch_transcribe"
    assert preset["settings"]["use_hotwords_file"] is True


def test_apply_preset_updates_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.nicegui_dashboard.services.transcription_service.CACHE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.nicegui_dashboard.services.transcription_service.QUEUE_STATE_FILE",
        tmp_path / "transcription_queue.json",
    )
    state = TranscriptionState()
    assert apply_preset(state, "scan_inkrementell")
    assert state.api_strategy == "scan_process"
    assert state.status["use_api"] is True
    assert state.active_preset == "scan_inkrementell"
    assert state.scan_config["batch_size"] == 4


def test_apply_default_preset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.nicegui_dashboard.services.transcription_service.CACHE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.nicegui_dashboard.services.transcription_service.QUEUE_STATE_FILE",
        tmp_path / "transcription_queue.json",
    )
    state = TranscriptionState()
    assert apply_default_preset(state)
    assert state.active_preset == DEFAULT_PRESET_ID
    assert state.settings["revision"] == "strict"
    assert state.status["use_api"] is True
    assert state.api_strategy == "transcribe"


def test_apply_unknown_preset_returns_false() -> None:
    state = TranscriptionState()
    assert apply_preset(state, "does_not_exist") is False


def test_create_transcription_state_applies_default_on_fresh(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.nicegui_dashboard.services.transcription_service.CACHE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.nicegui_dashboard.services.transcription_service.QUEUE_STATE_FILE",
        tmp_path / "transcription_queue.json",
    )
    state = create_transcription_state()
    assert state.active_preset == DEFAULT_PRESET_ID
    assert state.settings.get("api_fallback_local") is True
