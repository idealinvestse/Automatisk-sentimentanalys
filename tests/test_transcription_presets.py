"""Tests for transcription monitor presets."""

from __future__ import annotations

from app.nicegui_dashboard.services.transcription_presets import apply_preset, get_preset
from app.nicegui_dashboard.services.transcription_service import TranscriptionState


def test_get_preset_api_callcenter() -> None:
    preset = get_preset("api_callcenter")
    assert preset is not None
    assert preset["use_api"] is True
    assert preset["api_strategy"] == "batch_transcribe"
    assert "hotwords" in preset["settings"]


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


def test_apply_unknown_preset_returns_false() -> None:
    state = TranscriptionState()
    assert apply_preset(state, "does_not_exist") is False