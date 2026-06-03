"""Tests for secret status reporting (no value leakage)."""

from __future__ import annotations

from src.install.secrets_win import secret_status


def test_secret_status_shape(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    status = secret_status()
    assert "openrouter" in status
    assert "huggingface" in status
    for entry in status.values():
        assert "configured" in entry
        assert "source" in entry
        assert "preview" not in entry
        assert set(entry.keys()) == {"configured", "source"}
