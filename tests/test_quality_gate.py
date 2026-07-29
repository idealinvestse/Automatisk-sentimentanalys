"""Tests for Quality OS preference-gate configuration."""

from __future__ import annotations

from pathlib import Path

from scripts.evaluate_preference_gate import _load_gate_config
from src.quality.mqm import PreferencePair, evaluate_preference_gate


def _pair(index: int, preferred: str = "a") -> PreferencePair:
    return PreferencePair(
        call_id=f"call-{index}",
        output_a_id="candidate",
        output_b_id="baseline",
        preferred=preferred,
    )


def test_quality_config_requires_ten_pairs() -> None:
    config = _load_gate_config(Path("configs/quality_mqm.yaml"))

    assert config["min_pairs"] == 10
    assert config["min_win_rate"] == 0.55


def test_preference_gate_fails_below_configured_minimum() -> None:
    result = evaluate_preference_gate([_pair(index) for index in range(9)], min_pairs=10)

    assert result.passed is False
    assert result.n_pairs == 9


def test_preference_gate_passes_with_configured_corpus() -> None:
    result = evaluate_preference_gate([_pair(index) for index in range(10)], min_pairs=10)

    assert result.passed is True
    assert result.win_rate_a == 1.0
