"""Tests for negation module."""

from __future__ import annotations

from src.negation import (
    apply_negation_heuristic,
    detect_negation,
    detect_negation_with_position,
    flip_polarity,
    is_negated_example,
    tokenize_lower,
)


class TestDetectNegation:
    def test_simple_negation(self):
        assert detect_negation("Jag är inte nöjd med servicen")
        assert detect_negation("Det här är aldrig bra")
        assert detect_negation("Jag har ej fått svar")

    def test_no_negation(self):
        assert not detect_negation("Jag är väldigt nöjd")
        assert not detect_negation("Allt fungerar perfekt")

    def test_multi_word_negation(self):
        assert detect_negation("Det är absolut inte okej")
        assert detect_negation("Jag kommer aldrig mer tillbaka")

    def test_negation_without_scope(self):
        # "inte" alone without following words may not trigger (depends on window)
        # Short text with no scope should return False
        result = detect_negation("inte", window=3)
        assert not result  # no scope words after negation


class TestDetectNegationWithPosition:
    def test_returns_positions(self):
        hits = detect_negation_with_position("Jag är inte nöjd")
        assert len(hits) >= 1
        assert hits[0][1] == "single"

    def test_multi_word_position(self):
        hits = detect_negation_with_position("Det är absolut inte bra")
        assert len(hits) >= 1
        assert hits[0][1] == "multi"


class TestFlipPolarity:
    def test_full_flip(self):
        assert flip_polarity(0.8, 1.0) == -0.8
        assert flip_polarity(-0.5, 1.0) == 0.5

    def test_partial_flip(self):
        result = flip_polarity(0.8, 0.5)
        assert result == -0.4


class TestApplyNegationHeuristic:
    def test_flips_positive_to_negative(self):
        dist = {"negativ": 0.1, "neutral": 0.2, "positiv": 0.7}
        result = apply_negation_heuristic("Jag är inte nöjd", dist)
        assert result["negativ"] > result["positiv"]

    def test_no_negation_unchanged(self):
        dist = {"negativ": 0.1, "neutral": 0.2, "positiv": 0.7}
        result = apply_negation_heuristic("Jag är nöjd", dist)
        assert result == dist


class TestIsNegatedExample:
    def test_negated(self):
        assert is_negated_example("Jag är inte nöjd")
        assert is_negated_example("Aldrig sett maken")

    def test_not_negated(self):
        assert not is_negated_example("Jag är nöjd")
        assert not is_negated_example("Allt bra")


class TestTokenizeLower:
    def test_basic(self):
        tokens = tokenize_lower("Hej Världen!")
        assert tokens == ["hej", "världen"]

    def test_punctuation(self):
        tokens = tokenize_lower("Inte, alls. bra!")
        assert tokens == ["inte", "alls", "bra"]


def test_intensity_multiplier():
    from src.negation import get_intensity_multiplier
    assert get_intensity_multiplier("Det var väldigt bra") > 1.0
    assert get_intensity_multiplier("Det var lite bra") < 1.0
    assert get_intensity_multiplier("Det var okej") == 1.0
