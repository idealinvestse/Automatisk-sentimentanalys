"""Tests for blending module."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.blending import LearnedBlender, get_blender


class TestLearnedBlender:
    def test_default_blend(self):
        blender = LearnedBlender(default_lexicon_weight=0.3)
        model = {"negativ": 0.1, "neutral": 0.3, "positiv": 0.6}
        lex_tuple = (0.8, 0.1, 0.1)
        result = blender.blend(model, lex_tuple)
        assert sum(result.values()) == pytest.approx(1.0)
        assert all(k in result for k in ["negativ", "neutral", "positiv"])

    def test_pure_model(self):
        blender = LearnedBlender(default_lexicon_weight=0.0)
        model = {"negativ": 0.1, "neutral": 0.3, "positiv": 0.6}
        lex_tuple = (0.8, 0.1, 0.1)
        result = blender.blend(model, lex_tuple)
        assert result == model

    def test_pure_lexicon(self):
        blender = LearnedBlender(default_lexicon_weight=1.0)
        model = {"negativ": 0.1, "neutral": 0.3, "positiv": 0.6}
        lex_tuple = (0.8, 0.1, 0.1)
        result = blender.blend(model, lex_tuple)
        assert result["negativ"] == pytest.approx(0.8)

    def test_fit_and_blend(self):
        blender = LearnedBlender(default_lexicon_weight=0.5)
        model_dists = [
            {"negativ": 0.1, "neutral": 0.3, "positiv": 0.6},
            {"negativ": 0.7, "neutral": 0.2, "positiv": 0.1},
            {"negativ": 0.2, "neutral": 0.6, "positiv": 0.2},
        ]
        lex_dists = [
            (0.8, 0.1, 0.1),
            (0.1, 0.1, 0.8),
            (0.2, 0.6, 0.2),
        ]
        labels = ["positiv", "negativ", "neutral"]
        weights = blender.fit(model_dists, lex_dists, labels, epochs=50)
        assert blender.is_fitted
        assert all(0.0 <= v <= 1.0 for v in weights.values())

    def test_save_load(self):
        blender = LearnedBlender(default_lexicon_weight=0.7)
        blender.weights = {"negativ": 0.3, "neutral": 0.5, "positiv": 0.4}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            path = f.name
        try:
            blender.save(path)
            loaded = LearnedBlender(weight_path=path)
            assert loaded.weights == blender.weights
        finally:
            os.unlink(path)

    def test_get_blender_singleton(self):
        b1 = get_blender()
        b2 = get_blender()
        assert b1 is b2
