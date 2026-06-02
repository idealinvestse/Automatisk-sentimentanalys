"""Tests for lexicon module."""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from src.lexicon import (
    blend_distributions,
    blend_results_with_lexicon,
    load_lexicon,
    scalar_to_dist,
    score_text,
    tokenize,
)

SAMPLE_CSV = """term,polarity
bra,0.8
fantastiskt,1.0
dålig,-0.8
sämsta,-0.9
okej,0.0
"""


@pytest.fixture
def sample_lexicon_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_CSV)
        path = f.name
    yield path
    os.unlink(path)


class TestLoadLexicon:
    def test_load_basic(self, sample_lexicon_path):
        lex = load_lexicon(sample_lexicon_path)
        assert len(lex) == 5
        assert lex["bra"] == 0.8
        assert lex["fantastiskt"] == 1.0
        assert lex["dålig"] == -0.8

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_lexicon("nonexistent.csv")

    def test_clips_scores(self, sample_lexicon_path):
        # Write a CSV with scores out of bounds
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("term,polarity\nbra,1.5\ndålig,-2.0\n")
            path = f.name
        try:
            lex = load_lexicon(path)
            assert lex["bra"] == 1.0
            assert lex["dålig"] == -1.0
        finally:
            os.unlink(path)


class TestTokenize:
    def test_basic(self):
        tokens = list(tokenize("Det här är bra!"))
        assert "det" in tokens
        assert "här" in tokens
        assert "är" in tokens
        assert "bra" in tokens

    def test_swedish_chars(self):
        tokens = list(tokenize("äpple öl åsna"))
        assert "äpple" in tokens
        assert "öl" in tokens
        assert "åsna" in tokens

    def test_empty(self):
        assert list(tokenize("")) == []
        assert list(tokenize("...")) == []


class TestScoreText:
    def test_positive(self, sample_lexicon_path):
        lex = load_lexicon(sample_lexicon_path)
        score = score_text("Det här är bra och fantastiskt!", lex)
        assert score > 0.5

    def test_negative(self, sample_lexicon_path):
        lex = load_lexicon(sample_lexicon_path)
        score = score_text("Det här är dåligt och sämsta!", lex)
        assert score < -0.5

    def test_neutral(self, sample_lexicon_path):
        lex = load_lexicon(sample_lexicon_path)
        score = score_text("Det här är okej", lex)
        assert score == 0.0

    def test_no_match(self, sample_lexicon_path):
        lex = load_lexicon(sample_lexicon_path)
        score = score_text("Inga matchande ord här", lex)
        assert score == 0.0

    def test_negation_flips_polarity(self, sample_lexicon_path):
        lex = load_lexicon(sample_lexicon_path)
        score = score_text("Det här är inte bra", lex)
        assert score < 0


class TestScalarToDist:
    def test_positive(self):
        neg, neu, pos = scalar_to_dist(0.8)
        assert pos > neg
        assert pos > neu

    def test_negative(self):
        neg, neu, pos = scalar_to_dist(-0.8)
        assert neg > pos
        assert neg > neu

    def test_neutral(self):
        neg, neu, pos = scalar_to_dist(0.0)
        assert neu > neg
        assert neu > pos

    def test_extreme(self):
        neg, neu, pos = scalar_to_dist(1.0)
        assert pos == pytest.approx(1.0)
        assert neg == pytest.approx(0.0)
        assert neu == pytest.approx(0.0)


class TestBlendDistributions:
    def test_pure_model(self):
        model = {"negativ": 0.1, "neutral": 0.3, "positiv": 0.6}
        lex_tuple = (0.8, 0.1, 0.1)  # strong negative from lexicon
        result = blend_distributions(model, lex_tuple, 0.0)
        assert result == model  # weight 0 = pure model

    def test_pure_lexicon(self):
        model = {"negativ": 0.1, "neutral": 0.3, "positiv": 0.6}
        lex_tuple = (0.8, 0.1, 0.1)
        result = blend_distributions(model, lex_tuple, 1.0)
        assert result["negativ"] == pytest.approx(0.8)
        assert result["neutral"] == pytest.approx(0.1)
        assert result["positiv"] == pytest.approx(0.1)

    def test_blended(self):
        model = {"negativ": 0.0, "neutral": 0.5, "positiv": 0.5}
        lex_tuple = (0.0, 0.5, 0.5)
        result = blend_distributions(model, lex_tuple, 0.5)
        assert result["negativ"] == pytest.approx(0.0)
        assert result["neutral"] == pytest.approx(0.5)
        assert result["positiv"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests for the high-level blend_results_with_lexicon (previously untested)
# ---------------------------------------------------------------------------


class TestBlendResultsWithLexicon:
    @pytest.fixture
    def sample_lex_path(self, tmp_path):
        p = tmp_path / "lex.csv"
        p.write_text("term,polarity\nbra,0.9\ndålig,-0.8\n", encoding="utf-8")
        return str(p)

    def test_no_lexicon_returns_original(self, sample_lex_path):
        texts = ["Bra service"]
        results = [{"label": "positiv", "score": 0.8}]
        out = blend_results_with_lexicon(texts, results, None, 0.3)
        assert out is results  # same object when skipped

    def test_zero_weight_skips(self, sample_lex_path):
        texts = ["Bra"]
        results = [{"label": "positiv", "score": 0.9}]
        out = blend_results_with_lexicon(texts, results, sample_lex_path, 0.0)
        assert out == results

    def test_full_distribution_blending(self, sample_lex_path):
        texts = ["Det här är bra"]
        results = [
            [
                {"label": "negativ", "score": 0.1},
                {"label": "neutral", "score": 0.2},
                {"label": "positiv", "score": 0.7},
            ]
        ]
        out = blend_results_with_lexicon(texts, results, sample_lex_path, 0.5)
        assert isinstance(out[0], list)
        # lexicon "bra" positive should pull positiv up
        pos = next(x for x in out[0] if x["label"] == "positiv")["score"]
        assert pos > 0.7

    def test_single_label_input_produces_single_label_output(self, sample_lex_path):
        texts = ["Det här är dålig"]
        results = [{"label": "negativ", "score": 0.75}]
        out = blend_results_with_lexicon(texts, results, sample_lex_path, 0.4)
        assert isinstance(out[0], dict)
        assert out[0]["label"] in {"negativ", "neutral", "positiv"}

    def test_length_mismatch_does_not_crash(self, sample_lex_path, caplog):
        texts = ["ett", "två"]
        results = [{"label": "positiv", "score": 0.9}]
        with caplog.at_level(logging.WARNING):
            out = blend_results_with_lexicon(texts, results, sample_lex_path, 0.2)
        assert len(out) == 1
        assert "length mismatch" in caplog.text.lower()

    def test_bad_lexicon_file_returns_original_gracefully(self, caplog):
        texts = ["Bra"]
        results = [{"label": "positiv", "score": 0.9}]
        with caplog.at_level(logging.WARNING):
            out = blend_results_with_lexicon(texts, results, "/non/existent/lex.csv", 0.3)
        assert out == results
        assert "lexicon blending failed" in caplog.text.lower()
