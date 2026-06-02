"""Tests for core serialization helpers (previously uncovered critical glue code)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from src.core.serialization import (
    map_results_to_segment_dicts,
    score_dict,
    segment_time,
    single_label_distribution,
    texts_from_segments,
    top_label,
    utc_now_iso,
)


class TestUtcNowIso:
    def test_format_and_z_suffix(self):
        ts = utc_now_iso(trim_microseconds=True)
        assert ts.endswith("Z")
        assert "T" in ts
        # Should be parseable and recent
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert (datetime.now(UTC) - dt).total_seconds() < 5

    def test_trim_microseconds(self):
        ts = utc_now_iso(trim_microseconds=True)
        assert "." not in ts

    def test_keep_microseconds(self):
        ts = utc_now_iso(trim_microseconds=False)
        # May or may not have . depending on clock, but should not raise
        assert "Z" in ts


class TestScoreDict:
    def test_single_dict(self):
        entry = {"label": "positiv", "score": 0.87}
        scores = score_dict(entry)
        assert scores == {"negativ": 0.0, "neutral": 0.0, "positiv": 0.87}

    def test_list_of_dicts_full(self):
        entries = [
            {"label": "negativ", "score": 0.1},
            {"label": "neutral", "score": 0.2},
            {"label": "positiv", "score": 0.7},
        ]
        scores = score_dict(entries)
        assert scores["positiv"] == pytest.approx(0.7)

    def test_missing_labels_default_zero(self):
        scores = score_dict([{"label": "positiv", "score": 0.95}])
        assert scores["negativ"] == 0.0
        assert scores["neutral"] == 0.0

    def test_invalid_label_ignored(self):
        scores = score_dict({"label": "foo", "score": 1.0})
        assert scores == {"negativ": 0.0, "neutral": 0.0, "positiv": 0.0}

    def test_non_dict_non_list_returns_zeros(self):
        assert score_dict("not a dict") == {"negativ": 0.0, "neutral": 0.0, "positiv": 0.0}
        assert score_dict(None) == {"negativ": 0.0, "neutral": 0.0, "positiv": 0.0}

    def test_none_score_treated_as_zero(self):
        scores = score_dict({"label": "neutral", "score": None})
        assert scores["neutral"] == 0.0


class TestSingleLabelDistribution:
    def test_from_full_scores(self):
        result = {"label": "negativ", "score": 0.9}
        dist = single_label_distribution(result)
        assert dist["negativ"] == pytest.approx(0.9)

    def test_from_label_only_gives_full_weight(self):
        result = {"label": "positiv"}
        dist = single_label_distribution(result)
        assert dist["positiv"] == 1.0
        assert dist["negativ"] == 0.0

    def test_unknown_label_falls_back(self):
        dist = single_label_distribution({"label": "weird"})
        assert dist == {"negativ": 0.0, "neutral": 0.0, "positiv": 0.0}


class TestTopLabel:
    def test_highest_wins(self):
        lbl, score = top_label({"negativ": 0.1, "neutral": 0.2, "positiv": 0.7})
        assert lbl == "positiv"
        assert score == pytest.approx(0.7)

    def test_empty_fallback(self):
        lbl, score = top_label({})
        assert lbl == "neutral"
        assert score == 0.0

    def test_tie_returns_some_max(self):
        lbl, _ = top_label({"negativ": 0.5, "neutral": 0.5, "positiv": 0.0})
        assert lbl in {"negativ", "neutral"}


class TestSegmentTime:
    def test_valid(self):
        assert segment_time({"start": 1.23}, "start") == 1.23
        assert segment_time({"end": 4}, "end") == 4.0

    def test_missing_or_invalid(self):
        assert segment_time({}, "start") is None
        assert segment_time({"start": "bad"}, "start") is None
        assert segment_time({"foo": 1}, "foo") == 1.0


class TestTextsFromSegments:
    def test_normal(self):
        segs = [{"text": " Hej "}, {"text": "där"}]
        assert texts_from_segments(segs) == ["Hej", "där"]

    def test_all_empty_fallback_joins(self):
        segs = [{"text": ""}, {"text": "   "}, {"text": "Hello world"}]
        # Since some non-empty, returns individuals (the non-empty ones? wait impl: if any(texts) return per
        # Actually: texts = [strip for], if texts and any(texts): return the list (incl empties)
        assert texts_from_segments(segs) == ["", "", "Hello world"]

    def test_all_blank_joins_to_one(self):
        segs = [{"text": ""}, {"text": "   "}]
        # Impl: if no truthy text, joined='', and 'if joined else []' => []
        assert texts_from_segments(segs) == []

    def test_empty_list(self):
        assert texts_from_segments([]) == []


class TestMapResultsToSegmentDicts:
    def test_basic_mapping(self):
        texts = ["Bra!", "Dåligt"]
        results = [
            [{"label": "positiv", "score": 0.9}, {"label": "neutral", "score": 0.1}, {"label": "negativ", "score": 0.0}],
            {"label": "negativ", "score": 0.8},
        ]
        segs = [{"start": 0, "end": 1, "text": "x"}, {"start": 2, "end": 3}]
        mapped = map_results_to_segment_dicts(texts, results, segs)
        assert len(mapped) == 2
        assert mapped[0]["label"] == "positiv"
        assert mapped[0]["negativ"] == 0.0
        assert mapped[0]["start"] == 0
        assert mapped[1]["label"] == "negativ"
        assert mapped[1]["score"] == pytest.approx(0.8)

    def test_length_mismatch_warns_but_continues(self, caplog):
        with caplog.at_level(logging.WARNING):
            mapped = map_results_to_segment_dicts(["a", "b"], [ {"label":"x"} ], [{"start": 0}])
        # zip strict=False produces min len items; log is emitted
        assert len(mapped) == 1
        assert "length mismatch" in caplog.text.lower()

    def test_missing_segments_uses_empty(self):
        texts = ["foo"]
        results = [{"label": "neutral", "score": 0.5}]
        mapped = map_results_to_segment_dicts(texts, results, [])
        assert mapped[0]["start"] is None
        assert mapped[0]["text"] == "foo"
