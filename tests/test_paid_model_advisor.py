"""Tests for paid model advisor / analysis perspectives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm.paid_model_advisor import (
    ANALYSIS_PERSPECTIVES,
    ModelCandidate,
    collect_paid_candidates,
    list_analysis_profiles,
    recommend_for_perspective,
    score_candidate_for_perspective,
)


def test_all_perspectives_defined() -> None:
    assert len(ANALYSIS_PERSPECTIVES) >= 10
    for _pid, p in ANALYSIS_PERSPECTIVES.items():
        assert p["label"]
        assert 0 <= p["cost_priority"] <= 1
        assert 0 <= p["quality_priority"] <= 1
        assert p["max_usd_per_m_blended"] > 0


def test_score_prefers_cheaper_when_cost_priority_high() -> None:
    cheap = ModelCandidate(
        provider="openrouter",
        model_id="cheap-model",
        name="cheap",
        prompt_per_m_usd=0.05,
        completion_per_m_usd=0.1,
        blended_per_m_usd=0.08,
        context_length=32000,
        is_free=False,
        quality_score=0.5,
        swedish_score=0.5,
        eu_score=0.5,
    )
    dear = ModelCandidate(
        provider="openrouter",
        model_id="dear-model",
        name="dear",
        prompt_per_m_usd=10.0,
        completion_per_m_usd=30.0,
        blended_per_m_usd=22.0,
        context_length=128000,
        is_free=False,
        quality_score=0.95,
        swedish_score=0.5,
        eu_score=0.5,
    )
    persp = ANALYSIS_PERSPECTIVES["cost_saver"]
    s_cheap, _ = score_candidate_for_perspective(cheap, persp)
    s_dear, _ = score_candidate_for_perspective(dear, persp)
    assert s_cheap > s_dear


def test_score_prefers_quality_for_premium() -> None:
    cheap = ModelCandidate(
        provider="openrouter",
        model_id="mistralai/mistral-small",
        name="small",
        prompt_per_m_usd=0.1,
        completion_per_m_usd=0.2,
        blended_per_m_usd=0.16,
        context_length=32000,
        is_free=False,
        quality_score=0.45,
        swedish_score=0.8,
        eu_score=0.9,
    )
    dear = ModelCandidate(
        provider="openrouter",
        model_id="mistralai/mistral-large-2512",
        name="large",
        prompt_per_m_usd=0.5,
        completion_per_m_usd=1.5,
        blended_per_m_usd=1.1,
        context_length=128000,
        is_free=False,
        quality_score=0.9,
        swedish_score=0.9,
        eu_score=0.9,
    )
    persp = ANALYSIS_PERSPECTIVES["premium_reasoning"]
    s_cheap, _ = score_candidate_for_perspective(cheap, persp)
    s_dear, _ = score_candidate_for_perspective(dear, persp)
    assert s_dear > s_cheap


def test_list_profiles_returns_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Minimal fake openrouter catalog
    cat_dir = tmp_path / "cats"
    cat_dir.mkdir()
    models = [
        {
            "id": "mistralai/mistral-nemo",
            "name": "Mistral Nemo",
            "description": "Swedish capable",
            "context_length": 128000,
            "pricing": {"prompt_per_million_usd": 0.02, "completion_per_million_usd": 0.04},
            "is_free": False,
        },
        {
            "id": "mistralai/mistral-large-2512",
            "name": "Mistral Large",
            "description": "Strong reasoning",
            "context_length": 256000,
            "pricing": {"prompt_per_million_usd": 0.5, "completion_per_million_usd": 1.5},
            "is_free": False,
        },
        {
            "id": "free/model:free",
            "name": "Free",
            "pricing": {"prompt_per_million_usd": 0, "completion_per_million_usd": 0},
            "is_free": True,
        },
    ]
    (cat_dir / "openrouter.json").write_text(
        json.dumps({"models": models, "count": len(models)}), encoding="utf-8"
    )
    cfg = {
        "providers": {
            "openrouter": {"enabled": True},
            "mistral": {"enabled": False},
            "nvidia": {"enabled": False},
            "cerebras": {"enabled": False},
        },
        "catalog": {"dir": str(cat_dir)},
    }
    snap = list_analysis_profiles(top_k=2, config=cfg)
    assert snap["candidate_count"] == 2  # free excluded
    assert len(snap["profiles"]) == len(ANALYSIS_PERSPECTIVES)
    for p in snap["profiles"]:
        assert p["id"]
        assert p["label"]
        assert "selectable" in p
        # recommended may exist
        if p.get("recommended"):
            assert p["recommended"]["model_id"]
            assert p["selectable"]["llm_model"] == p["recommended"]["model_id"]


def test_recommend_coaching_has_selectable() -> None:
    cands = [
        ModelCandidate(
            provider="openrouter",
            model_id="mistralai/mistral-large-2512",
            name="large",
            prompt_per_m_usd=0.5,
            completion_per_m_usd=1.5,
            blended_per_m_usd=1.1,
            context_length=128000,
            is_free=False,
            quality_score=0.9,
            swedish_score=0.9,
            eu_score=0.9,
        )
    ]
    rec = recommend_for_perspective("coaching_qa", candidates=cands, top_k=1)
    pub = rec.to_public()
    assert pub["selectable"]["analysis_perspective"] == "coaching_qa"
    assert pub["selectable"]["llm_model"] == "mistralai/mistral-large-2512"
    assert pub["selectable"]["use_mistral_llm"] is True


def test_collect_skips_free(tmp_path: Path) -> None:
    cat = tmp_path / "openrouter.json"
    cat.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "x:free",
                        "is_free": True,
                        "pricing": {"prompt_per_million_usd": 0, "completion_per_million_usd": 0},
                    },
                    {
                        "id": "paid",
                        "is_free": False,
                        "pricing": {"prompt_per_million_usd": 1, "completion_per_million_usd": 2},
                        "context_length": 8000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "providers": {"openrouter": {}},
        "catalog": {"dir": str(tmp_path)},
    }
    cands = collect_paid_candidates(providers=["openrouter"], config=cfg)
    assert len(cands) == 1
    assert cands[0].model_id == "paid"
