from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def _get_callcenter_model() -> str:
    lora = "models/callcenter-sentiment-lora"
    return lora if os.path.isdir(lora) else DEFAULT_MODEL


# Minimal, pragmatic profile specifications
PROFILE_SPECS: dict[str, dict] = {
    "default": {
        "model": DEFAULT_MODEL,
        "max_length": 256,
        "cleaning": {
            "unescape_html": True,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
        },
        # LLM disabled for generic profiles (hybrid only on explicit flag or callcenter)
        "llm": {
            "enabled": False,
            "default_model": "mistralai/mistral-medium-3.5",
            "cost_budget_per_call": 0.05,
        },
    },
    "forum": {
        "model": DEFAULT_MODEL,
        "max_length": 256,
        "cleaning": {
            "unescape_html": True,
            "strip_html": True,
            "remove_urls": True,
            "remove_usernames": True,
            "remove_hashtags": True,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.3,
    },
    "magazine": {
        "model": DEFAULT_MODEL,
        "max_length": 512,
        "cleaning": {
            "unescape_html": True,
            "strip_html": True,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
        },
    },
    "news": {
        "model": DEFAULT_MODEL,
        "max_length": 512,
        "cleaning": {
            "unescape_html": True,
            "strip_html": True,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
        },
    },
    "social": {
        "model": DEFAULT_MODEL,
        "max_length": 256,
        "cleaning": {
            "unescape_html": True,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": True,
            "remove_hashtags": True,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.3,
    },
    "review": {
        "model": DEFAULT_MODEL,
        "max_length": 256,
        "cleaning": {
            "unescape_html": True,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
        },
    },
    "call": {
        "model": DEFAULT_MODEL,
        "max_length": 256,
        "cleaning": {
            # ASR text: no HTML, but keep punctuation restoration outside cleaner
            "unescape_html": False,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.25,
    },
    "callcenter": {
        "model": _get_callcenter_model(),
        "max_length": 384,
        "cleaning": {
            "unescape_html": False,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.25,
        # Task 1.5: aspects enabled by default via AspectAnalyzer registration
        "aspects": [
            "kundtjänst_kvalitet",
            "teknisk_lösning",
            "fakturering_pris",
            "väntetid",
            "agent_attityd",
            "produkt_kvalitet",
            "uppföljning",
            "annat",
        ],
        # Task 3.2.3: Mistral/OpenRouter LLM config for holistisk analysis (European-first)
        "llm": {
            "enabled": True,  # callcenter gets the deep path by default (selective via length/confidence)
            "default_model": "mistralai/mistral-medium-3.5",
            "fallback_model": "mistralai/mistral-medium-3.5",
            "routing_tier": "balanced",
            "cost_budget_per_call": 0.08,
            "anonymize_before_llm": False,  # Fas 3.4 hook
        },
    },
    # Analyzer-profile aliases (sentiment/cleaning similar to callcenter; analyzer sets from YAML)
    "sales": {
        "model": _get_callcenter_model(),
        "max_length": 384,
        "cleaning": {
            "unescape_html": False,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.25,
        "llm": {
            "enabled": False,
            "default_model": "mistralai/mistral-medium-3.5",
            "cost_budget_per_call": 0.05,
        },
    },
    "complaint": {
        "model": _get_callcenter_model(),
        "max_length": 384,
        "cleaning": {
            "unescape_html": False,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.25,
        "llm": {
            "enabled": True,
            "default_model": "mistralai/mistral-medium-3.5",
            "cost_budget_per_call": 0.08,
        },
    },
    "support": {
        "model": _get_callcenter_model(),
        "max_length": 384,
        "cleaning": {
            "unescape_html": False,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.25,
        "llm": {
            "enabled": True,
            "default_model": "mistralai/mistral-medium-3.5",
            "cost_budget_per_call": 0.06,
        },
    },
    "teknisk_support": {
        "model": DEFAULT_MODEL,
        "max_length": 384,
        "cleaning": {
            "unescape_html": False,
            "strip_html": False,
            "remove_urls": True,
            "remove_usernames": False,
            "remove_hashtags": False,
            "normalize_whitespace": True,
            "lowercase": False,
            "map_emojis": True,
            "normalize_swedish": True,
        },
        "lexicon_file": "data/sensaldo_lexicon.csv",
        "lexicon_weight": 0.2,
        "llm": {
            "enabled": False,
            "default_model": "mistralai/mistral-medium-3.5",
            "cost_budget_per_call": 0.05,
        },
    },
}

AVAILABLE_PROFILES = sorted(PROFILE_SPECS.keys())

SOURCE_TO_PROFILE = {
    "forum": "forum",
    "flashback": "forum",
    "reddit": "forum",
    "magazine": "magazine",
    "news": "news",
    "newspaper": "news",
    "blog": "social",
    "social": "social",
    "twitter": "social",
    "x": "social",
    "call": "call",
    "phone": "call",
    "telephony": "call",
    "callcenter": "callcenter",
    "customer_service": "callcenter",
    "sales": "sales",
    "complaint": "complaint",
    "support": "support",
    "teknisk_support": "teknisk_support",
}

DATATYPE_TO_PROFILE = {
    "article": "news",
    "story": "news",
    "post": "forum",
    "comment": "forum",
    "review": "review",
    "call": "call",
    "transcript": "call",
    "callcenter": "callcenter",
}


def resolve_profile(
    datatype: str | None = None,
    source: str | None = None,
    profile: str | None = None,
) -> tuple[str, dict]:
    """Resolve a profile name and spec based on explicit profile, source, or datatype."""
    if profile:
        name = profile.strip().lower()
        return (
            name if name in PROFILE_SPECS else "default",
            PROFILE_SPECS.get(name, PROFILE_SPECS["default"]),
        )

    if source:
        s = source.strip().lower()
        mapped = SOURCE_TO_PROFILE.get(s)
        if mapped:
            return mapped, PROFILE_SPECS[mapped]

    if datatype:
        d = datatype.strip().lower()
        mapped = DATATYPE_TO_PROFILE.get(d)
        if mapped:
            return mapped, PROFILE_SPECS[mapped]

    return "default", PROFILE_SPECS["default"]


_ANALYZER_PROFILES: dict[str, dict[str, Any]] = {}
_ANALYZER_PROFILES_LOADED = False


def _configs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def _load_analyzer_profiles() -> None:
    global _ANALYZER_PROFILES_LOADED
    if _ANALYZER_PROFILES_LOADED:
        return
    path = _configs_dir() / "analyzer_profiles.yaml"
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                _ANALYZER_PROFILES.update(data)
        except Exception as exc:
            logger.warning("Failed to load analyzer_profiles.yaml: %s", exc)
    _ANALYZER_PROFILES_LOADED = True


def get_analyzer_profile_spec(profile_name: str) -> dict[str, Any]:
    """Return merged profile spec including analyzer activation config from YAML."""
    _load_analyzer_profiles()
    name = (profile_name or "default").strip().lower()
    _, base_spec = resolve_profile(profile=name)
    spec = dict(base_spec)
    yaml_cfg = _ANALYZER_PROFILES.get(name) or _ANALYZER_PROFILES.get("default") or {}
    if yaml_cfg:
        spec["analyzers"] = yaml_cfg
    return spec
