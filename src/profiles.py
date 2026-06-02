from __future__ import annotations

import os

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
            "kundtjänst_kvalitet", "teknisk_lösning", "fakturering_pris", "väntetid",
            "agent_attityd", "produkt_kvalitet", "uppföljning", "annat"
        ],
        # Task 3.2.3: Mistral/OpenRouter LLM config for holistisk analysis (European-first)
        "llm": {
            "enabled": True,  # callcenter gets the deep path by default (selective via length/confidence)
            "default_model": "mistralai/mistral-medium-3.5",
            "fallback_model": "mistralai/mistral-medium-3.5",
            "cost_budget_per_call": 0.08,
            "anonymize_before_llm": False,  # Fas 3.4 hook
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
