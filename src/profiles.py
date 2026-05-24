from __future__ import annotations

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

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
        },
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
        },
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
}

DATATYPE_TO_PROFILE = {
    "article": "news",
    "story": "news",
    "post": "forum",
    "comment": "forum",
    "review": "review",
    "call": "call",
    "transcript": "call",
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
