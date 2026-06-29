from __future__ import annotations

import html
import re
from collections.abc import Iterable

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_USERNAME_RE = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"#[\wäöåÄÖÅ]+", re.UNICODE)


def clean_text(text: str, opts: dict) -> str:
    t = text or ""
    if opts.get("unescape_html", False):
        t = html.unescape(t)
    if opts.get("strip_html", False):
        t = _TAG_RE.sub(" ", t)
    if opts.get("remove_urls", False):
        t = _URL_RE.sub(" ", t)
    if opts.get("remove_usernames", False):
        t = _USERNAME_RE.sub(" ", t)
    if opts.get("remove_hashtags", False):
        # Remove '#' but keep the word
        t = _HASHTAG_RE.sub(lambda m: m.group(0).lstrip("#"), t)
    if opts.get("lowercase", False):
        t = t.lower()
    if opts.get("normalize_whitespace", False):
        t = " ".join(t.split())
    # New: basic emoji to sentiment word mapping (for social/forum/ASR)
    if opts.get("map_emojis", False):
        emoji_map = {
            "😊": " glad ",
            "😃": " glad ",
            "🙂": " glad ",
            ":)": " glad ",
            ":-)": " glad ",
            "😢": " ledsen ",
            "😭": " ledsen ",
            ":(": " ledsen ",
            "👍": " bra ",
            "👎": " dåligt ",
            "❤️": " älskar ",
            "💕": " älskar ",
            "🔥": " super ",
            "💯": " perfekt ",
        }
        for em, repl in emoji_map.items():
            t = t.replace(em, repl)
    # New: very basic ASR/typo normalizer for common Swedish spoken forms
    if opts.get("normalize_swedish", False):
        asr_fixes = {
            "ställde": "ställde",  # already correct
            "nöd": "nöjd",
            "nådde": "nådde",
            # add more as needed; lowercased match
        }
        lowered = t.lower()
        for bad, good in asr_fixes.items():
            if bad in lowered:
                t = t.replace(bad, good).replace(bad.capitalize(), good.capitalize())
    return t.strip()


def clean_texts(texts: Iterable[str], opts: dict) -> list[str]:
    return [clean_text(t, opts) for t in texts]
