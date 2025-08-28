from __future__ import annotations

import html
import re
from typing import Dict, Iterable, List

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_USERNAME_RE = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"#[\wäöåÄÖÅ]+", re.UNICODE)


def clean_text(text: str, opts: Dict) -> str:
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
        t = _HASHTAG_RE.sub(lambda m: m.group(0).lstrip('#'), t)
    if opts.get("lowercase", False):
        t = t.lower()
    if opts.get("normalize_whitespace", False):
        t = " ".join(t.split())
    return t.strip()


def clean_texts(texts: Iterable[str], opts: Dict) -> List[str]:
    return [clean_text(t, opts) for t in texts]
