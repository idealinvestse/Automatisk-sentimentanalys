"""Tests for clean module."""
from __future__ import annotations

from src.clean import clean_text, clean_texts


class TestCleanText:
    def test_no_opts(self):
        assert clean_text("Hello world", {}) == "Hello world"

    def test_lowercase(self):
        assert clean_text("Hello World", {"lowercase": True}) == "hello world"

    def test_normalize_whitespace(self):
        assert clean_text("  Hello   world  ", {"normalize_whitespace": True}) == "Hello world"

    def test_remove_urls(self):
        opts = {"remove_urls": True}
        assert clean_text("Check https://example.com now", opts) == "Check   now"
        assert clean_text("Visit www.example.com today", opts) == "Visit   today"

    def test_strip_html(self):
        opts = {"strip_html": True}
        assert clean_text("<p>Hello</p>", opts) == "Hello"

    def test_unescape_html(self):
        opts = {"unescape_html": True}
        assert clean_text("&amp; &lt; &gt;", opts) == "& < >"

    def test_remove_usernames(self):
        opts = {"remove_usernames": True}
        assert clean_text("@user123 said hello", opts) == "said hello"

    def test_remove_hashtags(self):
        opts = {"remove_hashtags": True}
        assert clean_text("#hashtag #svenska", opts) == "hashtag svenska"

    def test_combined(self):
        opts = {
            "unescape_html": True,
            "remove_urls": True,
            "normalize_whitespace": True,
            "lowercase": True,
        }
        result = clean_text("  Check &amp; see https://example.com  ", opts)
        assert result == "check & see"

    def test_empty_input(self):
        assert clean_text("", {}) == ""
        assert clean_text(None, {}) == ""


class TestCleanTexts:
    def test_batch(self):
        opts = {"lowercase": True}
        result = clean_texts(["Hello", "World"], opts)
        assert result == ["hello", "world"]

    def test_empty_list(self):
        assert clean_texts([], {}) == []
