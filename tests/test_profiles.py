"""Tests for profiles module."""

from __future__ import annotations

from src.profiles import AVAILABLE_PROFILES, PROFILE_SPECS, resolve_profile


class TestResolveProfile:
    def test_default(self):
        name, spec = resolve_profile()
        assert name == "default"
        assert "model" in spec
        assert "cleaning" in spec

    def test_explicit_profile(self):
        name, spec = resolve_profile(profile="forum")
        assert name == "forum"
        assert spec["cleaning"]["strip_html"] is True

    def test_unknown_profile_falls_back(self):
        name, spec = resolve_profile(profile="nonexistent")
        assert name == "default"

    def test_by_source(self):
        name, spec = resolve_profile(source="reddit")
        assert name == "forum"

    def test_by_datatype(self):
        name, spec = resolve_profile(datatype="call")
        assert name == "call"

    def test_call_profile(self):
        name, spec = resolve_profile(profile="call")
        assert name == "call"
        assert spec["cleaning"]["strip_html"] is False
        assert spec["cleaning"]["normalize_whitespace"] is True


class TestProfilesConfig:
    def test_all_profiles_have_required_keys(self):
        for name, spec in PROFILE_SPECS.items():
            assert "model" in spec, f"{name} missing model"
            assert "max_length" in spec, f"{name} missing max_length"
            assert "cleaning" in spec, f"{name} missing cleaning"

    def test_available_profiles(self):
        assert "default" in AVAILABLE_PROFILES
        assert "forum" in AVAILABLE_PROFILES
        assert "call" in AVAILABLE_PROFILES
        assert "callcenter" in AVAILABLE_PROFILES
        assert len(AVAILABLE_PROFILES) >= 7
