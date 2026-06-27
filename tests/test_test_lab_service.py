"""Unit tests for Testlabb service helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.nicegui_dashboard.services.test_lab_service import (
    list_audio_reports,
    list_sample_rows,
    load_examples_txt,
    resolve_api_audio_path,
    run_scenario_ui,
)
from tests.fixtures.ravdess_catalog import build_mini_ravdess_catalog, full_ravdess_available


@pytest.fixture
def ravdess_catalog_root(tmp_path, monkeypatch):
    if full_ravdess_available():
        return None
    mini = build_mini_ravdess_catalog(tmp_path / "audio_mini")
    monkeypatch.setattr(
        "src.benchmarks.audio_catalog.default_audio_root",
        lambda start=None: mini,
    )
    return mini


class TestListSampleRows:
    def test_returns_ravdess_rows_with_emotion(self, ravdess_catalog_root):
        rows = list_sample_rows(pack_id="ravdess_en", limit=5)
        assert len(rows) == 5
        assert rows[0]["pack"] == "ravdess_en"
        assert rows[0]["emotion"] != "—"
        assert "abs_path" in rows[0]

    def test_search_filters_paths(self, ravdess_catalog_root):
        rows = list_sample_rows(pack_id="ravdess_en", search="Actor_01", limit=10)
        assert rows
        assert all("Actor_01" in r["path"] for r in rows)


class TestResolveApiAudioPath:
    def test_relative_under_media_root(self, tmp_path):
        media = tmp_path / "repo"
        audio = media / "samples" / "audio"
        audio.mkdir(parents=True)
        wav = audio / "test.wav"
        wav.write_bytes(b"")
        api_path, warning = resolve_api_audio_path(str(wav), media_root=str(media))
        assert api_path == "samples/audio/test.wav"
        assert warning is None

    def test_outside_media_root_warns(self, tmp_path):
        media = tmp_path / "media"
        media.mkdir()
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"")
        api_path, warning = resolve_api_audio_path(str(outside), media_root=str(media))
        assert warning is not None
        assert "API_MEDIA_ROOT" in warning


class TestListAudioReports:
    def test_finds_json_in_tmp(self, tmp_path, monkeypatch):
        reports = tmp_path / "reports"
        reports.mkdir()
        payload = {"scenario": "smoke", "n_files": 3, "summary": {"n_success": 3}}
        (reports / "audio_smoke_test.json").write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(
            "app.nicegui_dashboard.services.test_lab_service.reports_dir",
            lambda: reports,
        )
        rows = list_audio_reports(limit=5)
        assert len(rows) == 1
        assert rows[0]["scenario"] == "smoke"
        assert rows[0]["files"] == 3


class TestExamplesAndDryRun:
    def test_load_examples_txt(self):
        examples = load_examples_txt()
        assert len(examples) >= 3
        assert all(isinstance(e, str) for e in examples)

    def test_scenario_dry_run_catalog(self, ravdess_catalog_root):
        report = run_scenario_ui("catalog", pack_ids=["ravdess_en"], limit=2, dry_run=True)
        assert report["scenario"] == "catalog"
        assert report["n_files"] == 2
        assert report["dry_run"] is True