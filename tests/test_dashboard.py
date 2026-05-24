"""Tests for dashboard helper functions."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dashboard import load_report, load_sample_data


class TestLoadReport:
    def test_load_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
            path = f.name

        try:
            result = load_report(path)
            assert result == {"key": "value"}
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        result = load_report("/nonexistent/path/report.json")
        assert result == {}


class TestLoadSampleData:
    def test_load_sample_data_structure(self):
        data = load_sample_data()
        assert "calls" in data
        assert "agents" in data
        assert len(data["calls"]) == 20
        assert len(data["agents"]) == 5

        # Check call structure
        call = data["calls"][0]
        assert "id" in call
        assert "timestamp" in call
        assert "overall_sentiment" in call
        assert call["overall_sentiment"] in ("positiv", "neutral", "negativ")

        # Check agent structure
        agent = data["agents"][0]
        assert "name" in agent
        assert "calls" in agent
        assert "avg_resolution_rate" in agent

    def test_load_sample_data_deterministic(self):
        """Verify local RNG gives deterministic output."""
        data1 = load_sample_data()
        data2 = load_sample_data()

        assert data1["calls"][0]["id"] == data2["calls"][0]["id"]
        assert data1["calls"][0]["duration_s"] == data2["calls"][0]["duration_s"]
        assert data1["agents"][0]["calls"] == data2["agents"][0]["calls"]
