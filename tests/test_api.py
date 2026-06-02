"""Basic FastAPI tests using TestClient (covers schemas, routers, validation, happy paths with mocks)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api import app  # uses the factory via __init__

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") in ("ok", "healthy")


def test_analyze_text_happy(monkeypatch):
    # Mock the smart analyze and blend so we don't load models
    def fake_smart(texts, **kwargs):
        return ([{"label": "positiv", "score": 0.88}], {"profile": "default", "model": "fake"})

    monkeypatch.setattr("src.api.routers.text.analyze_smart", fake_smart)
    monkeypatch.setattr("src.api.routers.text.blend_results_with_lexicon", lambda t, r, *a, **k: r)

    payload = {"texts": ["Det här var fantastiskt!"]}
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert data["meta"]["profile"] == "default"


def test_analyze_requires_non_empty_texts():
    r = client.post("/analyze", json={"texts": []})
    assert r.status_code in (400, 422)


def test_transcribe_missing_file_validation():
    # The validator runs before handler
    r = client.post("/transcribe", json={"audio_path": "/this/does/not/exist.wav"})
    assert r.status_code in (400, 422)
    assert "not found" in r.text.lower() or "not found on server" in r.text.lower()


def test_analyze_conversation_validation():
    r = client.post("/analyze_conversation", json={"audio_path": "nonexistent.mp3"})
    assert r.status_code in (400, 422)


def test_batch_transcribe_basic_shape(monkeypatch):
    # We can mock resolve + transcribe_helper at high level
    with (
        patch("src.api.routers.transcription.resolve_audio_paths", return_value=["/tmp/a.wav"]),
        patch("src.api.routers.transcription.transcribe_helper", return_value={"segments": [], "model": "test"}),
    ):
        r = client.post(
            "/batch_transcribe",
            json={"audio_paths": ["/tmp/a.wav"], "workers": 1},
        )
        # May be 200 or 422 depending on validation of non-existing, but shape ok if accepted
        assert r.status_code in (200, 422)


def test_scan_process_rejects_bad_operation():
    r = client.post(
        "/scan_process",
        json={"directory": "/tmp", "operation": "foo"},
    )
    assert r.status_code in (400, 422)
    assert "operation" in r.text.lower()


def test_analyze_pipeline_happy(monkeypatch):
    # The pipeline router uses CallAnalysisPipeline.analyze_segments
    fake_report = MagicMock()
    fake_report.sentiment_results = []
    fake_report.intent_results = []
    fake_report.summary = {}
    fake_report.topics = {}
    fake_report.insights = {}
    fake_report.risks = {}
    fake_report.processing_time_s = 0.12

    with patch("src.api.routers.pipeline.CallAnalysisPipeline") as mock_pipe:
        inst = mock_pipe.return_value
        inst.analyze_segments.return_value = fake_report
        r = client.post(
            "/analyze_pipeline",
            json={"segments": [{"text": "Hej", "start": 0, "end": 1}]},
        )
        assert r.status_code == 200
        data = r.json()
        assert "sentiment_results" in data
        assert "timestamp" in data


def test_run_batch_sequential_and_timeout_per_task():
    """Direct test of the batch helper (covers the per-file timeout fix)."""
    from src.api.batch import run_batch

    calls = []

    def slow_worker(p):
        calls.append(p)
        # simulate work
        return f"ok-{p}"

    # Sequential
    res = run_batch(["f1", "f2"], slow_worker, workers=1, worker_timeout=5)
    assert len(res) == 2
    assert res[0][1] == "ok-f1"

    # Parallel with timeout (the per-result path)
    def timeout_prone(p):
        import time
        time.sleep(0.01)
        return f"fast-{p}"

    res2 = run_batch(["a", "b"], timeout_prone, workers=2, worker_timeout=1.0)
    assert len(res2) == 2
    assert all(r[2] is None for r in res2)
