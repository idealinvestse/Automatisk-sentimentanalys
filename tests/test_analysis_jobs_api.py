"""HTTP contract for POST /analysis/jobs (PII gate + queue)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app, raise_server_exceptions=False)

_SEGMENTS = [{"text": "Mitt personnummer är 19850101-1234", "start": 0, "end": 1}]


def test_analysis_job_rejects_when_redaction_raises() -> None:
    with patch(
        "src.llm.pii_redactor.redact_segments",
        side_effect=RuntimeError("redaction engine crashed"),
    ):
        response = client.post(
            "/analysis/jobs", json={"segments": _SEGMENTS, "profile": "complaint"}
        )
    assert response.status_code == 422
    assert "redaction" in response.text.lower()


def test_analysis_job_force_redacts_before_queue() -> None:
    fake_job = MagicMock()
    fake_job.to_dict.return_value = {
        "job_id": "abc123",
        "status": "queued",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "phase": "queued",
        "error_code": None,
        "result_available": False,
        "cancel_requested": False,
        "meta": {},
    }
    with patch.object(app.state.analysis_jobs, "submit", return_value=fake_job) as submit:
        response = client.post(
            "/analysis/jobs",
            json={"segments": _SEGMENTS, "profile": "default"},
        )
    assert response.status_code == 202
    payload = submit.call_args.args[0]
    text = payload["segments"][0]["text"]
    assert "19850101-1234" not in text
    assert "[REDACTED_PNR]" in text
