"""Tests for alerting webhook (retry, circuit breaker, config)."""

from unittest.mock import MagicMock, patch

import pytest

from src.alerting import Alert, AlertEngine, EvidenceSpan, load_alerting_config


@pytest.fixture
def sample_alert():
    return Alert(
        rule_id="high_escalation_risk",
        severity="high",
        message="Test alert",
        evidence_spans=[EvidenceSpan(text="test=1")],
        triggered_values={"customer_sentiment": -0.8},
        recommended_actions=["flag_supervisor"],
        source="per_call",
    )


class TestWebhookDelivery:
    """HTTP webhook delivery with retry and circuit breaker."""

    def test_successful_post(self, sample_alert):
        """Successful POST returns payload and resets failure counter."""
        engine = AlertEngine()
        engine._consecutive_failures = 2
        with patch("src.alerting.httpx.post") as mock_post:
            mock_resp = MagicMock(status_code=200)
            mock_post.return_value = mock_resp
            payload = engine.notify_webhook(sample_alert, url="https://example.com/webhook")
            assert payload["rule_id"] == "high_escalation_risk"
            assert engine._consecutive_failures == 0
            mock_post.assert_called_once()

    def test_retry_on_timeout(self, sample_alert):
        """Retries 3 times on TimeoutException, then gives up."""
        engine = AlertEngine()
        with patch("src.alerting.httpx.post") as mock_post:
            mock_post.side_effect = Exception("timeout")
            payload = engine.notify_webhook(sample_alert, url="https://example.com/webhook")
            assert mock_post.call_count == 3
            assert payload is not None

    def test_circuit_breaker_triggers_after_5_failures(self, sample_alert):
        """After 5 consecutive failures, webhook is disabled."""
        engine = AlertEngine()
        engine._consecutive_failures = 4
        with patch("src.alerting.httpx.post") as mock_post:
            mock_post.side_effect = Exception("fail")
            engine.notify_webhook(sample_alert, url="https://example.com/webhook")
            assert getattr(engine, "_webhook_disabled", False) is True

    def test_noop_when_url_missing(self, sample_alert):
        """No HTTP call when url is None or empty."""
        engine = AlertEngine()
        with patch("src.alerting.httpx.post") as mock_post:
            payload = engine.notify_webhook(sample_alert, url=None)
            mock_post.assert_not_called()
            assert "rule_id" in payload

    def test_disabled_webhook_skips_delivery(self, sample_alert):
        """If circuit breaker open, delivery is skipped with warning log."""
        engine = AlertEngine()
        engine._webhook_disabled = True
        with patch("src.alerting.httpx.post") as mock_post:
            engine.notify_webhook(sample_alert, url="https://example.com/webhook")
            mock_post.assert_not_called()


class TestConfigLoading:
    """YAML + env override config loading."""

    def test_defaults_when_file_missing(self, tmp_path, monkeypatch):
        """Missing YAML falls back to env + hardcoded defaults."""
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://from-env.example.com")
        cfg = load_alerting_config(tmp_path / "nonexistent.yaml")
        assert cfg["webhook"]["url"] == "https://from-env.example.com"
        assert cfg["webhook"]["timeout_seconds"] == 10

    def test_env_override_priority(self, tmp_path, monkeypatch):
        """Env vars override YAML values."""
        cfg_file = tmp_path / "alerting_config.yaml"
        cfg_file.write_text("webhook:\n  url: https://from-yaml.example.com\n  timeout_seconds: 5\n")
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://from-env.example.com")
        cfg = load_alerting_config(cfg_file)
        assert cfg["webhook"]["url"] == "https://from-env.example.com"
        assert cfg["webhook"]["timeout_seconds"] == 5  # env not set, YAML wins for this key

    def test_graceful_on_bad_yaml(self, tmp_path, monkeypatch):
        """Corrupt YAML does not crash; returns defaults."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("webhook: [this is not valid")
        cfg = load_alerting_config(bad)
        assert "webhook" in cfg
        assert cfg["webhook"]["enabled"] is True
