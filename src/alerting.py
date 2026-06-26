    def get_webhook_status(self) -> dict[str, Any]:
        """Return current webhook / circuit breaker status.

        Useful for dashboard to show real-time health.
        """
        return {
            "enabled": self.config.get("webhook", {}).get("enabled", True),
            "url_configured": bool(self.config.get("webhook", {}).get("url")),
            "circuit_breaker_open": getattr(self, "_webhook_disabled", False),
            "consecutive_failures": getattr(self, "_consecutive_failures", 0),
            "max_retries": self.config.get("webhook", {}).get("max_retries", 3),
            "circuit_breaker_threshold": self.config.get("webhook", {}).get(
                "circuit_breaker_threshold", 5
            ),
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (useful for testing and ops)."""
        self._webhook_disabled = False
        self._consecutive_failures = 0
        logger.info("Webhook circuit breaker manually reset")