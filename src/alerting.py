    def notify_webhook(self, alert: Alert, url: str | None = None, call_id: str | None = None) -> dict:
        """Send webhook notification with retry, backoff, and circuit breaker.

        Uses AlertingState (in-memory or Redis) for failure tracking.
        """
        payload = self.build_webhook_payload(alert, call_id)
        if not url:
            logger.info("Alert webhook payload (no url configured): %s", payload)
            return payload

        # Check circuit breaker via state
        if self.state.is_circuit_breaker_open():
            logger.warning("Webhook circuit breaker OPEN – delivery skipped")
            return payload

        max_retries = self.config.get("webhook", {}).get("max_retries", 3)
        base_backoff = self.config.get("webhook", {}).get("retry_backoff_base", 1.0)
        timeout = self.config.get("webhook", {}).get("timeout_seconds", 10.0)

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                resp = httpx.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                if resp.status_code < 400:
                    logger.info(
                        "Webhook delivered (attempt %d/%d): %s -> %s",
                        attempt, max_retries, alert.rule_id, resp.status_code
                    )
                    self.state.reset()  # success → reset failures
                    return payload
                else:
                    logger.warning(
                        "Webhook HTTP %s (attempt %d): %s",
                        resp.status_code, attempt, resp.text[:200]
                    )
            except httpx.TimeoutException:
                logger.warning("Webhook timeout (attempt %d/%d)", attempt, max_retries)
            except Exception as exc:
                logger.warning("Webhook error (attempt %d/%d): %s", attempt, max_retries, exc)

            # Failure path
            failures = self.state.increment_failures()
            threshold = self.config.get("webhook", {}).get("circuit_breaker_threshold", 5)

            if failures >= threshold:
                self.state.set_circuit_breaker_open(True)
                logger.error(
                    "Webhook circuit breaker OPEN after %d consecutive failures",
                    failures
                )
                break

            if attempt < max_retries:
                sleep_s = base_backoff * (2 ** (attempt - 1))
                logger.debug("Backing off %.1fs before retry", sleep_s)
                time.sleep(sleep_s)

        return payload