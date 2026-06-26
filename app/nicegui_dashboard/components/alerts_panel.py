        # Webhook / Circuit breaker status
        status = getattr(state, "alerting_status", None) or {}
        is_open = status.get("circuit_breaker_open", False)
        color = "negative" if is_open else "positive"
        status_text = "OPEN" if is_open else "CLOSED"

        with ui.row().classes("items-center gap-2 q-mb-sm"):
            ui.chip("Webhook", color="primary").props("dense")
            ui.chip(f"Circuit breaker: {status_text}", color=color).props("dense")
            if status:
                failures = status.get("consecutive_failures", 0)
                threshold = status.get("circuit_breaker_threshold", 5)
                ui.label(f"({failures}/{threshold} failures)").classes("text-caption text-grey")