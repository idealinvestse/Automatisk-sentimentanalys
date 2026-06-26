    @ui.refreshable
    async def alerts_section() -> None:
        reports = filter_reports(state.reports, state.filters)

        # Try to fetch fresh alerting status if we have an API client
        if state.api_client and not getattr(state, "alerting_status", None):
            try:
                status = await state.api_client.get_alerting_status()
                state.alerting_status = status.get("webhook", {})
            except Exception:
                state.alerting_status = {"circuit_breaker_open": False}  # fallback

        active = active_alerts(reports, state.dismissed_alert_keys)

        title = "🚨 Alerts & Actions" if not compact else "Alerts"
        ui.label(title).classes("text-subtitle1 q-mb-sm" if not compact else "text-subtitle2")

        # Dynamic webhook / circuit breaker status
        status = getattr(state, "alerting_status", {}) or {}
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

        if not active:
            ui.label("Inga aktiva alerts just nu.").classes("text-caption text-positive q-mb-md")
        else:
            ui.label(f"{len(active)} aktiva alerts").classes("text-caption q-mb-xs")