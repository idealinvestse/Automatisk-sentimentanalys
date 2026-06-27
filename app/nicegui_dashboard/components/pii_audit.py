"""PII Redaction Audit Panel for NiceGUI Dashboard.

Displays PII redaction events from pipeline results['pii_redaction'].
Follows existing component patterns (demo data fallback, empty state, dark theme).
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state


def render_pii_audit_panel(results: dict[str, Any] | None = None) -> None:
    """Render PII audit panel showing redaction breakdown.

    Args:
        results: Pipeline output containing optional 'pii_redaction' key.
                 If None or missing, shows demo/empty state.
    """
    pii_log = results.get("pii_redaction") if results else None

    with ui.card().classes("w-full"):
        ui.label("PII-redaktion – granskning").classes("text-h6 q-mb-sm")
        ui.separator()

        if not pii_log or not pii_log.get("events"):
            render_empty_state(
                icon="shield",
                title="Ingen PII-redaktion utförd",
                hint=(
                    "När profilens llm.anonymize_before_llm=True och PII hittas, "
                    "visas detaljer här."
                ),
            )
            return

        _render_summary_stats(pii_log)
        _render_breakdown(pii_log)
        _render_sample_events(pii_log)


def _render_summary_stats(pii_log: dict[str, Any]) -> None:
    """Render total events and types redacted."""
    total = pii_log.get("total_redacted", 0)
    types = pii_log.get("types_redacted", [])
    profile = pii_log.get("profile", "callcenter")

    with ui.row().classes("w-full items-center q-mb-md"):
        with ui.badge(f"Totalt: {total} händelser", color="primary"):
            pass
        for t in types:
            ui.badge(t, color="grey-8")
        ui.label(f"Profil: {profile}").classes("text-caption text-grey-6 q-ml-auto")


def _render_breakdown(pii_log: dict[str, Any]) -> None:
    """Render per-PII-type breakdown counts."""
    events = pii_log.get("events", [])
    if not events:
        return

    # Count by type
    counts: dict[str, int] = {}
    for ev in events:
        t = ev.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1

    ui.label("Fördelning per typ").classes("text-subtitle2 q-mb-xs")
    with ui.row().classes("q-gutter-sm"):
        for t, c in sorted(counts.items()):
            ui.chip(f"{t}: {c}", icon="visibility_off").classes("q-px-sm")


def _render_sample_events(pii_log: dict[str, Any]) -> None:
    """Render up to 5 sample redacted snippets."""
    events = pii_log.get("events", [])
    if not events:
        return

    ui.label("Exempel på redigerade värden").classes("text-subtitle2 q-mt-md q-mb-xs")
    with ui.list().classes("w-full"):
        for ev in events[:5]:
            orig = ev.get("original", "")
            repl = ev.get("replacement", "[REDACTED]")
            t = ev.get("type", "?")
            snippet = orig[:48] + "..." if len(orig) > 50 else orig

            with ui.item():
                with ui.item_section():
                    ui.item_label(f"[{t}]").classes("text-caption")
                    ui.item_label(snippet).classes("text-mono text-grey-8")
                with ui.item_section().props("side"):
                    ui.chip(repl, color="red-6").props("dense")


def render_pii_section_in_call_detail(report: dict[str, Any] | None = None) -> None:
    """Convenience wrapper for embedding inside call_detail tab.

    Safe to call even if report has no pii_redaction — shows empty state.
    """
    if report and "pii_redaction" in report.get("results", {}):
        render_pii_audit_panel(report["results"])
    else:
        # Demo fallback (headless mode)
        demo_log = {
            "total_redacted": 3,
            "types_redacted": ["personnummer", "email", "phone"],
            "profile": "callcenter",
            "events": [
                {"type": "personnummer", "original": "19850101-1234", "replacement": "[REDACTED_PNR]"},
                {"type": "email", "original": "anna@example.se", "replacement": "[REDACTED_EMAIL]"},
                {"type": "phone", "original": "070-123 45 67", "replacement": "[REDACTED_PHONE]"},
            ],
        }
        render_pii_audit_panel({"pii_redaction": demo_log})
