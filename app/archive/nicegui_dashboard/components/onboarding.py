"""First-visit onboarding banner."""

from __future__ import annotations

from nicegui import app, ui

_STORAGE_KEY = "onboarding_dismissed"


def render_onboarding_banner() -> None:
    """Show a dismissible 3-step guide for new users."""

    @ui.refreshable
    def banner() -> None:
        if app.storage.user.get(_STORAGE_KEY):
            return
        with ui.card().classes("w-full q-mb-md onboarding-banner"):
            with ui.row().classes("w-full items-start justify-between gap-4"):
                with ui.column().classes("gap-1"):
                    ui.label("Kom igång").classes("text-subtitle1")
                    ui.label(
                        "1. Välj ett samtal i tabellen under Översikt · "
                        "2. Läs transkript och insikter i Samtalsdetalj · "
                        "3. Starta transkribering under Transkribering"
                    ).classes("text-caption")
                ui.button(
                    icon="close",
                    on_click=lambda: (
                        app.storage.user.__setitem__(_STORAGE_KEY, True),
                        banner.refresh(),
                    ),
                ).props("flat round dense").tooltip("Stäng")

    banner()
