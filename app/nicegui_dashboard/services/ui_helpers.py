"""UI helper utilities: notifications and loading patterns.

Fas 4 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from nicegui import ui

from app.nicegui_dashboard.services.nicegui_api_client import APIError

T = TypeVar("T")


def notify_success(message: str) -> None:
    ui.notify(message, type="positive")


def notify_warning(message: str) -> None:
    ui.notify(message, type="warning")


def notify_error(message: str) -> None:
    ui.notify(message, type="negative")


def notify_api_error(err: APIError | Exception, *, prefix: str = "") -> None:
    """Show user-friendly API error notification."""
    msg = str(err)
    if isinstance(err, APIError) and err.status_code:
        msg = f"{prefix}{msg} (HTTP {err.status_code})".strip()
    elif prefix:
        msg = f"{prefix}{msg}"
    notify_error(msg)


async def with_loading(
    container: ui.element,
    coro: Awaitable[T],
    *,
    loading_text: str = "Laddar...",
    on_error: Callable[[Exception], None] | None = None,
) -> T | None:
    """Run async task with spinner in container; return None on error."""
    container.clear()
    with container:
        ui.spinner(size="lg")
        ui.label(loading_text).classes("text-caption q-mt-sm")
    try:
        result = await coro
        container.clear()
        return result
    except Exception as err:
        container.clear()
        with container:
            ui.label(f"Fel: {err}").classes("text-negative text-caption")
        if on_error:
            on_error(err)
        else:
            notify_api_error(err)
        return None
