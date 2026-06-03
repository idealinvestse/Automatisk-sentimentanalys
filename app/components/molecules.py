"""Molecular UI elements (KPI cards, action rows)."""

from __future__ import annotations

from typing import Any


def render_kpi_card(title: str, value: str, *, delta: str | None = None, **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.metric(title, value, delta=delta)
    except Exception:
        pass


def render_action_buttons(actions: list[dict[str, Any]] | None = None, **kwargs: Any) -> None:
    try:
        import streamlit as st

        for action in actions or []:
            label = action.get("label", "Action")
            if st.button(label, key=kwargs.get("key", label)):
                pass
    except Exception:
        pass
