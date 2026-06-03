"""Atomic UI elements for the dashboard."""

from __future__ import annotations

from typing import Any


def render_sentiment_badge(label: str, **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.caption(f"Sentiment: {label}")
    except Exception:
        pass


def render_qa_badge(score: float | None = None, **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.caption(f"QA: {score if score is not None else '—'}")
    except Exception:
        pass


def render_risk_badge(level: str = "low", **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.caption(f"Risk: {level}")
    except Exception:
        pass
