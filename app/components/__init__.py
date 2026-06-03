"""Dashboard UI components (Streamlit)."""

from __future__ import annotations


def inject_global_styles() -> None:
    """Inject scoped CSS for dashboard (idempotent)."""
    try:
        import streamlit as st

        st.markdown(
            """
            <style>
            .sentiment-badge { padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass
