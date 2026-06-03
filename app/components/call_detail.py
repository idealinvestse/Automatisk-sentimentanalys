"""Call detail sub-components (design-system layer)."""

from __future__ import annotations

from typing import Any


def render_call_header(report: dict[str, Any], *, key: str | None = None, **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.subheader(report.get("call_id") or report.get("title") or "Samtal")
    except Exception:
        pass


def render_timeline(report: dict[str, Any], enriched: list[dict[str, Any]], **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.caption(f"Timeline: {len(enriched)} segment")
    except Exception:
        pass


def render_transcript(
    report: dict[str, Any],
    enriched: list[dict[str, Any]],
    **kwargs: Any,
) -> set[int]:
    try:
        import streamlit as st

        for seg in enriched:
            st.text(f"[{seg.get('speaker', '?')}] {seg.get('text', '')[:120]}")
    except Exception:
        pass
    return set()


def render_structured_insight(report: dict[str, Any], **kwargs: Any) -> None:
    try:
        import streamlit as st

        with st.expander("Insights"):
            st.json(report.get("results", {}))
    except Exception:
        pass
