"""Call Detail View MVP components."""

from __future__ import annotations

from typing import Any


def render_call_detail_header(report: dict[str, Any], call_id: str, **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.header(f"Samtal {call_id}")
        render_sentiment = report.get("overall_sentiment") or "—"
        st.caption(f"Sentiment: {render_sentiment}")
    except Exception:
        pass


def render_interactive_timeline(
    report: dict[str, Any],
    enriched: list[dict[str, Any]],
    *,
    key_prefix: str = "",
    **kwargs: Any,
) -> int | None:
    try:
        import streamlit as st

        if not enriched:
            st.info("Inga segment.")
            return None
        labels = [f"{i}: {s.get('speaker', '?')}" for i, s in enumerate(enriched)]
        choice = st.selectbox("Segment", range(len(labels)), format_func=lambda i: labels[i])
        return int(choice)
    except Exception:
        return None


def render_transcript(
    report: dict[str, Any],
    enriched: list[dict[str, Any]],
    *,
    selected_idx: int | None = None,
    search_term: str = "",
    key_prefix: str = "",
    **kwargs: Any,
) -> set[int]:
    try:
        import streamlit as st

        term = (search_term or st.text_input("Sök i transkript", key=f"tx_search_{key_prefix}")).lower()
        flagged: set[int] = set()
        for i, seg in enumerate(enriched):
            text = seg.get("text", "")
            if term and term not in text.lower():
                continue
            prefix = ">> " if selected_idx == i else ""
            st.markdown(f"{prefix}**{seg.get('speaker', '?')}**: {text}")
            if seg.get("has_compliance_flag"):
                flagged.add(i)
        return flagged
    except Exception:
        return set()


def render_structured_insights(report: dict[str, Any], *, key_prefix: str = "", **kwargs: Any) -> None:
    try:
        import streamlit as st

        with st.expander("Strukturerade insikter", expanded=True):
            for key in ("summary", "insights", "llm", "results"):
                if report.get(key):
                    st.markdown(f"**{key}**")
                    st.json(report[key])
    except Exception:
        pass


def render_action_panel(report: dict[str, Any], call_id: str, **kwargs: Any) -> None:
    try:
        import streamlit as st

        c1, c2 = st.columns(2)
        with c1:
            st.button("Lägg i coaching-kö", key=f"coach_{call_id}")
        with c2:
            st.button("Flagga samtal", key=f"flag_{call_id}")
    except Exception:
        pass
