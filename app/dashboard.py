"""Streamlit dashboard for Swedish call center conversation intelligence.

Run:
    streamlit run app/dashboard.py

Requires: streamlit, pandas, plotly (install with pip install streamlit plotly)
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit plotly")
    sys.exit(1)

try:
    import plotly.express as px

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    print("Warning: plotly not installed. Charts will be disabled.")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Call Center Insights",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📞 Svensk Call Center – Samtalsintelligens")
st.markdown("Dashboard för sentiment, intent, topics och agent-prestanda.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Inställningar")
    date_range = st.date_input("Datumintervall", value=())
    profile = st.selectbox("Sentimentprofil", ["callcenter", "call", "default", "forum", "news"])
    min_confidence = st.slider("Min konfidens", 0.0, 1.0, 0.5)

    st.divider()
    st.subheader("🧠 Mistral LLM (Fas 3)")
    use_llm = st.checkbox(
        "Aktivera holistisk analys (Mistral)",
        value=False,
        help="Använder ConversationMistralAnalyzer via OpenRouter för trajectory, root cause, actionable insights, agent assessment. Kräver OPENROUTER_API_KEY. Loggar alltid extern dataöverföring.",
    )
    llm_model_choice = st.text_input(
        "Modell (tom = default)",
        value="",
        placeholder="mistralai/mistral-medium-3.5",
    )

    st.divider()
    uploaded_file = st.file_uploader("Ladda upp samtalsdata (JSON)", type=["json"])
    st.divider()
    st.caption("Automatisk Sentimentanalys v0.3.0")


# ---------------------------------------------------------------------------
# Helper: load data
# ---------------------------------------------------------------------------
def load_report(path: str) -> dict[str, Any]:
    """Load a JSON report file."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sample_data() -> dict[str, Any]:
    """Load sample/baseline data for demo."""
    data: dict[str, Any] = {"calls": []}

    # Try to load baseline report
    base = load_report("reports/baseline_results.json")
    if base:
        data["baseline"] = base

    # Generate synthetic demo data with local RNG (avoids global state side effects)
    rng = random.Random(42)

    demo_calls = []
    for i in range(20):
        sentiments = rng.choices(
            ["positiv", "neutral", "negativ"], weights=[0.4, 0.35, 0.25], k=rng.randint(5, 20)
        )
        intents = rng.choices(
            [
                "billing_inquiry",
                "technical_support",
                "information_request",
                "complaint",
                "account_update",
            ],
            weights=[0.3, 0.25, 0.2, 0.15, 0.1],
            k=len(sentiments),
        )
        demo_calls.append(
            {
                "id": f"CALL-{i+1:04d}",
                "timestamp": f"2026-05-{rng.randint(1,25):02d}T{rng.randint(8,18):02d}:00:00Z",
                "duration_s": rng.randint(120, 900),
                "segments": len(sentiments),
                "overall_sentiment": max(set(sentiments), key=sentiments.count),
                "top_intent": max(set(intents), key=intents.count),
                "resolution": rng.choices(
                    ["resolved", "pending", "escalated"], weights=[0.5, 0.3, 0.2]
                )[0],
                "agent": f"Agent-{rng.randint(1,6)}",
            }
        )
    data["calls"] = demo_calls

    # Agent stats
    agents = ["Agent-1", "Agent-2", "Agent-3", "Agent-4", "Agent-5"]
    data["agents"] = [
        {
            "name": a,
            "calls": rng.randint(10, 50),
            "avg_resolution_rate": round(rng.uniform(0.4, 0.9), 2),
            "avg_sentiment_positive": round(rng.uniform(0.3, 0.7), 2),
        }
        for a in agents
    ]

    return data


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    data = json.loads(uploaded_file.read())
else:
    data = load_sample_data()

calls = data.get("calls", [])
df_calls = pd.DataFrame(calls) if calls else pd.DataFrame()

# ---------------------------------------------------------------------------
# Row 1: Key metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

if not df_calls.empty:
    total_calls = len(df_calls)
    resolved = (
        len(df_calls[df_calls.get("resolution", "") == "resolved"])
        if "resolution" in df_calls.columns
        else 0
    )
    avg_duration = int(df_calls["duration_s"].mean()) if "duration_s" in df_calls.columns else 0
    pos_ratio = (
        len(df_calls[df_calls.get("overall_sentiment", "") == "positiv"]) / max(1, total_calls)
        if "overall_sentiment" in df_calls.columns
        else 0
    )
    escalated = (
        len(df_calls[df_calls.get("resolution", "") == "escalated"])
        if "resolution" in df_calls.columns
        else 0
    )

    col1.metric("📞 Totalt samtal", total_calls)
    col2.metric("✅ Lösta", f"{resolved} ({resolved/max(1,total_calls):.0%})")
    col3.metric("⏱️ Snittlängd", f"{avg_duration//60}m {avg_duration%60}s")
    col4.metric("😊 Positiva", f"{pos_ratio:.0%}")
    col5.metric("🚨 Eskalerade", escalated)

# ---------------------------------------------------------------------------
# Row 2: Charts
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("😊 Sentiment-trend")
    if not df_calls.empty and "timestamp" in df_calls.columns:
        df_calls["date"] = pd.to_datetime(df_calls["timestamp"]).dt.date
        sent_trend = df_calls.groupby(["date", "overall_sentiment"]).size().unstack(fill_value=0)
        if _HAS_PLOTLY and not sent_trend.empty:
            fig = px.line(
                sent_trend,
                title="Sentiment över tid",
                labels={"value": "Antal samtal", "date": "Datum"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(sent_trend)
    else:
        st.info("Inga trenddata tillgängliga.")

with col_right:
    st.subheader("🏷️ Top Intents")
    if not df_calls.empty and "top_intent" in df_calls.columns:
        intent_counts = df_calls["top_intent"].value_counts()
        if _HAS_PLOTLY:
            fig = px.pie(
                values=intent_counts.values, names=intent_counts.index, title="Ärendefördelning"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(intent_counts)
    else:
        st.info("Inga intent-data tillgängliga.")

# ---------------------------------------------------------------------------
# Row 3: Agent performance + Hot topics
# ---------------------------------------------------------------------------
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("👤 Agent-prestanda")
    agents = data.get("agents", [])
    if agents:
        df_agents = pd.DataFrame(agents)
        st.dataframe(df_agents, use_container_width=True, hide_index=True)
    else:
        st.info("Inga agent-data tillgängliga.")

with col_right2:
    st.subheader("🔥 Hot Topics")
    baseline = data.get("baseline", {})
    scenarios = baseline.get("scenarios", {})
    if scenarios:
        for name, info in scenarios.items():
            with st.expander(f"{name}: macro-F1 = {info.get('macro_f1', 0):.2%}"):
                st.json(info.get("per_class", {}))
    else:
        st.info("Inga topic-data. Kör python -m src.evaluate scenarios för att generera.")

# ---------------------------------------------------------------------------
# Row 4: Recent calls
# ---------------------------------------------------------------------------
st.subheader("📋 Senaste samtalen")
if not df_calls.empty:
    display_cols = [
        c
        for c in [
            "id",
            "timestamp",
            "duration_s",
            "overall_sentiment",
            "top_intent",
            "resolution",
            "agent",
        ]
        if c in df_calls.columns
    ]
    st.dataframe(df_calls[display_cols].head(10), use_container_width=True, hide_index=True)
else:
    st.info("Inga samtalsdata. Ladda upp en JSON-fil eller generera data först.")

# ---------------------------------------------------------------------------
# Row 5: Live Analysis (run pipeline on user input)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧪 Live-analys")

live_text = st.text_area(
    "Klistra in transkriberade segments (JSON-format):",
    placeholder='[\n  {"text": "Jag har problem med fakturan", "speaker": "kund"},\n  {"text": "Jag ska hjälpa dig", "speaker": "agent"}\n]',
    height=120,
)

if st.button("Analysera") and live_text:
    try:
        live_segments = json.loads(live_text)
        if isinstance(live_segments, list) and live_segments:
            with st.spinner("Kör pipeline..."):
                from src.pipeline import CallAnalysisPipeline

                llm_model = llm_model_choice.strip() or None
                pipe = CallAnalysisPipeline(
                    profile=profile,
                    use_mistral_llm=use_llm,
                    llm_model=llm_model,
                    deep_analysis=use_llm,
                )
                report = pipe.analyze_segments(live_segments)

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Sentiment", report.sentiment_results[0].get("label", "?") if report.sentiment_results else "N/A")
            with col_r2:
                top_intent = report.intent_results[0][0] if report.intent_results else "N/A"
                st.metric("Intent", top_intent)
            with col_r3:
                risk_level = report.risks.get("risk_level", "N/A")
                st.metric("Risknivå", risk_level)

            # LLM-enhanced badge + rich sections (Task 3.3.2)
            llm = getattr(report, "llm", {}) or {}
            llm_meta = llm.get("meta", {}) if isinstance(llm, dict) else {}
            if llm_meta.get("llm_used") or llm.get("actionable_summary"):
                st.success("✨ LLM-enhanced (Mistral via OpenRouter) – se nedan för holistisk analys")
                if llm_meta.get("cost_usd") is not None:
                    st.caption(f"Uppskattad kostnad: ${llm_meta.get('cost_usd'):.5f} | Modell: {llm_meta.get('model', 'okänd')} | Cached: {llm_meta.get('cached', False)}")

                # Actionable insights
                if llm.get("actionable_summary"):
                    with st.expander("📋 Actionable Summary (QA-rekommendationer)", expanded=True):
                        act = llm["actionable_summary"]
                        st.markdown(f"**Problem:** {act.get('problem', '')}")
                        st.markdown(f"**Kundens slutläge:** {act.get('final_customer_state', '')}")
                        recs = act.get("recommendations_for_qa", [])
                        if recs:
                            st.markdown("**Rekommendationer för coachning:**")
                            for r in recs:
                                st.markdown(f"- {r}")

                # Agent assessment
                if llm.get("agent_assessment"):
                    with st.expander("👤 Agent Assessment"):
                        aa = llm["agent_assessment"]
                        st.metric("Empati-score", f"{aa.get('empathy_score', 0):.2f}")
                        if aa.get("compliance_flags"):
                            st.warning("Compliance-flaggor: " + ", ".join(aa.get("compliance_flags", [])))
                        if aa.get("strengths"):
                            st.success("Styrkor: " + "; ".join(aa.get("strengths", [])))

                # Trajectory / root cause (textual)
                if llm.get("trajectory"):
                    with st.expander("📈 Trajectory (kundresa)"):
                        tr = llm["trajectory"]
                        st.markdown(tr.get("summary", ""))
                        if tr.get("escalation_events"):
                            st.markdown("**Eskalationer:** " + " | ".join(tr.get("escalation_events", [])))

                if llm.get("root_cause"):
                    with st.expander("🔍 Root Cause"):
                        rc = llm["root_cause"]
                        st.markdown(f"**Primär orsak:** {rc.get('primary_cause', '')}")
                        st.json(rc)

            with st.expander("Detaljerade resultat (full CallAnalysisReport)"):
                st.json(report.to_dict())
        else:
            st.warning("Inmatningen måste vara en lista med segment.")
    except json.JSONDecodeError as e:
        st.error(f"Ogiltig JSON: {e}")
    except Exception as e:
        st.error(f"Analys misslyckades: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(f"Automatisk Sentimentanalys v0.3.0 | Genererad {datetime.now().isoformat()[:19]}")
