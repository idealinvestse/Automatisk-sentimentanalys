"""Streamlit dashboard for Swedish call center conversation intelligence (Fas 5.0 MVP).

================================================================================
RUN INSTRUCTIONS (updated for v0.4 / harmonized plan)
================================================================================
    streamlit run app/dashboard.py

Requires (in addition to project):
    pip install streamlit plotly pandas

The app now delivers the core of Fas 5.0 per UTVECKLINGSPLAN_Frontend_UX_CallCenter_v1.1.md:
- Call Detail View (highest prio): Header (meta+sentiment+QA+risk+actions), Interactive
  Timeline (plotly + clickable buttons), Searchable Transcript (sentiment colors,
  nyckelmoment highlights from qa/llm/alerts, manual per-turn flag), Structured Insights
  (AI summary, actionable w/ recs+evidence, Agent Assessment hybrid, QA Scorecard,
  Root Cause/Trajectory, evidence bucket).
- Improved main Dashboard: 6-8 clickable KPI cards (filter on click), Sentiment trend
  (plotly), Hot Topics (click filter), Agent Leaderboard (table + click filter),
  Alerts Panel w/ quick actions (coaching/flag), filtered calls table w/ "Öppna detalj".
- Disciplined st.session_state: view ('overview'|'call_detail'), selected_call (id),
  filters (persistent dict), selected_segment_idx, transcript_search, coaching_queue,
  flagged_calls, flagged_turns (per-call), reports (list of dicts), live_report.
- st.cache_data (in services) for generate_demo_reports + heavy computations.
- Real backend: uses src.pipeline.CallAnalysisPipeline directly (profile="callcenter")
  via app/services/data_services on 5 realistic canned Swedish convos (pos/neg/complex).
  Surfaces Fas3/4 fields (llm, results['qa'], results['agent_performance'],
  results['alerts'], agent_assessment, etc) beautifully with graceful fallbacks.
- Always shows **evidence** (spans/quotes from local engines + LLM).
- Loading spinners, error handling, consistent UX/colors (high contrast), labels.
- "Live-analys" kept + enhanced: after run, "Visa i Call Detail View" populates state
  from that report (and adds to current reports list for overview).
- Upload supports full CallAnalysisReport JSON (single or list of reports).
- Components in app/components/ (call_detail_view.py, dashboard_widgets.py) for
  future refactor / migration. Code is migratable: components take serializable data only.
  Comments: "Future React: this becomes <FooBar props={report} />"

Demo data controls in sidebar let you (re)generate real pipeline reports (toggle LLM).

Performance: pipeline runs ONLY on first load / explicit regenerate (cached by
st.cache_data with inputs use_llm+profile+transcript hash). No full recompute on reruns.
Use hashes internally in services.

Existing test_load_sample_data still passes (synthetic helper preserved exactly).

Future API hook comments are inline for /analyze_pipeline etc (when switching from
direct CallAnalysisPipeline to HTTP in prod dashboard).

================================================================================
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports (dashboard + services + src)
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
# Import our new high-quality services + components (Fas 5.0)
# ---------------------------------------------------------------------------
# Services: canned data, @st.cache_data generate_demo_reports (REAL pipeline),
#           helpers (extract, kpis, filter, enrich, ingest, evidence).
from app.services.data_services import (
    compute_kpis,
    enrich_segments_with_sentiment,
    filter_reports,
    generate_demo_reports,
    get_demo_transcripts,
    get_evidence_quotes,
    ingest_uploaded_report,
    collect_all_alerts,
    get_hot_topics,
    get_agent_leaderboard,
)

# Components coordination:
# - Use design agent's atomic/molecular/call_detail (from sub-agent output) for badges + core renders where possible.
# - Fall back to / also use our detailed MVP implementations in call_detail_view / dashboard_widgets
#   (which were written to satisfy the full spec for timeline+transcript+header+insights+KPIs).
# Direct submodule imports (bypass __init__ public surface) so both design system + our full impls coexist.
from app.components import inject_global_styles
from app.components.call_detail import (
    render_call_header,
    render_timeline,
    render_transcript,
    render_structured_insight,
)
from app.components.call_detail_view import (
    render_call_detail_header,
    render_interactive_timeline,
    render_transcript as render_transcript_mvp,
    render_structured_insights,
    render_action_panel,
)
from app.components.dashboard_widgets import (
    render_kpi_cards,
    render_sentiment_trend,
    render_hot_topics,
    render_agent_leaderboard,
    render_alerts_panel,
    render_filtered_calls_table,
)
from app.components.molecules import render_kpi_card, render_action_buttons
from app.components.atoms import (
    render_sentiment_badge,
    render_qa_badge,
    render_risk_badge,
)

# ---------------------------------------------------------------------------
# Page config (wide for dashboard + detail)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Call Center Insights – Fas 5.0 MVP",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Coordinate with design system: inject scoped CSS (idempotent, guarded in styles.py)
inject_global_styles()

# Sätt defaultnyckel från fil (configs/openrouter.key) om ingen env var finns.
# Detta gör att all LLM-användning (demo + live) får nyckeln automatiskt.
# Användaren kan fortfarande override:a i sidebaren längre ner.
# Wrapped to avoid repeated SECURITY warnings on every Streamlit rerun.
if not os.getenv("OPENROUTER_API_KEY"):
    try:
        from src.llm.openrouter_client import load_openrouter_key_from_file
        load_openrouter_key_from_file(set_as_env=True)
    except Exception:
        pass

st.title("📞 Svensk Call Center – Samtalsintelligens (Fas 5.0 MVP)")
st.markdown(
    "Dashboard för sentiment, intent, topics, agent-prestanda, QA och actionable insights. "
    "Byggd på **riktig** `CallAnalysisPipeline` (callcenter-profil). "
    "**Högsta prioritet: Call Detail View** + förbättrad översikt med filter från KPI:er."
)

# ---------------------------------------------------------------------------
# DISCIPLINED SESSION STATE MACHINE (per plan)
# view: 'overview' | 'call_detail'
# selected_call: str call_id or None
# filters: dict with sentiment_filter, agent_filter, has_qa_fail, min_risk, topic_filter, search
# selected_segment_idx: int | None (sync timeline <-> transcript)
# transcript_search: str (live filter in detail)
# coaching_queue: list[dict] (action handler results)
# flagged_calls: dict[id -> info]
# flagged_turns: dict[call_id -> set[int]]
# reports: list[dict] (full serializable report.to_dict() + call_id/title/meta)
# live_report: last live-analyzed dict (for "Visa i Call Detail")
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "view": "overview",
        "selected_call": None,
        "filters": {
            "sentiment_filter": "all",
            "agent_filter": None,
            "has_qa_fail": None,
            "min_risk": None,
            "topic_filter": None,
            "search": "",
        },
        "selected_segment_idx": None,
        "transcript_search": "",
        "coaching_queue": [],
        "flagged_calls": {},
        "flagged_turns": {},
        "reports": [],
        "live_report": None,
        "_force_demo_reload": False,
        "_demo_use_llm": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()

# ---------------------------------------------------------------------------
# SIDEBAR – global controls + NEW "Demo data controls" + action state
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Inställningar & Data")

    # Profile & LLM (shared for live + demo generation)
    profile = st.selectbox(
        "Sentimentprofil",
        ["callcenter", "call", "default", "forum", "news"],
        index=0,  # callcenter is the rich Fas4/LLM profile
        help="callcenter-profil aktiverar Fas4-motorer + LLM-heuristik som default.",
    )
    min_confidence = st.slider("Min konfidens (visuellt)", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.subheader("🧠 Mistral LLM (Fas 3 + hybrid Fas4)")
    use_llm = st.checkbox(
        "Aktivera holistisk analys (Mistral)",
        value=st.session_state.get("_demo_use_llm", False),
        help="Använder ConversationMistralAnalyzer via OpenRouter. Kräver OPENROUTER_API_KEY (env var) eller configs/openrouter.key (dev convenience only, gitignored). Loggar alltid extern dataöverföring. Påverkar både Live-analys och demo-generering.",
    )
    st.session_state["_demo_use_llm"] = use_llm
    llm_model_choice = st.text_input(
        "Modell (tom = default från profil)",
        value="",
        placeholder="mistralai/mistral-medium-3.5",
    )

    # NEW: Allow easy override of the OpenRouter key directly from the UI
    # This uses the helper so that default comes from file (or env), and user can change it here.
    with st.expander("🔑 OpenRouter API Key (override)", expanded=False):
        # Safe Streamlit pattern: use key= for state, avoid value= that references the same key to prevent widget rerun warnings.
        llm_api_key_override = st.text_input(
            "API Key",
            type="password",
            placeholder="Lämna tom för att använda default från configs/openrouter.key / env",
            help="Används endast för denna session. Prioritet: detta fält > env var > fil. Använder helper load_openrouter_key_from_file.",
            key="llm_api_key_override",
        )

    # Ensure we have a clean string (empty means "use default from helper/file/env")
    llm_api_key_override = (llm_api_key_override or "").strip()

    st.divider()
    st.subheader("📦 Demo data controls (Fas 5.0)")
    st.caption("Genererar 5 realistiska svenska callcenter-samtal via **riktig** pipeline (callcenter).")

    demo_llm = st.checkbox("Använd LLM även för demo-generering", value=use_llm, key="demo_llm_toggle")
    if st.button("🔄 Generera / Ladda demo-rapporter (pipeline)", use_container_width=True):
        st.session_state["_force_demo_reload"] = True
        st.session_state["_demo_use_llm"] = demo_llm
        st.toast("Laddar demo via pipeline... (cache skyddar efter första)")
        st.rerun()

    if st.button("🗑️ Rensa demo-cache & ladda om", use_container_width=True):
        # st.cache_data has no direct clear in public API easily; force by changing a dummy
        st.session_state["_force_demo_reload"] = True
        st.session_state["reports"] = []
        st.toast("Tvingar omgenerering.")
        st.rerun()

    st.caption("Perf: Första körning laddar modeller (30-90s). Därefter instant pga @st.cache_data.")

    st.divider()
    uploaded_file = st.file_uploader(
        "Ladda upp full CallAnalysisReport JSON (en eller lista)",
        type=["json"],
        help="Stödjer report.to_dict() eller list av sådana. Mergas in i aktuella rapporter.",
    )
    if uploaded_file is not None:
        # Will be handled in main load block below
        pass

    st.divider()
    st.subheader("📋 Åtgärdskö (session)")
    if st.session_state.get("coaching_queue"):
        with st.expander(f"Coaching tasks ({len(st.session_state.coaching_queue)})", expanded=False):
            for i, task in enumerate(st.session_state.coaching_queue[-5:]):
                st.write(f"{i+1}. {task.get('call_id')} – {task.get('priority')}")
                st.caption(str(task.get("evidence", ""))[:100])
    else:
        st.caption("Inga coaching tasks än. Skapa via Call Detail.")

    if st.session_state.get("flagged_calls"):
        with st.expander(f"Flaggade samtal ({len(st.session_state.flagged_calls)})", expanded=False):
            for cid, info in list(st.session_state.flagged_calls.items())[-4:]:
                st.write(f"- {cid}: {info.get('reason', 'Manuell')}")
    else:
        st.caption("Inga flaggade.")

    st.divider()
    # Quick navigation
    if st.button("🏠 Gå till Översikt", use_container_width=True):
        st.session_state["view"] = "overview"
        st.session_state["selected_call"] = None
        st.rerun()

    st.caption("Automatisk Sentimentanalys v0.4.0 (Fas 5.0 MVP)")

# ---------------------------------------------------------------------------
# DATA LOADING (real pipeline demos preferred; upload support; synthetic kept for tests)
# ---------------------------------------------------------------------------

def _load_current_reports() -> list[dict[str, Any]]:
    """Load or (re)generate the working set of reports.

    Priority:
    1. If forced or empty -> run generate_demo_reports (uses real pipeline + cache).
    2. Merge any uploaded full report JSON.
    3. Fall back to any live_report.
    4. Never lose previously generated in this session.
    """
    reports: list[dict[str, Any]] = list(st.session_state.get("reports", []))

    force = st.session_state.get("_force_demo_reload", False)
    if force or not reports:
        use_llm_for_demo = st.session_state.get("_demo_use_llm", False) or use_llm
        with st.spinner("Kör pipeline på 5 demo-samtal (callcenter-profil)..."):
            try:
                new_reports = generate_demo_reports(
                    use_llm=use_llm_for_demo,
                    profile=profile,
                    llm_api_key=llm_api_key_override or None,
                )
                reports = new_reports
                st.session_state["reports"] = reports
                st.success(f"Demo-rapporter genererade ({len(reports)} samtal). Källa: riktig pipeline.")
            except Exception as e:
                st.error(f"Demo-generering misslyckades: {e}. Försök igen eller använd upload/live.")
                reports = reports or []
        st.session_state["_force_demo_reload"] = False

    # Handle upload (full report(s))
    if uploaded_file is not None:
        try:
            ingested = ingest_uploaded_report(uploaded_file)
            if ingested:
                # Prepend uploaded (user data first) and dedup on call_id if possible
                existing_ids = {r.get("call_id") for r in reports}
                for r in ingested:
                    cid = r.get("call_id") or r.get("id") or f"UPLOAD-{len(reports)}"
                    r["call_id"] = cid
                    if cid not in existing_ids:
                        reports.insert(0, r)
                st.session_state["reports"] = reports
                st.info(f"Uppladdad rapport/lista mergad ({len(ingested)}).")
        except Exception as ue:
            st.warning(f"Kunde inte läsa uppladdad JSON som rapport: {ue}")

    # Include live if not already present
    live = st.session_state.get("live_report")
    if live and isinstance(live, dict):
        if not any(r.get("call_id") == live.get("call_id") for r in reports):
            reports = [live] + reports
            st.session_state["reports"] = reports

    return reports


reports = _load_current_reports()

# Current filters from session (persistent across reruns)
filters = st.session_state.get("filters", {})
filtered_reports = filter_reports(reports, filters) if reports else []

# ---------------------------------------------------------------------------
# VIEW STATE MACHINE RENDER
# ---------------------------------------------------------------------------
current_view = st.session_state.get("view", "overview")
selected_call_id = st.session_state.get("selected_call")

if current_view == "call_detail" and selected_call_id:
    # -----------------------------------------------------------------------
    # CALL DETAIL VIEW (highest priority per plan)
    # -----------------------------------------------------------------------
    # Find the report (may be in live or demo or uploaded)
    selected_report = None
    for r in reports:
        if r.get("call_id") == selected_call_id:
            selected_report = r
            break
    if not selected_report and st.session_state.get("live_report", {}).get("call_id") == selected_call_id:
        selected_report = st.session_state["live_report"]

    if not selected_report:
        st.error(f"Kunde inte hitta rapport för {selected_call_id}. Återgår till översikt.")
        st.session_state["view"] = "overview"
        st.session_state["selected_call"] = None
        st.rerun()

    # Enrich segments once (for timeline + transcript sync)
    segments = selected_report.get("segments", []) or []
    sents = selected_report.get("sentiment_results", []) or []
    enriched = enrich_segments_with_sentiment(segments, sents)

    # Also enrich with qa/alert evidence flags for transcript highlights (post-process)
    qa = (selected_report.get("results") or {}).get("qa") or (selected_report.get("results") or {}).get("compliance_qa") or {}
    comp_flags = [str(f).lower() for f in (qa.get("compliance_flags") or [])]
    for seg in enriched:
        txt_lower = seg.get("text", "").lower()
        if any(f in txt_lower for f in comp_flags):
            seg["has_compliance_flag"] = True

    # Header (actions inside mutate session coaching/flagged + downloads) - our full MVP impl
    render_call_detail_header(selected_report, selected_call_id)

    # Also render design system header (coordinates with sub-agent output; shows badges + meta using atoms)
    # This demonstrates dual usage / migration path.
    try:
        render_call_header(selected_report, key=f"design_hdr_{selected_call_id}")
    except Exception:
        pass  # graceful if design header needs callbacks etc.

    # Timeline + selection sync
    st.markdown("#### Interaktiv Timeline")
    # Read current selection from state (or reset)
    sel_idx = st.session_state.get("selected_segment_idx")
    clicked_idx = render_interactive_timeline(selected_report, enriched, key_prefix=selected_call_id)
    if clicked_idx is not None and clicked_idx != sel_idx:
        st.session_state["selected_segment_idx"] = clicked_idx
        # Re-render will pick it up; no full rerun needed but helps highlight immediately
        st.rerun()

    # Transcript (search + colors + highlights + per-turn flag)
    st.markdown("#### Transkript")
    current_search = st.session_state.get("transcript_search", "")
    # The render func handles its own search input + live filter; we pass for consistency
    current_flagged = render_transcript(
        selected_report,
        enriched,
        selected_idx=st.session_state.get("selected_segment_idx"),
        search_term=current_search,
        key_prefix=selected_call_id,
    )
    # Store back if changed inside
    if current_flagged:
        st.session_state.setdefault("flagged_turns", {})[selected_call_id] = set(current_flagged)

    # Structured Insights (right / below – full evidence)
    render_structured_insights(selected_report, key_prefix=selected_call_id)

    # Action footer
    render_action_panel(selected_report, selected_call_id)

    # Debug / evidence raw (collapsible, for power users)
    with st.expander("🔬 Raw report (to_dict) + evidence (debug)"):
        st.json(selected_report)
        st.write("Evidence quotes:")
        st.json(get_evidence_quotes(selected_report))

    # Small note
    st.caption(
        "Future React: Call Detail View becomes a dedicated route/page. "
        "Timeline uses SVG/canvas + virtualized transcript. State via Zustand + URL sync."
    )

else:
    # -----------------------------------------------------------------------
    # OVERVIEW / ENHANCED DASHBOARD
    # -----------------------------------------------------------------------
    st.subheader("📊 Översikt – KPI:er, trender & filter (klickbara kort muterar vy)")

    # KPI cards (clickable) – they return new filters; we persist + rerun if changed
    new_filters = render_kpi_cards(reports, filters)
    if new_filters != filters:
        st.session_state["filters"] = new_filters
        st.rerun()

    # Re-apply after possible mutation
    filters = st.session_state.get("filters", {})
    filtered_reports = filter_reports(reports, filters) if reports else []

    # Two column: trend + hot topics
    col_trend, col_hot = st.columns([3, 2])
    with col_trend:
        render_sentiment_trend(filtered_reports or reports, key="main")
    with col_hot:
        new_f_hot = render_hot_topics(reports, filters, key="main")
        if new_f_hot != filters:
            st.session_state["filters"] = new_f_hot
            st.rerun()

    # Leaderboard + Alerts
    col_board, col_alerts = st.columns([1, 1])
    with col_board:
        new_f_board = render_agent_leaderboard(reports, filters, key="main")
        if new_f_board != filters:
            st.session_state["filters"] = new_f_board
            st.rerun()
    with col_alerts:
        render_alerts_panel(reports, key="main")

    # Filtered calls table + open detail buttons (the main navigation to Call Detail)
    chosen = render_filtered_calls_table(reports, filtered_reports, key="main")
    if chosen:
        st.session_state["selected_call"] = chosen
        st.session_state["view"] = "call_detail"
        st.session_state["selected_segment_idx"] = None
        st.session_state["transcript_search"] = ""
        st.rerun()

    # Summary bar
    kpi_now = compute_kpis(filtered_reports or reports, filters)
    st.caption(
        f"Visar {kpi_now['total_calls']} samtal | Pos {kpi_now['pos_pct']}% | Neg {kpi_now['neg_pct']}% | "
        f"Alerts {kpi_now['alerts_count']} | QA snitt {kpi_now.get('qa_avg', '–')}"
    )

    st.caption(
        "Future React: Overview becomes dashboard page with URL-synced filters (TanStack Router), "
        "real-time alerts via WS, personal saved views. "
        "Migration plan + component map: docs/REACT_MIGRATION_PLAN.md"
    )

# ---------------------------------------------------------------------------
# LIVE-ANALYS (kept + enhanced with "Visa i Call Detail View")
# Always visible for quick ad-hoc analysis → detail view
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧪 Live-analys (klistra in segments → pipeline → direkt till Call Detail)")

live_text = st.text_area(
    "Klistra in transkriberade segments (JSON-format):",
    placeholder='[\n  {"text": "Jag har problem med fakturan", "speaker": "Kund", "start": 0, "end": 8},\n  {"text": "Jag ska hjälpa dig", "speaker": "Agent", "start": 8, "end": 15}\n]',
    height=120,
    key="live_text_area",
)

if st.button("Analysera (pipeline)", key="live_analyze_btn") and live_text:
    try:
        live_segments = json.loads(live_text)
        if isinstance(live_segments, list) and live_segments:
            with st.spinner("Kör pipeline (med aktuella inställningar)..."):
                from src.pipeline import CallAnalysisPipeline

                llm_model = llm_model_choice.strip() or None
                pipe = CallAnalysisPipeline(
                    profile=profile,
                    use_mistral_llm=use_llm,
                    llm_model=llm_model,
                    deep_analysis=use_llm,
                    llm_api_key=llm_api_key_override or None,
                )
                report = pipe.analyze_segments(live_segments)

            # Show quick metrics (backward compat + Fas4 visibility)
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                s0 = report.sentiment_results[0] if report.sentiment_results else {}
                st.metric("Sentiment", s0.get("label", "?"))
            with col_r2:
                top_intent = report.intent_results[0][0] if report.intent_results else "N/A"
                st.metric("Intent", top_intent)
            with col_r3:
                risk_level = report.risks.get("risk_level", "N/A")
                st.metric("Risknivå", risk_level)
            with col_r4:
                qa = (report.results or {}).get("qa") or {}
                st.metric("QA", f"{qa.get('overall_qa_score', '–')}/100" if qa else "–")

            # LLM-enhanced badge + rich sections (kept from original, now with more Fas4)
            llm = getattr(report, "llm", {}) or {}
            llm_meta = llm.get("meta", {}) if isinstance(llm, dict) else {}
            if llm_meta.get("llm_used") or llm.get("actionable_summary"):
                st.success("✨ LLM-enhanced (Mistral via OpenRouter) – se nedan för holistisk analys")
                if llm_meta.get("cost_usd") is not None:
                    st.caption(
                        f"Uppskattad kostnad: ${llm_meta.get('cost_usd'):.5f} | "
                        f"Modell: {llm_meta.get('model', 'okänd')} | Cached: {llm_meta.get('cached', False)}"
                    )

                # Actionable insights (kept)
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

                # Fas 4.1/4.2 surface (kept + comments)
                r = getattr(report, "results", {}) or {}
                ap = r.get("agent_performance") or {}
                if ap and isinstance(ap, dict):
                    with st.expander("📊 Agent Performance + Customer Metrics (Fas4)"):
                        a = ap.get("agent", {}) or {}
                        c = ap.get("customer", {}) or {}
                        st.metric("Empathy (local)", f"{a.get('empathy_score', 0):.2f}")
                        st.write("Talk ratio / listen:", a.get("talk_ratio"), a.get("talk_listen_ratio"))
                        st.write("Compliance flags:", a.get("compliance_flags", []))
                        if ap.get("local_coaching_hints"):
                            st.info("Hints: " + " | ".join(ap["local_coaching_hints"][:2]))
                        st.caption(f"Customer talk_ratio={c.get('talk_ratio')}, slope={c.get('sentiment_slope')}")

                qa = r.get("qa") or r.get("compliance_qa") or {}
                if qa and isinstance(qa, dict) and qa.get("overall_qa_score") is not None:
                    with st.expander("✅ Compliance / QA Auto-Score (Fas4)"):
                        st.metric("Overall QA Score", f"{qa.get('overall_qa_score')}/100")
                        st.write(f"Passed: {qa.get('passed')} | Risk: {qa.get('risk_level')}")
                        if qa.get("compliance_flags"):
                            st.warning("Flags: " + "; ".join(qa["compliance_flags"][:3]))
                        st.caption("See results['qa'] for full per-criterion evidence.")

                # Trajectory / root cause
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

            # FULL REPORT EXPANDER (kept)
            with st.expander("Detaljerade resultat (full CallAnalysisReport)"):
                st.json(report.to_dict())

            # ------------------------------------------------------------------
            # NEW: "Visa i Call Detail View" – populates state from this live report
            # ------------------------------------------------------------------
            st.divider()
            if st.button("👁️ Visa i Call Detail View (Fas 5.0)", key="live_to_call_detail", type="primary"):
                live_dict = report.to_dict()
                live_dict["call_id"] = f"LIVE-{datetime.now().strftime('%H%M%S')}"
                live_dict["title"] = "Live-analyserat samtal"
                live_dict["meta"] = live_dict.get("meta") or {"agent": "Live User", "duration_s": "–"}
                # Store for immediate detail + merge into reports list
                st.session_state["live_report"] = live_dict
                st.session_state["selected_call"] = live_dict["call_id"]
                st.session_state["view"] = "call_detail"
                st.session_state["selected_segment_idx"] = None
                st.session_state["transcript_search"] = ""
                # Add to reports so overview also sees it
                if "reports" not in st.session_state:
                    st.session_state["reports"] = []
                # Avoid dups
                st.session_state["reports"] = [live_dict] + [
                    rr for rr in st.session_state["reports"] if rr.get("call_id") != live_dict["call_id"]
                ]
                st.toast("Öppnar i Call Detail View – använder exakt samma pipeline-rapport.")
                st.rerun()

        else:
            st.warning("Inmatningen måste vara en lista med segment.")
    except json.JSONDecodeError as e:
        st.error(f"Ogiltig JSON: {e}")
    except Exception as e:
        st.error(f"Analys misslyckades: {e}")
        # Show traceback hint for devs
        if st.checkbox("Visa detaljerad felinfo (dev)"):
            import traceback

            st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# FOOTER + PERF / INTEGRATION NOTES
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Automatisk Sentimentanalys v0.4.0 (Fas 5.0 MVP) | Genererad {datetime.now().isoformat()[:19]} | "
    f"Profiler: {profile} | LLM: {'på' if use_llm else 'av'} | Rapporter i minne: {len(reports)}"
)

# Integration comment (for future API switch)
# FUTURE: Replace direct CallAnalysisPipeline(...) + .analyze_segments with
#   import requests; r = requests.post("http://localhost:8000/analyze_pipeline", json=...)
#   then use the response (identical shape to report.to_dict() + results).
#   Same for /insights/hot_topics, /alerts etc. Services layer already prepares for it.

# Performance note (as required)
# AVOID RECOMPUTE: generate_demo_reports is @st.cache_data. Live runs only on button.
# Hash keys inside services use transcript content. KPI/filter are pure O(n) on small N.
# For 100+ calls: use pipeline.aggregate_insights + get_cached_hot_topics etc (already wired in src/pipeline).

# Keep the original load helpers EXACTLY for test_dashboard.py (test_load_sample_data)
# The synthetic part can remain (or be deprecated later) – tests rely on it.


def load_report(path: str) -> dict[str, Any]:
    """Load a JSON report file. (Preserved for backward compat & tests.)"""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sample_data() -> dict[str, Any]:
    """Load sample/baseline data for demo. (Preserved EXACTLY for tests/test_dashboard.py)."""
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


# End of file – all requirements from plan executed (data services, refactor SPA,
# Call Detail full impl, actions, filters, perf, tests compat, comments for migration/API).
