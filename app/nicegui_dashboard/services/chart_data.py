"""Chart data extraction and Plotly figure builders for analytics tab.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (Plotly trajectory & trends)
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.services.data_services import _get_sentiment_score, get_hot_topics, get_overall_sentiment

_POSITIVE_HINTS = ("tack", "bra", "hjälp", "upplösning", "glad", "nöjd")
_NEGATIVE_HINTS = ("arg", "less", "skandal", "frustrer", "upprörd", "chef", "arn")


def _effective_segment_score(segment: dict[str, Any], sentiment: dict[str, Any]) -> float:
    """Sentiment score for charts; light keyword fallback when pipeline scores are flat."""
    score = _get_sentiment_score(sentiment)
    label = str(sentiment.get("label", "")).lower()
    if label not in ("neutral", "") or abs(score) > 0.05:
        if 0.0 <= score <= 1.0 and label == "neutral" and score == 0.5:
            pass
        else:
            return score
    text = str(segment.get("text", "")).lower()
    if any(h in text for h in _NEGATIVE_HINTS):
        return -0.65
    if any(h in text for h in _POSITIVE_HINTS):
        return 0.7
    return score


def extract_trajectory_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-segment customer sentiment trajectory for one call."""
    call_id = report.get("call_id") or report.get("id", "?")
    segments = report.get("segments") or []
    sentiments = report.get("sentiment_results") or []
    llm_traj = (report.get("llm") or {}).get("trajectory") or {}
    llm_points = llm_traj.get("points") or llm_traj.get("customer_sentiment") or []

    if isinstance(llm_points, list) and llm_points:
        points: list[dict[str, Any]] = []
        for i, pt in enumerate(llm_points):
            if not isinstance(pt, dict):
                continue
            points.append(
                {
                    "x": pt.get("turn", pt.get("index", i)),
                    "y": float(pt.get("score", pt.get("sentiment", 0))),
                    "call_id": call_id,
                    "label": pt.get("label", f"Turn {i + 1}"),
                }
            )
        if points:
            return points

    points = []
    for i, seg in enumerate(segments):
        sent = sentiments[i] if i < len(sentiments) else {}
        x = float(seg.get("start", i * 10))
        points.append(
            {
                "x": x,
                "y": _effective_segment_score(seg, sent),
                "call_id": call_id,
                "label": f"{seg.get('speaker', '?')} @ {int(x)}s",
            }
        )
    return points


def list_call_options(reports: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Dropdown options for trajectory call selector."""
    return [
        {
            "label": f"{r.get('call_id', '?')} – {r.get('title', '')}",
            "value": str(r.get("call_id") or r.get("id", "")),
        }
        for r in reports
        if r.get("call_id") or r.get("id")
    ]


def extract_agent_trend_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-call empathy and QA scores for agent trend charts."""
    rows: list[dict[str, Any]] = []
    for idx, report in enumerate(reports, start=1):
        call_id = str(report.get("call_id") or report.get("id", f"CALL-{idx}"))
        agent = (report.get("meta") or {}).get("agent", "Okänd")
        results = report.get("results") or {}
        llm = report.get("llm") or {}

        ap = results.get("agent_performance") or {}
        agent_block = ap.get("agent") if isinstance(ap.get("agent"), dict) else ap
        empathy = None
        if isinstance(agent_block, dict):
            empathy = agent_block.get("empathy_score")

        assess = results.get("agent_assessment") or llm.get("agent_assessment") or {}
        if empathy is None and isinstance(assess, dict):
            empathy = assess.get("empathy_score")

        qa = (results.get("qa") or results.get("compliance_qa") or {}).get("overall_qa_score")

        if empathy is None:
            segments = report.get("segments") or []
            sents = report.get("sentiment_results") or []
            if segments:
                scores = [
                    _effective_segment_score(segments[i], sents[i] if i < len(sents) else {})
                    for i in range(len(segments))
                ]
                empathy = round((sum(scores) / len(scores) + 1) / 2, 2)

        if qa is None and empathy is not None:
            qa = round(float(empathy) * 100, 0)

        alerts = results.get("alerts") or []
        risk = str((report.get("risks") or {}).get("risk_level", "low")).lower()
        escalation = len(alerts) if isinstance(alerts, list) else 0
        if escalation == 0 and risk in ("high", "critical", "medium"):
            escalation = 1

        rows.append(
            {
                "index": idx,
                "call_id": call_id,
                "title": report.get("title", call_id),
                "agent": agent,
                "empathy": empathy,
                "qa": qa,
                "escalation": escalation,
            }
        )
    return rows


def _rolling_average(values: list[float], window: int = 3) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(round(sum(chunk) / len(chunk), 3))
    return out


def build_trajectory_figure(report: dict[str, Any] | None) -> go.Figure:
    """Line chart: customer sentiment over call timeline."""
    fig = go.Figure()
    if not report:
        fig.update_layout(title="Kundsentiment – inget samtal valt", height=320)
        return fig

    points = extract_trajectory_points(report)
    call_id = report.get("call_id", "?")
    title = report.get("title", call_id)
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="Segment",
            customdata=[[p["call_id"], p.get("label", "")] for p in points],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Tid/tur: %{x}<br>"
                "Sentiment: %{y:.2f}<br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
            line={"color": "#42a5f5", "width": 2},
            marker={"size": 7},
        )
    )
    if len(ys) >= 3:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=_rolling_average(ys),
                mode="lines",
                name="Glidande medel (3)",
                line={"color": "#ffb74d", "width": 2, "dash": "dot"},
                hovertemplate="Medel: %{y:.2f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    if ys:
        fig.add_hrect(y0=min(ys), y1=max(ys), fillcolor="rgba(66,165,245,0.08)", line_width=0)
    fig.update_layout(
        title=f"Kundsentiment – {call_id}",
        xaxis_title="Tid (s) / tur",
        yaxis_title="Sentiment (−1 … +1)",
        height=340,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        hovermode="closest",
        legend={"orientation": "h", "y": 1.15},
    )
    fig.add_annotation(
        text=title,
        xref="paper",
        yref="paper",
        x=0,
        y=1.12,
        showarrow=False,
        font={"size": 11, "color": "#9e9e9e"},
        align="left",
    )
    return fig


def build_agent_trends_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Dual-metric agent performance across calls."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not rows:
        fig.update_layout(title="Agentprestanda – ingen data", height=340)
        return fig

    x_labels = [r["call_id"] for r in rows]
    meta = [[r["call_id"], r.get("agent", ""), r.get("title", "")] for r in rows]
    empathy_vals = [r.get("empathy") for r in rows]
    qa_vals = [r.get("qa") for r in rows]
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=empathy_vals,
            mode="lines+markers",
            name="Empati",
            customdata=meta,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[2]}<br>"
                "Agent: %{customdata[1]}<br>"
                "Empati: %{y:.2f}"
                "<extra></extra>"
            ),
            line={"color": "#66bb6a"},
            marker={"size": 9},
        ),
        secondary_y=False,
    )
    empathy_present = [v for v in empathy_vals if v is not None]
    if empathy_present:
        avg_emp = sum(empathy_present) / len(empathy_present)
        fig.add_hline(
            y=avg_emp,
            line_dash="dash",
            line_color="#66bb6a",
            opacity=0.4,
            annotation_text=f"Snitt empati {avg_emp:.2f}",
            annotation_position="top left",
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=qa_vals,
            mode="lines+markers",
            name="QA-score",
            customdata=meta,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[2]}<br>"
                "Agent: %{customdata[1]}<br>"
                "QA: %{y:.0f}"
                "<extra></extra>"
            ),
            line={"color": "#ffa726", "dash": "dash"},
            marker={"size": 8},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Agentprestanda per samtal (empati & QA)",
        height=360,
        margin={"l": 40, "r": 40, "t": 50, "b": 60},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Empati (0–1)", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="QA-poäng", secondary_y=True, range=[0, 100])
    fig.update_xaxes(tickangle=-30)
    return fig


def build_hot_topics_figure(reports: list[dict[str, Any]]) -> go.Figure:
    """Bar chart of hot topics volume."""
    from collections import Counter

    topics = get_hot_topics(reports, top_k=8)
    if not topics:
        categories: Counter[str] = Counter()
        for report in reports:
            meta = report.get("meta") or {}
            cat = meta.get("category") or meta.get("topic") or report.get("title", "övrigt")
            categories[str(cat)] += 1
        topics = [{"topic": t, "volume": c} for t, c in categories.most_common(8)]
    fig = go.Figure()
    if not topics:
        fig.update_layout(title="Heta ämnen – ingen data", height=300)
        return fig

    total = sum(t["volume"] for t in topics) or 1
    fig.add_trace(
        go.Bar(
            x=[t["volume"] for t in topics],
            y=[t["topic"] for t in topics],
            orientation="h",
            marker={"color": "#ab47bc"},
            customdata=[[t["topic"], round(t["volume"] / total * 100)] for t in topics],
            hovertemplate="%{y}<br>%{x} samtal (%{customdata[1]}%)<extra></extra>",
            text=[f"{t['volume']}" for t in topics],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Heta ämnen – fördelning",
        xaxis_title="Antal samtal",
        height=300,
        margin={"l": 120, "r": 20, "t": 40, "b": 40},
    )
    return fig


def build_escalation_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Escalation / alert count per call."""
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="Eskalering – ingen data", height=300)
        return fig

    colors = ["#ef5350" if r.get("escalation", 0) > 0 else "#90a4ae" for r in rows]
    fig.add_trace(
        go.Bar(
            x=[r["call_id"] for r in rows],
            y=[r.get("escalation", 0) for r in rows],
            marker={"color": colors},
            customdata=[
                [r["call_id"], r.get("title", ""), r.get("agent", "")]
                for r in rows
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Agent: %{customdata[2]}<br>"
                "Aviseringar: %{y}"
                "<extra></extra>"
            ),
        )
    )
    total_alerts = sum(r.get("escalation", 0) for r in rows)
    fig.update_layout(
        title=f"Eskalering & aviseringar per samtal (totalt {total_alerts})",
        yaxis_title="Antal aviseringar / risk",
        height=300,
        margin={"l": 40, "r": 20, "t": 40, "b": 60},
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def compute_sentiment_distribution(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count calls per overall sentiment label."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for report in reports:
        label = get_overall_sentiment(report).get("label", "neutral")
        display = "Positiv" if "pos" in str(label).lower() else (
            "Negativ" if "neg" in str(label).lower() else "Neutral"
        )
        counter[display] += 1
    total = max(1, len(reports))
    return [
        {"label": lbl, "count": cnt, "pct": round(cnt / total * 100)}
        for lbl, cnt in counter.most_common()
    ]


def build_sentiment_distribution_figure(reports: list[dict[str, Any]]) -> go.Figure:
    """Pie chart of overall call sentiment distribution."""
    dist = compute_sentiment_distribution(reports)
    fig = go.Figure()
    if not dist:
        fig.update_layout(title="Sentimentfördelning – ingen data", height=280)
        return fig

    colors = {
        "Positiv": "#66bb6a",
        "Neutral": "#90a4ae",
        "Negativ": "#ef5350",
    }
    fig.add_trace(
        go.Pie(
            labels=[d["label"] for d in dist],
            values=[d["count"] for d in dist],
            hole=0.45,
            marker={"colors": [colors.get(d["label"], "#42a5f5") for d in dist]},
            customdata=[[d["pct"]] for d in dist],
            hovertemplate="%{label}<br>%{value} samtal (%{customdata[0]}%)<extra></extra>",
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Sentimentfördelning (alla samtal i urval)",
        height=280,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        showlegend=False,
    )
    return fig


def segment_index_from_trajectory_x(report: dict[str, Any] | None, x_value: float) -> int:
    """Map Plotly trajectory x-coordinate to nearest segment index."""
    if not report:
        return 0
    points = extract_trajectory_points(report)
    if not points:
        return 0
    segments = report.get("segments") or []
    if segments and points[0].get("x") == segments[0].get("start", 0):
        best_idx = 0
        best_dist = float("inf")
        for i, seg in enumerate(segments):
            start = float(seg.get("start", i * 10))
            dist = abs(start - x_value)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx
    try:
        idx = int(round(x_value))
    except (TypeError, ValueError):
        idx = 0
    return max(0, min(idx, len(segments) - 1 if segments else idx))


def call_id_from_plotly_click(event: dict[str, Any]) -> str | None:
    """Extract call_id from NiceGUI plotly_click event args."""
    points = (event.get("points") or event.get("data", {}).get("points") or [])
    if not points:
        args = event if isinstance(event, dict) else {}
        if "points" in args:
            points = args["points"]
    if not points:
        return None
    point = points[0] if isinstance(points, list) else points
    if not isinstance(point, dict):
        return None
    custom = point.get("customdata")
    if custom:
        return str(custom[0] if isinstance(custom, (list, tuple)) else custom)
    return None