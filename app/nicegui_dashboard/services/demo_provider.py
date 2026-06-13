"""Demo data provider for NiceGUI dashboard.

Fas 3 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
Loads reports via nicegui_api_client (/analyze_pipeline) with local fallback.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.nicegui_dashboard.services.qa_display import qa_score_css_class
from app.services.data_services import (
    _generate_fallback_reports,
    get_demo_transcripts,
    get_overall_sentiment,
)

logger = logging.getLogger(__name__)


def _enrich_report(rdict: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    rdict["call_id"] = transcript.get("id", "UNKNOWN")
    rdict["title"] = transcript.get("title", rdict["call_id"])
    rdict["meta"] = transcript.get("meta", {})
    rdict.setdefault("demo_meta", {})["transcript_id"] = transcript["id"]
    return rdict


def _run_pipeline_on_transcripts(
    transcripts: list[dict[str, Any]],
    *,
    use_llm: bool = False,
    profile: str = "callcenter",
    llm_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Run CallAnalysisPipeline on canned transcripts (local path)."""
    from src.pipeline import CallAnalysisPipeline

    reports: list[dict[str, Any]] = []
    for t in transcripts:
        try:
            pipe = CallAnalysisPipeline(
                profile=profile,
                use_mistral_llm=use_llm,
                deep_analysis=use_llm,
                llm_api_key=llm_api_key,
            )
            report = pipe.analyze_segments(t["segments"])
            rdict: dict[str, Any] = report.to_dict()
        except Exception as run_err:
            logger.warning("Pipeline error on %s: %s", t.get("id"), run_err)
            rdict = {
                "segments": t["segments"],
                "sentiment_results": [{"label": "neutral", "score": 0.5} for _ in t["segments"]],
                "intent_results": [],
                "results": {},
                "llm": {},
                "risks": {},
            }
        reports.append(_enrich_report(rdict, t))
    return reports


async def load_reports_from_api(
    client: Any,
    *,
    use_llm: bool = False,
) -> list[dict[str, Any]]:
    """Load demo reports via POST /analyze_pipeline for each canned transcript."""
    from app.nicegui_dashboard.services.nicegui_api_client import APIError

    transcripts = get_demo_transcripts()
    reports: list[dict[str, Any]] = []
    for t in transcripts:
        try:
            rdict = await client.analyze_pipeline(
                t["segments"],
                use_mistral_llm=use_llm,
                deep_analysis=use_llm,
            )
            reports.append(_enrich_report(rdict, t))
            logger.info("API report loaded for %s", t.get("id"))
        except APIError as err:
            logger.warning("API error on %s: %s", t.get("id"), err)
            raise
        except Exception as err:
            logger.warning("Unexpected API error on %s: %s", t.get("id"), err)
            raise
    return reports


@lru_cache(maxsize=8)
def load_demo_reports(
    use_llm: bool = False,
    profile: str = "callcenter",
    *,
    use_pipeline: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Load demo reports locally with lru_cache (sync fallback path)."""
    transcripts = get_demo_transcripts()
    if use_pipeline:
        try:
            reports = _run_pipeline_on_transcripts(transcripts, use_llm=use_llm, profile=profile)
            return tuple(reports)
        except Exception as err:
            logger.warning("Pipeline unavailable, using fallback reports: %s", err)
    reports = _generate_fallback_reports(transcripts)
    return tuple(reports)


def reports_to_table_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map CallAnalysisReport dicts to ui.table row format."""
    rows: list[dict[str, Any]] = []
    for r in reports:
        call_id = r.get("call_id") or r.get("id", "UNKNOWN")
        meta = r.get("meta") or {}
        qa = (r.get("results") or {}).get("qa") or {}
        qa_score = qa.get("overall_qa_score")
        sentiment = get_overall_sentiment(r)
        display_score = qa_score if qa_score is not None else "—"
        rows.append(
            {
                "call_id": call_id,
                "title": r.get("title", call_id),
                "agent": meta.get("agent", "Okänd"),
                "sentiment": sentiment.get("label", "neutral"),
                "qa_score": display_score,
                "qa_class": qa_score_css_class(display_score),
            }
        )
    return rows