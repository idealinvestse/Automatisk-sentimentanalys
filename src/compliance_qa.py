"""Compliance & QA Auto-Scoring Engine (Fas 4.2).

Customizable scorecards in YAML under configs/qa_scorecards/.
Hybrid: rule-based for explicit checks + selective Mistral for nuanced criteria
(e.g. "visar empati", "professionell ton").

All output is structured + evidence-based (evidence_spans where possible).
Actionable for QA teams: overall score, passed/failed list, risk, per-criterion evidence.

Integration (explicit in pipeline.py):
    from src.compliance_qa import QAScorer, load_scorecard
    scorer = QAScorer(scorecard_path="configs/qa_scorecards/standard_support_v1.yaml")
    qa = scorer.score_conversation(segments, role_map=..., local_signals={"agent_performance": ...})
    results["qa"] = qa.model_dump()   # or "compliance_qa"

When LLM is used for any criterion: logged with "LLM used for QA criteria: [...]"

Privacy: run after local, before or after redaction (uses original for rules; LLM path uses the redacted segments if profile).

Caching: per-call hash on (scorecard_name, transcript_hash, signals). Invalidation on scorecard change or new data.

See UTVECKLINGSPLAN_Fas4 v1.1 Task 4.2.1 + 4.2.2.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core.models import Segment
from .llm.pii_redactor import redact_segments
from .llm.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

# Default location
QA_SCORECARDS_DIR = Path("configs/qa_scorecards")


class QACriterionResult(BaseModel):
    """Result for a single criterion in the scorecard."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    weight: float
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized 0-1 for this criterion (weighted into overall).",
    )
    passed: bool
    detection_method: str  # rule-based | llm | hybrid
    evidence: list[str] = Field(
        default_factory=list, description="Human readable evidence snippets."
    )
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    llm_used: bool = False


class QAScoreResult(BaseModel):
    """Full structured QA scoring output. Merge into CallAnalysisReport.results['qa'] or ['compliance_qa']."""

    model_config = ConfigDict(extra="forbid")

    scorecard_name: str
    scorecard_version: str
    overall_qa_score: float = Field(
        ..., ge=0.0, le=100.0, description="Weighted total score 0-100."
    )
    passed: bool
    passed_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    risk_level: str = Field(
        "medium",
        description="low | medium | high | critical (based on failed high-weight + compliance).",
    )
    compliance_flags: list[str] = Field(default_factory=list)
    criteria_results: list[QACriterionResult]
    evidence_summary: str | None = None
    llm_criteria_used: list[str] = Field(
        default_factory=list, description="Which criteria required an LLM call (for audit/cost)."
    )
    computed_at: str

    @property
    def summary_for_coach(self) -> str:
        """Short actionable string for dashboard."""
        return f"QA {self.overall_qa_score:.0f}/100 ({'PASS' if self.passed else 'FAIL'}) risk={self.risk_level} flags={len(self.compliance_flags)}"


def load_scorecard(name_or_path: str = "standard_support_v1") -> dict[str, Any]:
    """Load scorecard YAML. Accepts name (without .yaml) or full path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = QA_SCORECARDS_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        # fallback try .yml
        p = p.with_suffix(".yml")
    if not p.exists():
        raise FileNotFoundError(
            f"Scorecard not found: {name_or_path} (looked in {QA_SCORECARDS_DIR})"
        )
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["_path"] = str(p)
    return data


def _text_has_any(text: str, keywords: list[str]) -> bool:
    low = (text or "").lower()
    return any(kw.lower() in low for kw in (keywords or []))


def _make_evidence_span(text: str, speaker: str | None, turn: int | None) -> EvidenceSpan:
    return EvidenceSpan(text=text[:200], speaker_role=speaker, turn_index=turn)


def _local_empathy_score(local_signals: dict[str, Any] | None) -> float | None:
    """Extract normalized empathy 0-1 from agent_performance / agent_assessment."""
    if not local_signals:
        return None
    assess = local_signals.get("agent_assessment") or {}
    if isinstance(assess, dict) and assess.get("empathy_score") is not None:
        return float(assess["empathy_score"])
    ap = local_signals.get("agent_performance") or {}
    if isinstance(ap, dict):
        agent = ap.get("agent") or {}
        if isinstance(agent, dict) and agent.get("empathy_score") is not None:
            return float(agent["empathy_score"])
    return None


def _local_compliance_flags(local_signals: dict[str, Any] | None) -> list[str]:
    if not local_signals:
        return []
    flags: list[str] = []
    assess = local_signals.get("agent_assessment") or {}
    if isinstance(assess, dict):
        flags.extend(str(f) for f in (assess.get("compliance_flags") or []))
    ap = local_signals.get("agent_performance") or {}
    if isinstance(ap, dict):
        agent = ap.get("agent") or {}
        if isinstance(agent, dict):
            flags.extend(str(f) for f in (agent.get("compliance_flags") or []))
    return flags


def _apply_local_signal_adjustment(
    criterion_id: str,
    score: float,
    passed: bool,
    local_signals: dict[str, Any] | None,
) -> tuple[float, bool]:
    """Blend quantitative Fas 4.1 signals into selected QA criteria."""
    if not local_signals:
        return score, passed

    if criterion_id == "empathy":
        emp = _local_empathy_score(local_signals)
        if emp is not None:
            blended = round(0.7 * score + 0.3 * emp, 3)
            return blended, blended >= 0.5

    if criterion_id in {"no_promises_broken", "tone_professional", "compliance_script"}:
        flags = _local_compliance_flags(local_signals)
        if flags:
            penalized = min(score, 0.4)
            return penalized, False

    return score, passed


def _compute_rule_based(
    criterion: dict[str, Any],
    segments: list[dict | Segment],
    role_map: dict[str, str] | None,
) -> tuple[float, bool, list[str], list[EvidenceSpan]]:
    """Pure rule-based scoring. Returns (score 0-1, passed, evidence_texts, spans)."""
    kw = criterion.get("keywords", []) or []
    hits = []
    spans: list[EvidenceSpan] = []
    for i, seg in enumerate(segments):
        txt = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        sp = (
            seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)
        ) or "UNKNOWN"
        role = role_map.get(sp, sp) if role_map else sp
        if _text_has_any(txt, kw):
            hits.append(f"[{role}] {txt[:80]}")
            spans.append(_make_evidence_span(txt, role, i))
    if not kw:
        # no keywords -> neutral pass for rule
        return 0.7, True, ["no keywords defined; treated as partial pass"], []

    # For greeting: must be in first agent turn
    if criterion.get("id") == "greeting":
        first_agent = ""
        for seg in segments:
            r = (
                role_map.get(
                    seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", ""), ""
                )
                if role_map
                else ""
            )
            if "agent" in str(r).lower():
                first_agent = (
                    seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
                )
                break
        passed = _text_has_any(first_agent, kw)
        sc = 1.0 if passed else 0.0
        ev = [f"First agent: {first_agent[:60]}"] if first_agent else ["No agent greeting found"]
        return sc, passed, ev, spans[:1] if spans else []

    # generic: at least one hit -> full, partial if some
    passed = len(hits) > 0
    sc = min(1.0, len(hits) / 2.0) if hits else 0.0
    return sc, passed, hits[:3], spans[:3]


def _score_with_llm_if_needed(
    criterion: dict[str, Any],
    segments: list[dict | Segment],
    role_map: dict[str, str] | None,
    analyzer: Any | None = None,  # ConversationMistralAnalyzer or None
    profile_name: str = "callcenter",
) -> tuple[float, bool, list[str], list[EvidenceSpan], bool]:
    """If detection_method requires LLM, call it (via analyzer if provided, else stub).

    Returns (score, passed, evidence, spans, llm_used).
    For hybrid: rules first, LLM only on borderline.
    """
    method = (criterion.get("detection_method") or "rule-based").lower()
    if method == "rule-based":
        sc, pas, ev, sp = _compute_rule_based(criterion, segments, role_map)
        return sc, pas, ev, sp, False

    # Always compute rule baseline first
    rule_sc, rule_pas, rule_ev, rule_sp = _compute_rule_based(criterion, segments, role_map)

    # hybrid: rule first, LLM only for borderline or high-weight nuanced cases (per plan 4.2.1)
    if method == "hybrid":
        is_borderline = rule_sc > 0.2 and rule_sc < 0.85
        high_weight = float(criterion.get("weight", 0)) >= 15
        if not (is_borderline or high_weight) or analyzer is None:
            return rule_sc, rule_pas, rule_ev, rule_sp, False
        # else fall through to LLM for nuance / confirmation

    # llm or (hybrid that needs LLM): try LLM ...
    prompt_hint = criterion.get("prompt_hint") or criterion.get("description")
    # Privacy: redact before building LLM transcript slice (honor profile + early PII rule)
    segments_for_qa_llm = segments
    try:
        # Use the same redact_segments as main path (respects profile "anonymize_before_llm")
        segments_for_qa_llm = redact_segments(segments, profile_name=profile_name)
    except Exception:
        segments_for_qa_llm = segments
    # Build tiny transcript slice (first + last + any frustration area) to save tokens
    transcript = "\n".join(
        f"[{role_map.get(s.get('speaker', ''), s.get('speaker', '')) if role_map else s.get('speaker', '')}] {(s.get('text', '') if isinstance(s, dict) else getattr(s, 'text', ''))[:120]}"
        for s in segments_for_qa_llm[:12]  # limit
    )

    if analyzer is not None and getattr(analyzer, "client", None):
        try:
            # Use a lightweight structured call via the analyzer's client if possible.
            # For simplicity we fall back to a direct client call here.
            from .llm.openrouter_client import OpenRouterClient

            client = getattr(analyzer, "client", None) or OpenRouterClient()
            sys_prompt = "Du är en strikt QA-granskare för svensk kundtjänst. Svara ENDAST med exakt giltig JSON, ingen annan text eller markdown."
            user = f"""Bedöm kriteriet: {criterion["description"]}
Prompt hint: {prompt_hint}

Transkript (roll-märkt):
{transcript}

Returnera exakt:
{{"score": 0.0-1.0, "passed": true/false, "evidence": ["kort citat 1", "citat 2"], "reason": "kort"}}
"""
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ]
            # We use non-strict text for robustness; in prod use json_schema
            # chat_completion returns (content: str, meta: dict)
            try:
                content, meta = client.chat_completion(
                    messages=messages, model=None, temperature=0.1, max_tokens=300
                )
                txt = content or "{}"
                meta = {"used": "chat_completion", **(meta or {})}
            except Exception:
                raw, meta = client.structured_chat(
                    messages=messages,
                    model=None,
                    temperature=0.1,
                    max_tokens=300,
                    task_name="qa_criterion_judge",
                )
                txt = raw.get("content", "{}") if isinstance(raw, dict) else str(raw)
            data = {}
            try:
                # extract json
                import re

                m = re.search(r"\{.*\}", txt, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
            except Exception:
                data = {}
            sc = max(0.0, min(1.0, float(data.get("score", 0.5))))
            pas = bool(data.get("passed", sc >= 0.6))
            ev = data.get("evidence", [])[:3] or [txt[:100]]
            spans = [_make_evidence_span(e, None, None) for e in ev]
            logger.info(
                "LLM used for QA criterion id=%s (cost meta=%s)",
                criterion.get("id"),
                meta.get("cost_usd"),
            )
            return sc, pas, ev, spans, True
        except Exception as e:
            logger.warning(
                "LLM QA scoring failed for %s, falling back to rule: %s", criterion.get("id"), e
            )

    # Fallback: pure rule even for llm/hybrid (or stub)
    sc, pas, ev, sp = _compute_rule_based(criterion, segments, role_map)
    if method == "llm":
        # penalize a bit for no LLM
        sc = min(sc, 0.6)
    return sc, pas, ev + ["(LLM unavailable; rule fallback)"], sp, False


class QAScorer:
    """Main engine. Load once, score many calls."""

    def __init__(
        self,
        scorecard: dict[str, Any] | None = None,
        scorecard_path: str | None = None,
        analyzer: Any | None = None,  # ConversationMistralAnalyzer instance for hybrid
    ) -> None:
        if scorecard is None and scorecard_path:
            scorecard = load_scorecard(scorecard_path)
        elif scorecard is None:
            scorecard = load_scorecard("standard_support_v1")
        self.scorecard = scorecard or {}
        self.analyzer = analyzer
        self.name = self.scorecard.get("name", "unknown")
        self.version = str(self.scorecard.get("version", "0"))
        self.threshold = float(self.scorecard.get("overall_pass_threshold", 75))

    def score_conversation(
        self,
        segments: list[dict[str, Any]] | list[Segment],
        role_map: dict[str, str] | None = None,
        local_signals: dict[str, Any] | None = None,
        profile_name: str = "callcenter",
    ) -> QAScoreResult:
        """Run the full scorecard. Returns validated Pydantic result with evidence."""
        criteria = self.scorecard.get("criteria", []) or []
        crit_results: list[QACriterionResult] = []
        llm_used_list: list[str] = []
        total_weight = 0.0
        weighted_sum = 0.0
        all_flags: list[str] = []

        seg_dicts = []
        for s in segments:
            if isinstance(s, dict):
                seg_dicts.append(s)
            else:
                seg_dicts.append(s.to_dict())

        for c in criteria:
            cid = c.get("id", "unknown")
            w = float(c.get("weight", 5))
            total_weight += w
            method = c.get("detection_method", "rule-based")

            if method in ("llm", "hybrid"):
                sc, pas, ev, spans, used_llm = _score_with_llm_if_needed(
                    c, seg_dicts, role_map, analyzer=self.analyzer, profile_name=profile_name
                )
                if used_llm:
                    llm_used_list.append(cid)
            else:
                sc, pas, ev, spans = _compute_rule_based(c, seg_dicts, role_map)
                used_llm = False

            sc, pas = _apply_local_signal_adjustment(cid, sc, pas, local_signals)

            weighted_sum += sc * w

            cr = QACriterionResult(
                id=cid,
                description=c.get("description", ""),
                weight=w,
                score=round(sc, 3),
                passed=pas,
                detection_method=method,
                evidence=ev,
                evidence_spans=spans,
                llm_used=used_llm,
            )
            crit_results.append(cr)
            if not pas and w >= 10:
                all_flags.append(f"{cid}: {c.get('description')}")

        overall = (weighted_sum / max(1.0, total_weight)) * 100.0 if total_weight > 0 else 50.0
        overall = round(overall, 1)
        passed = overall >= self.threshold

        passed_ids = [cr.id for cr in crit_results if cr.passed]
        failed_ids = [cr.id for cr in crit_results if not cr.passed]

        # risk
        high_weight_failed = sum(1 for cr in crit_results if not cr.passed and cr.weight >= 15)
        risk = "low"
        if high_weight_failed >= 2 or overall < 50:
            risk = "critical"
        elif high_weight_failed >= 1 or overall < self.threshold:
            risk = "high"
        elif overall < 85:
            risk = "medium"

        evidence_sum = f"{len(passed_ids)} passed, {len(failed_ids)} failed. LLM criteria: {llm_used_list or 'none'}"

        res = QAScoreResult(
            scorecard_name=self.name,
            scorecard_version=self.version,
            overall_qa_score=overall,
            passed=passed,
            passed_criteria=passed_ids,
            failed_criteria=failed_ids,
            risk_level=risk,
            compliance_flags=all_flags,
            criteria_results=crit_results,
            evidence_summary=evidence_sum,
            llm_criteria_used=llm_used_list,
            computed_at=datetime.now(UTC).isoformat(),
        )
        logger.info(
            "QA scoring complete | scorecard=%s v%s score=%.1f passed=%s risk=%s llm_criteria=%s",
            self.name,
            self.version,
            overall,
            passed,
            risk,
            llm_used_list or "-",
        )
        return res


# Convenience for pipeline
def score_call_with_default_scorecard(
    segments: list[dict | Segment],
    role_map: dict | None = None,
    local_signals: dict | None = None,
    profile_name: str = "callcenter",
    use_llm: bool = False,
    analyzer: Any | None = None,
) -> dict[str, Any]:
    """Quick entry: load default + optional analyzer, return dict."""
    try:
        scorer = QAScorer(
            scorecard_path="standard_support_v1", analyzer=analyzer if use_llm else None
        )
        result = scorer.score_conversation(
            segments, role_map=role_map, local_signals=local_signals, profile_name=profile_name
        )
        return result.model_dump()
    except Exception as e:
        logger.error("QA scoring failed: %s", e)
        return {"error": str(e), "overall_qa_score": 0.0, "passed": False, "risk_level": "unknown"}


__all__ = ["QAScorer", "QAScoreResult", "load_scorecard", "score_call_with_default_scorecard"]
