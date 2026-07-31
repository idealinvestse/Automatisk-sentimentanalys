"""Paid-model advisor: recommend models per analysis perspective with cost optimization.

Loads live catalogs (OpenRouter + native providers), scores paid models against
analysis-perspective profiles (sentiment, root-cause, coaching, …), and returns
simple selectable options with clear cost/quality tradeoffs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model_catalog import load_catalog, load_provider_catalog
from .provider_secrets import load_provider_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Analysis perspectives (selectable profiles)
# ---------------------------------------------------------------------------

# cost_priority: 0.0 = ignore cost, 1.0 = minimize cost hard
# quality_priority: 0.0–1.0 relative weight for quality/reasoning
ANALYSIS_PERSPECTIVES: dict[str, dict[str, Any]] = {
    "cost_saver": {
        "label": "Kostnadssparare",
        "label_en": "Cost saver",
        "description": "Billigast möjliga paid-modell som fortfarande klarar svensk callcenter-text.",
        "icon": "wallet",
        "cost_priority": 0.9,
        "quality_priority": 0.25,
        "max_usd_per_m_blended": 0.5,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 8000,
        "tasks": ["summary", "sentiment", "intent"],
        "use_when": "Batch, nattjobb, hög volym",
    },
    "batch_throughput": {
        "label": "Batch / volym",
        "label_en": "Batch throughput",
        "description": "Snabb och billig för många korta samtal. Prioriterar latency + lågt pris.",
        "icon": "gauge",
        "cost_priority": 0.75,
        "quality_priority": 0.35,
        "max_usd_per_m_blended": 1.0,
        "prefer_swedish": True,
        "prefer_eu": False,
        "min_context": 8000,
        "tasks": ["sentiment", "intent", "summary"],
        "use_when": "Massanalys, köer, nightly eval",
    },
    "sentiment_refine": {
        "label": "Sentiment (fördjupad)",
        "label_en": "Sentiment refine",
        "description": "Nyanserad känsla/polarity utöver lokala XLM-R-resultat.",
        "icon": "heart",
        "cost_priority": 0.55,
        "quality_priority": 0.55,
        "max_usd_per_m_blended": 3.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 16000,
        "tasks": ["sentiment", "emotion", "trajectory"],
        "use_when": "När lokal sentiment är osäker eller sarkasm misstänks",
    },
    "intent_routing": {
        "label": "Intent / ärenderouting",
        "label_en": "Intent routing",
        "description": "Klassificera ärendetyp och nästa steg (tillstånd, faktura, teknik …).",
        "icon": "route",
        "cost_priority": 0.6,
        "quality_priority": 0.5,
        "max_usd_per_m_blended": 2.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 16000,
        "tasks": ["intent", "topics"],
        "use_when": "Triage, automatisk köstyrning",
    },
    "root_cause": {
        "label": "Rotorsak",
        "label_en": "Root cause",
        "description": "Hitta verklig orsak bakom klagomål — kräver resonemang över flera turer.",
        "icon": "search",
        "cost_priority": 0.35,
        "quality_priority": 0.8,
        "max_usd_per_m_blended": 12.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 32000,
        "tasks": ["root_cause", "trajectory"],
        "use_when": "Eskalerade / komplexa samtal",
    },
    "coaching_qa": {
        "label": "QA / coachning",
        "label_en": "QA coaching",
        "description": "Agentbedömning, empathy, konkreta coachningspunkter med evidence.",
        "icon": "graduation",
        "cost_priority": 0.3,
        "quality_priority": 0.85,
        "max_usd_per_m_blended": 15.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 32000,
        "tasks": ["agent_assessment", "actionable_summary", "active_listening"],
        "use_when": "Kvalitetssäkring, utbildning",
    },
    "compliance_risk": {
        "label": "Compliance / risk",
        "label_en": "Compliance risk",
        "description": "Policybrott, PII-läckage-signaler, eskalationsrisk — precision framför pris.",
        "icon": "shield",
        "cost_priority": 0.25,
        "quality_priority": 0.9,
        "max_usd_per_m_blended": 20.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 32000,
        "tasks": ["compliance_risk", "risks"],
        "use_when": "Reglerade brancher, revision",
    },
    "summary_actions": {
        "label": "Sammanfattning + actions",
        "label_en": "Summary & actions",
        "description": "Kort call summary, outcome och action items på svenska.",
        "icon": "list",
        "cost_priority": 0.5,
        "quality_priority": 0.6,
        "max_usd_per_m_blended": 4.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 16000,
        "tasks": ["summary", "actionable_summary"],
        "use_when": "CRM-notering, efterarbete",
    },
    "swedish_quality": {
        "label": "Svenska (kvalitet)",
        "label_en": "Swedish quality",
        "description": "Bästa svenska språkkvalitet bland paid-modeller (EU-first när möjligt).",
        "icon": "flag",
        "cost_priority": 0.4,
        "quality_priority": 0.75,
        "max_usd_per_m_blended": 12.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 32000,
        "tasks": ["full_holistic_call_analysis"],
        "use_when": "Kundnära texter, känsliga toner",
    },
    "holistic_deep": {
        "label": "Holistisk djupanalys",
        "label_en": "Holistic deep",
        "description": "Full call-analys: trajectory, root cause, coaching, risks i ett svep.",
        "icon": "brain",
        "cost_priority": 0.2,
        "quality_priority": 0.95,
        "max_usd_per_m_blended": 25.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 64000,
        "tasks": [
            "trajectory",
            "root_cause",
            "actionable_summary",
            "agent_assessment",
            "emotion_trajectory",
        ],
        "use_when": "VIP-samtal, pilot-QA, svåra cases",
    },
    "balanced_ops": {
        "label": "Balanserad drift",
        "label_en": "Balanced ops",
        "description": "Standardval för produktion: bra kvalitet till rimlig kostnad.",
        "icon": "scale",
        "cost_priority": 0.5,
        "quality_priority": 0.65,
        "max_usd_per_m_blended": 5.0,
        "prefer_swedish": True,
        "prefer_eu": True,
        "min_context": 32000,
        "tasks": ["full_holistic_call_analysis"],
        "use_when": "Default för daglig drift",
    },
    "premium_reasoning": {
        "label": "Premium resonemang",
        "label_en": "Premium reasoning",
        "description": "Högsta resonemangskvalitet oavsett pris (inom rimlig max).",
        "icon": "sparkles",
        "cost_priority": 0.1,
        "quality_priority": 1.0,
        "max_usd_per_m_blended": 40.0,
        "prefer_swedish": False,
        "prefer_eu": False,
        "min_context": 64000,
        "tasks": ["root_cause", "agent_assessment_detailed", "full_holistic_call_analysis"],
        "use_when": "Enstaka kritiska samtal",
    },
}


# Soft preferences for Swedish / EU / known-good families (id substrings)
_SV_BOOST = (
    "mistral",
    "mistralai",
    "nemo",
    "voss",
    "scandi",
    "nordic",
    "kb-whisper",  # unlikely in LLM catalog but harmless
)
_EU_HOST_HINTS = ("mistral", "mistralai", "scaleway", "ovh", "aleph", "cohere")  # cohere soft
_QUALITY_SIZE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    # Explicit weak first so nano/mini don't inherit "gpt-5" boost
    (re.compile(r"\b(3b|1b|tiny|micro|nano)\b", re.I), 0.3),
    (re.compile(r"\b(small|8b|7b|9b|mini|flash|nemo)\b", re.I), 0.5),
    (re.compile(r"\b(medium|32b|34b|27b|22b)\b", re.I), 0.7),
    (re.compile(r"\b(large|70b|72b|405b|ultra|pro)\b", re.I), 0.85),
    (re.compile(r"\b(opus|o1|o3|gpt-5(?!.*nano)|sonnet-4|deepseek-r1)\b", re.I), 1.0),
]


@dataclass
class ModelCandidate:
    provider: str
    model_id: str
    name: str
    prompt_per_m_usd: float
    completion_per_m_usd: float
    blended_per_m_usd: float  # assume 40% prompt / 60% completion typical call analysis
    context_length: int | None
    is_free: bool
    description: str = ""
    quality_score: float = 0.5
    swedish_score: float = 0.0
    eu_score: float = 0.0
    est_cost_per_call_usd: float = 0.0  # ~4k in + 1.5k out tokens

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        # round money for UI
        for k in (
            "prompt_per_m_usd",
            "completion_per_m_usd",
            "blended_per_m_usd",
            "est_cost_per_call_usd",
            "quality_score",
            "swedish_score",
            "eu_score",
        ):
            d[k] = round(float(d[k]), 4)
        return d


@dataclass
class PerspectiveRecommendation:
    perspective_id: str
    label: str
    description: str
    icon: str
    use_when: str
    cost_priority: float
    quality_priority: float
    max_usd_per_m_blended: float
    tasks: list[str]
    recommended: dict[str, Any] | None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    provider_for_api: str = "openrouter"
    selectable: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.perspective_id,
            "label": self.label,
            "description": self.description,
            "icon": self.icon,
            "use_when": self.use_when,
            "cost_priority": self.cost_priority,
            "quality_priority": self.quality_priority,
            "max_usd_per_m_blended": self.max_usd_per_m_blended,
            "tasks": self.tasks,
            "recommended": self.recommended,
            "alternatives": self.alternatives,
            "score_breakdown": self.score_breakdown,
            "provider": self.provider_for_api,
            "selectable": self.selectable,
        }


def _blended_cost(prompt_m: float, completion_m: float, pin_ratio: float = 0.4) -> float:
    return prompt_m * pin_ratio + completion_m * (1.0 - pin_ratio)


def _est_call_cost(prompt_m: float, completion_m: float, in_tok: int = 4000, out_tok: int = 1500) -> float:
    return (in_tok / 1_000_000) * prompt_m + (out_tok / 1_000_000) * completion_m


def _quality_from_id(model_id: str, name: str = "") -> float:
    text = f"{model_id} {name}"
    for pat, score in _QUALITY_SIZE_PATTERNS:
        if pat.search(text):
            return score
    return 0.55


def _swedish_score(model_id: str, name: str = "", description: str = "") -> float:
    text = f"{model_id} {name} {description}".lower()
    score = 0.0
    for token in _SV_BOOST:
        if token in text:
            score += 0.35
    if "swedish" in text or "svenska" in text or "nordic" in text:
        score += 0.4
    # Mistral family is strong on Swedish in practice
    if "mistral" in text:
        score += 0.25
    return min(1.0, score)


def _eu_score(model_id: str, provider: str) -> float:
    text = model_id.lower()
    if provider == "mistral":
        return 1.0
    for h in _EU_HOST_HINTS:
        if h in text or h in provider:
            return 0.85
    if provider == "openrouter" and text.startswith("mistralai/"):
        return 0.9
    return 0.2


def _extract_pricing(m: dict[str, Any]) -> tuple[float, float]:
    pr = m.get("pricing") or {}
    pin = float(pr.get("prompt_per_million_usd") or 0.0)
    pout = float(pr.get("completion_per_million_usd") or 0.0)
    if pin == 0.0 and pout == 0.0:
        # raw per-token
        try:
            pin = float(pr.get("prompt") or 0.0) * 1_000_000
            pout = float(pr.get("completion") or 0.0) * 1_000_000
        except (TypeError, ValueError):
            pass
    return pin, pout


_SKIP_ID_PATTERNS = (
    re.compile(r"^openrouter/(auto|fusion|pareto|router)", re.I),
    re.compile(r":(free|batch)$", re.I),
    re.compile(r"\b(moderation|embedding|embed|tts|whisper|rerank)\b", re.I),
)


def _should_skip_model(model_id: str) -> bool:
    mid = model_id.strip()
    return any(p.search(mid) for p in _SKIP_ID_PATTERNS)


def collect_paid_candidates(
    *,
    providers: list[str] | None = None,
    include_free: bool = False,
    config: dict[str, Any] | None = None,
) -> list[ModelCandidate]:
    """Flatten catalogs into scored paid (or all) candidates."""
    cfg = config or load_provider_config()
    targets = providers or ["openrouter", "mistral", "nvidia", "cerebras"]
    out: list[ModelCandidate] = []

    for provider in targets:
        cat = load_provider_catalog(provider, cfg)
        if provider == "openrouter" and (not cat or not cat.get("models")):
            cat = load_catalog("data/openrouter_models_catalog.json")
        if not cat:
            continue
        for m in cat.get("models") or []:
            if not isinstance(m, dict) or not m.get("id"):
                continue
            mid = str(m["id"])
            if _should_skip_model(mid):
                continue
            pin, pout = _extract_pricing(m)
            # Negative / nonsense pricing (router meta models) → skip
            if pin < 0 or pout < 0 or pin > 500 or pout > 2000:
                continue
            is_free = bool(m.get("is_free")) or mid.endswith(":free") or (pin == 0.0 and pout == 0.0)
            if is_free and not include_free:
                # Native catalogs often lack pricing → treat curated non-free carefully
                if provider != "openrouter" and pin == 0.0 and pout == 0.0:
                    # unknown price native models: include as "paid-unknown" with mid cost prior
                    pin, pout = 0.5, 1.5
                    is_free = False
                else:
                    continue
            if pin == 0.0 and pout == 0.0 and not include_free:
                continue

            name = str(m.get("name") or mid)
            desc = str(m.get("description") or "")
            ctx = m.get("context_length")
            try:
                ctx_i = int(ctx) if ctx is not None else None
            except (TypeError, ValueError):
                ctx_i = None

            blended = _blended_cost(pin, pout)
            if blended <= 0:
                continue
            cand = ModelCandidate(
                provider=provider,
                model_id=mid,
                name=name,
                prompt_per_m_usd=pin,
                completion_per_m_usd=pout,
                blended_per_m_usd=blended,
                context_length=ctx_i,
                is_free=is_free,
                description=desc[:240],
                quality_score=_quality_from_id(mid, name),
                swedish_score=_swedish_score(mid, name, desc),
                eu_score=_eu_score(mid, provider),
                est_cost_per_call_usd=_est_call_cost(pin, pout),
            )
            out.append(cand)
    return out


def score_candidate_for_perspective(
    cand: ModelCandidate,
    perspective: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Higher is better. Returns (score, breakdown)."""
    cost_p = float(perspective.get("cost_priority") or 0.5)
    qual_p = float(perspective.get("quality_priority") or 0.5)
    max_c = float(perspective.get("max_usd_per_m_blended") or 50.0)
    min_ctx = int(perspective.get("min_context") or 0)
    prefer_sv = bool(perspective.get("prefer_swedish"))
    prefer_eu = bool(perspective.get("prefer_eu"))

    # Hard filters → large penalty
    if cand.blended_per_m_usd > max_c * 1.25:
        return -1e9, {"reject": "over_budget"}
    if min_ctx and cand.context_length and cand.context_length < min_ctx * 0.5:
        return -1e9, {"reject": "context_too_small"}

    # Normalize cost into 0..1 where 0 = free/cheap, 1 = at max budget
    cost_norm = min(1.0, cand.blended_per_m_usd / max(max_c, 1e-6))
    cost_term = (1.0 - cost_norm) * cost_p

    quality_term = cand.quality_score * qual_p
    sv_term = cand.swedish_score * (0.35 if prefer_sv else 0.1)
    eu_term = cand.eu_score * (0.2 if prefer_eu else 0.05)

    # Soft preference under budget
    budget_headroom = max(0.0, 1.0 - cost_norm) * 0.1

    total = cost_term + quality_term + sv_term + eu_term + budget_headroom
    breakdown = {
        "cost_term": round(cost_term, 4),
        "quality_term": round(quality_term, 4),
        "swedish_term": round(sv_term, 4),
        "eu_term": round(eu_term, 4),
        "budget_headroom": round(budget_headroom, 4),
        "total": round(total, 4),
        "blended_per_m_usd": round(cand.blended_per_m_usd, 4),
    }
    return total, breakdown


def recommend_for_perspective(
    perspective_id: str,
    *,
    candidates: list[ModelCandidate] | None = None,
    top_k: int = 3,
    config: dict[str, Any] | None = None,
) -> PerspectiveRecommendation:
    if perspective_id not in ANALYSIS_PERSPECTIVES:
        raise KeyError(f"Unknown perspective: {perspective_id}")
    persp = ANALYSIS_PERSPECTIVES[perspective_id]
    cands = candidates if candidates is not None else collect_paid_candidates(config=config)

    ranked: list[tuple[float, dict[str, float], ModelCandidate]] = []
    for c in cands:
        score, br = score_candidate_for_perspective(c, persp)
        if score < -1e8:
            continue
        ranked.append((score, br, c))
    ranked.sort(key=lambda x: x[0], reverse=True)

    top = ranked[: max(1, top_k)]
    recommended = None
    alts: list[dict[str, Any]] = []
    breakdown: dict[str, float] = {}
    provider = "openrouter"

    if top:
        score, breakdown, best = top[0]
        recommended = {
            **best.to_public(),
            "score": round(score, 4),
        }
        provider = best.provider
        for sc, _br, c in top[1:]:
            alts.append({**c.to_public(), "score": round(sc, 4)})

    # API selectable payload — what Testlabb / pipeline should send
    selectable = {
        "provider": provider if provider != "openrouter" else "openrouter",
        "llm_model": recommended["model_id"] if recommended else None,
        "analysis_perspective": perspective_id,
        "use_mistral_llm": True,
        "deep_analysis": perspective_id in {"holistic_deep", "premium_reasoning", "root_cause", "coaching_qa"},
    }

    return PerspectiveRecommendation(
        perspective_id=perspective_id,
        label=str(persp["label"]),
        description=str(persp["description"]),
        icon=str(persp.get("icon") or "scale"),
        use_when=str(persp.get("use_when") or ""),
        cost_priority=float(persp["cost_priority"]),
        quality_priority=float(persp["quality_priority"]),
        max_usd_per_m_blended=float(persp["max_usd_per_m_blended"]),
        tasks=list(persp.get("tasks") or []),
        recommended=recommended,
        alternatives=alts,
        score_breakdown=breakdown,
        provider_for_api=provider,
        selectable=selectable,
    )


def list_analysis_profiles(
    *,
    top_k: int = 3,
    refresh_note: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build full selectable menu of perspectives with cost-aware paid picks."""
    cfg = config or load_provider_config()
    candidates = collect_paid_candidates(config=cfg)
    profiles = [
        recommend_for_perspective(pid, candidates=candidates, top_k=top_k, config=cfg).to_public()
        for pid in ANALYSIS_PERSPECTIVES
    ]
    # Sort menu: balanced first-ish by putting balanced_ops near top, then by label
    order = [
        "balanced_ops",
        "cost_saver",
        "batch_throughput",
        "swedish_quality",
        "sentiment_refine",
        "intent_routing",
        "summary_actions",
        "root_cause",
        "coaching_qa",
        "compliance_risk",
        "holistic_deep",
        "premium_reasoning",
    ]
    rank = {pid: i for i, pid in enumerate(order)}
    profiles.sort(key=lambda p: (rank.get(p["id"], 99), p["label"]))

    # Persist snapshot for UI offline/cache
    snapshot = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_count": len(candidates),
        "profiles": profiles,
        "notes": {
            "cost_unit": "USD per 1M tokens (blended 40% prompt / 60% completion)",
            "est_cost_per_call": "Approx. for ~4k input + 1.5k output tokens",
            "selection": "Send selectable.provider + selectable.llm_model to POST /analyze_pipeline",
        },
    }
    out_dir = Path((cfg.get("catalog") or {}).get("dir") or "data/model_catalogs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "analysis_profiles.json"
    try:
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.debug("could not write profiles snapshot: %s", exc)

    if refresh_note:
        snapshot["catalog_path"] = str(path)
    return snapshot


def load_profiles_snapshot(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path or "data/model_catalogs/analysis_profiles.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


if __name__ == "__main__":
    snap = list_analysis_profiles()
    print(json.dumps({"count": len(snap["profiles"]), "generated_at": snap["generated_at"]}, indent=2))
    for p in snap["profiles"]:
        rec = p.get("recommended") or {}
        print(
            f"- {p['id']:20} → {rec.get('provider','?'):10} {rec.get('model_id','(none)')} "
            f"${rec.get('blended_per_m_usd','?')}/M  est/call=${rec.get('est_cost_per_call_usd','?')}"
        )
