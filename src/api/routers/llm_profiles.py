"""LLM analysis profiles & paid-model recommendations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ...llm.paid_model_advisor import (
    ANALYSIS_PERSPECTIVES,
    list_analysis_profiles,
    load_profiles_snapshot,
    recommend_for_perspective,
)
from ...llm.provider_secrets import list_configured_providers
from ..router_errors import run_route_sync

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.get(
    "/analysis-profiles", summary="List selectable analysis perspectives with paid model picks"
)
def get_analysis_profiles(
    top_k: int = Query(3, ge=1, le=8, description="Alternatives per perspective"),
    refresh: bool = Query(True, description="Recompute from catalogs (false = cached snapshot)"),
) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        if not refresh:
            snap = load_profiles_snapshot()
            if snap:
                snap["cached"] = True
                snap["providers_configured"] = list_configured_providers()
                snap.setdefault(
                    "menu",
                    [
                        {
                            "id": p["id"],
                            "label": p["label"],
                            "description": p["description"],
                            "use_when": p.get("use_when"),
                            "icon": p.get("icon"),
                            "model": (p.get("recommended") or {}).get("model_id"),
                            "provider": (p.get("recommended") or {}).get("provider")
                            or p.get("provider"),
                            "blended_usd_per_m": (p.get("recommended") or {}).get(
                                "blended_per_m_usd"
                            ),
                            "est_cost_per_call_usd": (p.get("recommended") or {}).get(
                                "est_cost_per_call_usd"
                            ),
                            "selectable": p.get("selectable"),
                        }
                        for p in snap.get("profiles") or []
                    ],
                )
                return snap
        snap = list_analysis_profiles(top_k=top_k)
        snap["cached"] = False
        snap["providers_configured"] = list_configured_providers()
        snap["menu"] = [
            {
                "id": p["id"],
                "label": p["label"],
                "description": p["description"],
                "use_when": p["use_when"],
                "icon": p["icon"],
                "cost_priority": p["cost_priority"],
                "quality_priority": p["quality_priority"],
                "model": (p.get("recommended") or {}).get("model_id"),
                "provider": (p.get("recommended") or {}).get("provider") or p.get("provider"),
                "blended_usd_per_m": (p.get("recommended") or {}).get("blended_per_m_usd"),
                "est_cost_per_call_usd": (p.get("recommended") or {}).get("est_cost_per_call_usd"),
                "selectable": p.get("selectable"),
            }
            for p in snap.get("profiles") or []
        ]
        return snap

    return run_route_sync("llm.analysis_profiles", _run)


@router.get(
    "/analysis-profiles/{perspective_id}",
    summary="Detail + ranked paid models for one analysis perspective",
)
def get_analysis_profile_detail(
    perspective_id: str,
    top_k: int = Query(5, ge=1, le=15),
) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        if perspective_id not in ANALYSIS_PERSPECTIVES:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown_perspective",
                    "available": sorted(ANALYSIS_PERSPECTIVES.keys()),
                },
            )
        rec = recommend_for_perspective(perspective_id, top_k=top_k)
        out = rec.to_public()
        out["providers_configured"] = list_configured_providers()
        return out

    return run_route_sync("llm.analysis_profile_detail", _run)


@router.get("/providers", summary="Configured LLM providers (key present?)")
def get_llm_providers() -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        providers = list_configured_providers()
        lmstudio: dict[str, Any] = {"configured": providers.get("lmstudio", False)}
        try:
            from ...llm.client_factory import resolve_llm_client

            resolved = resolve_llm_client("lmstudio")
            if resolved.client is None:
                raise RuntimeError("LM Studio client was not resolved")
            status = resolved.client.model_status(resolved.model)
            lmstudio.update(status.to_dict())
            lmstudio["reachable"] = True
            lmstudio["context_ready"] = bool(
                status.loaded_context and status.loaded_context >= resolved.client.requested_context
            )
            lmstudio["reasoning_ready"] = status.reasoning_default in {None, "off"}
            lmstudio["ready"] = bool(lmstudio["context_ready"])
            lmstudio["requested_context"] = resolved.client.requested_context
        except Exception as exc:
            lmstudio.update(
                {
                    "reachable": False,
                    "context_ready": False,
                    "reasoning_ready": False,
                    "ready": False,
                    "error_code": getattr(exc, "error_code", "lmstudio_unavailable"),
                }
            )
        return {"providers": providers, "lmstudio": lmstudio}

    return run_route_sync("llm.providers", _run)
