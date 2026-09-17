"""Server-owned call artifacts and provenance for the operator pilot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import Request

from ..customers import CustomerContext
from .call_store import CallStore, call_idempotency_key, new_call_id
from .schemas import CustomerRef
from .settings import get_api_settings

logger = logging.getLogger(__name__)


def get_call_store(request: Request) -> CallStore:
    """Return the app-scoped :class:`CallStore`, creating it on first use."""
    store = getattr(request.app.state, "call_store", None)
    if store is None:
        store = CallStore(get_api_settings().state_dir)
        request.app.state.call_store = store
    return store


def customer_ref(ctx: CustomerContext | None) -> CustomerRef | None:
    """API-facing customer identity (routing context, not authentication)."""
    if ctx is None:
        return None
    return CustomerRef(
        customer_id=ctx.customer_id,
        display_name=ctx.display_name,
        analyzer_profile=ctx.analyzer_profile,
        registry_version=ctx.registry_version,
        config_fingerprint=ctx.config_fingerprint,
    )


def customer_meta(ctx: CustomerContext | None) -> dict[str, Any] | None:
    """JSON-ready frozen customer context for job/call meta."""
    if ctx is None:
        return None
    return ctx.model_dump(mode="json")


def report_as_dict(report: Any) -> dict[str, Any]:
    """Best-effort JSON payload from a pipeline report or test double."""
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return payload
    results = getattr(report, "results", None)
    return {"results": results if isinstance(results, dict) else {}}


def fail_reason_from_exc(exc: BaseException) -> str:
    """Stable provenance code; avoid dumping exception text that may contain paths."""
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and code.strip():
        return code.strip()
    return type(exc).__name__


def persist_call_artifact(
    store: CallStore,
    *,
    status: str,
    transcript: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    customer: CustomerContext | None = None,
    original_filename: str | None = None,
    audio_path: str | None = None,
    route: str,
    call_id: str | None = None,
    fail_reason: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist transcript and/or report with customer provenance.

    Idempotent on ``customer_id + source + config_fingerprint``. A matching
    existing record is updated in place (same server-issued id).
    """
    source = original_filename or (audio_path or "")
    fingerprint = customer.config_fingerprint if customer is not None else None
    customer_id = customer.customer_id if customer is not None else None
    key = call_idempotency_key(customer_id, source, fingerprint)
    existing = store.find_by_idempotency(key) if key else None
    resolved_id = call_id or (existing or {}).get("id") or new_call_id()
    meta = dict(extra_meta or {})
    if customer is not None:
        meta["customer"] = customer.model_dump(mode="json")
    provenance = {
        "original_filename": original_filename,
        "audio_path": audio_path,
        "config_fingerprint": fingerprint,
        "route": route,
        "fail_reason": fail_reason,
    }
    return store.save(
        resolved_id,
        {
            "status": status,
            "customer_id": customer_id,
            "idempotency_key": key,
            "transcript": transcript,
            "report": report,
            "meta": meta,
            "provenance": provenance,
            "created_at": (existing or {}).get("created_at"),
        },
    )


def persist_intake_file(
    store: CallStore | None,
    *,
    audio_path: str,
    route: str,
    status: str,
    transcript: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    customer: CustomerContext | None = None,
    error: BaseException | None = None,
    must_succeed: bool = True,
) -> dict[str, Any] | None:
    """Persist a batch/scan artifact.

    Completed/transcribed jobs require a successful store write
    (``must_succeed=True``). Failure-provenance writes are best-effort
    (``must_succeed=False``) so the original worker error stays visible.
    """
    if store is None:
        return None
    try:
        return persist_call_artifact(
            store,
            status=status,
            transcript=transcript,
            report=report,
            customer=customer,
            original_filename=Path(audio_path).name,
            audio_path=audio_path,
            route=route,
            fail_reason=fail_reason_from_exc(error) if error is not None else None,
        )
    except Exception:
        logger.exception("Failed to persist intake artifact for %s via %s", audio_path, route)
        if must_succeed:
            raise
        return None
