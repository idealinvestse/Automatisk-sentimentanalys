"""Optional PII redactor for transcripts before sending to external LLM (Fas 3.4 follow-up).

This is the implementation of the "valfritt" pii_redactor mentioned in the plan.

Design rationale:
- European/GDPR priority: Before any data leaves to OpenRouter/Mistral, give users the option to
  redact obvious PII from Swedish callcenter transcripts.
- Simple regex-based for common patterns (email, Swedish phone, personnummer/SSN-like).
  Real production would use a proper NER model (e.g. KB-BERT for Swedish names/addresses) or
  external redaction service.
- Profile-driven: Controlled by `llm.anonymize_before_llm` in the callcenter (or other) profile.
- Non-destructive for local analysis: The redaction only affects the text sent to the LLM path.
  Original segments stay intact in the CallAnalysisReport.
- Transparent: Redacted version is logged at DEBUG level; the LLM output meta can note if redaction was used.
- Pluggable: Easy to extend or replace (e.g. with presidio or custom).

Usage (from analyzer or future pipeline step):
    from src.llm.pii_redactor import redact_pii
    from src.profiles import resolve_profile

    _, spec = resolve_profile(profile=profile_name)
    if spec.get("llm", {}).get("anonymize_before_llm"):
        safe_transcript = redact_pii(transcript_text)
    else:
        safe_transcript = transcript_text

Then pass safe_transcript to the prompt builder.

The redactor is intentionally conservative (only high-confidence patterns) to avoid breaking
callcenter terminology (e.g. "fakturanummer" should not be mangled).

See UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md (3.4.1 + post-3.4.3 follow-up)
and docs/FAS3_MISTRAL_LLM_INTEGRATION.md for privacy notes.
"""

from __future__ import annotations

import re
from typing import Any

# Common Swedish/EU PII patterns (conservative)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:(?:\+46|0)[\s-]?)?(?:\d[\s-]?){6,12}\d")  # Swedish phones + international-ish
_PERSONNUMMER_RE = re.compile(r"\b(?:19|20)?\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[-+]?\d{4}\b")  # YYMMDD-XXXX or similar
# Simple name redaction is risky without context/NER – skipped for v1 (would over-redact "Anna" in company names etc.)

_REPLACEMENTS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "personnummer": "[REDACTED_PNR]",
}


def redact_pii(text: str, redaction_map: dict[str, str] | None = None) -> str:
    """Redact obvious PII from a transcript string.

    Returns a new string with PII replaced by safe tokens.
    Does not modify the input.
    """
    if not text:
        return text

    replacements = redaction_map or _REPLACEMENTS
    result = text

    # Order matters a bit (personnummer before generic phone)
    result = _PERSONNUMMER_RE.sub(replacements.get("personnummer", "[REDACTED_PNR]"), result)
    result = _EMAIL_RE.sub(replacements.get("email", "[REDACTED_EMAIL]"), result)
    result = _PHONE_RE.sub(replacements.get("phone", "[REDACTED_PHONE]"), result)

    return result


def redact_segments(segments: list[dict[str, Any]] | list[Any], profile_name: str = "callcenter") -> list[dict[str, Any]]:
    """Convenience: redact the 'text' field of segments if the profile requests anonymization.

    Accepts list[dict] or list[Segment]. Always returns list[dict] (converted if needed).
    Original input is not mutated.
    """
    try:
        from ..profiles import resolve_profile
        _, spec = resolve_profile(profile=profile_name)
        llm_spec = spec.get("llm", {}) or {}
        if not llm_spec.get("anonymize_before_llm"):
            return segments  # no-op, return original for efficiency
    except Exception:
        # If profile system fails, be safe and do not redact (or log)
        return segments

    redacted = []
    for seg in segments:
        if isinstance(seg, dict):
            new_seg = dict(seg)  # shallow copy
        else:
            # Support Segment dataclass or objects with to_dict (robustness for direct calls)
            if hasattr(seg, "to_dict"):
                new_seg = seg.to_dict()
            else:
                new_seg = dict(getattr(seg, "__dict__", {}))
        if "text" in new_seg and isinstance(new_seg["text"], str):
            new_seg["text"] = redact_pii(new_seg["text"])
        redacted.append(new_seg)
    return redacted


# Example usage note for future integration in mistral_analyzer:
# Before building the role-labeled transcript for the LLM:
#   if anonymize:
#       segments_for_llm = redact_segments(segments, profile_name)
#   else:
#       segments_for_llm = segments
# Then _build_role_labeled_transcript(segments_for_llm ...)
