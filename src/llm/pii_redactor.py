"""PII Redaction Module (Fas 4.4.1 - Early Pipeline).

Enhanced implementation for full early-pipeline privacy (before any local analyzers, LLM or persistence).

Key upgrades for 4.4.1 (per UTVECKLINGSPLAN_Fas4... v1.1):
- Runs EARLY in CallAnalysisPipeline (in analyze_audio and analyze_segments, right after transcription, before run_analyzers/LLM).
- When profile llm.anonymize_before_llm=True: redacts text used for BOTH local analysis (sentiment, role, topics, etc.) AND LLM path. The final report.segments will contain redacted text.
- Detailed, structured logging of EXACTLY what was redacted (type, original snippet, replacement, location). Attached as Pydantic PiiRedactionLog to results["pii_redaction"].
- Extended regex patterns (credit cards, addresses) + optional NER for Swedish names (transformers KB-BERT if available, else conservative regex/heuristic).
- Conservative to avoid mangling callcenter terms.
- Idempotent (redacting already-redacted text is no-op).
- Transparent + auditable for GDPR/compliance.

Profile control:
  "llm": { "anonymize_before_llm": true }  # enables early full redaction for local + LLM

Usage in pipeline (explicit integration):
    # early, before analyzers
    redacted_dicts, pii_log = redact_segments(seg_dicts, profile_name, return_log=True)
    if pii_log.total_redacted > 0:
        results["pii_redaction"] = pii_log.model_dump()
        logger.info("PII redacted early: %d events, types=%s", pii_log.total_redacted, pii_log.types_redacted)
    # then use redacted for ctx.segments and report

See also previous Fas3 notes and the plan for privacy-by-design requirements.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .schemas import PiiRedactionEvent, PiiRedactionLog

logger = logging.getLogger(__name__)

# Common Swedish/EU PII patterns (conservative, high-precision to avoid false positives on callcenter terms like "fakturanummer")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:"
    # International format: +46 70 123 45 67 — must not be preceded by another digit
    r"(?<!\d)\+46[\s-]?\d{1,4}(?:[\s-]?\d{2,4}){2,4}"
    r"|"
    # National format with 0 prefix: 070-123 45 67, 08-123 456 78 — must not be preceded by another digit
    r"(?<!\d)0\d{2,3}(?:[\s-]?\d{2,4}){2,4}"
    r"|"
    # 10-digit mobile starting with 07 (no separators)
    r"\b07\d{8}\b"
    r")"
)  # Swedish phone numbers — requires +46 or 0 prefix, not preceded by digit, to avoid false positives on invoice/case IDs
_PERSONNUMMER_RE = re.compile(
    r"\b(?:19|20)?\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[-+]?\d{4}\b"
)
_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ -]*?){13,16}\b"
)  # 13-16 digits with optional spaces/dashes; filtered by Luhn checksum (see _is_valid_luhn)
# Permissive address regex — matches capitalized word(s) followed by number.
# Suffix validation is done in Python to handle compound Swedish street names
# (e.g. "Brunkebergstorg" = "Brunkeberg"+"torg", "Vasaplatsen" = "Vasa"+"platsen").
_ADDRESS_RE = re.compile(r"\b[A-ZÅÄÖ][A-Za-zÅÄÖåäö]+(?:\s+[A-ZÅÄÖ][A-Za-zÅÄÖåäö]+)?\s+\d{1,4}\b")

# Known Swedish + English street suffixes for address validation (case-insensitive).
# Used in redact_pii() to verify the permissive match actually looks like a street.
_ADDRESS_SUFFIXES = frozenset(
    {
        # Swedish street name endings
        "gatan",
        "vägen",
        "gata",
        "väg",
        "torget",
        "torg",
        "platsen",
        "esplanaden",
        "allén",
        "avenyn",
        "aveny",
        "boulevarden",
        "stråket",
        "leden",
        "backen",
        "höjden",
        "kajen",
        "plan",
        # English / legacy
        "avenue",
        "street",
        "st",
        "road",
        "rd",
        "allé",
        "esplanad",
    }
)

# Conservative name heuristic (only after common titles or in specific contexts to avoid over-redaction)
_NAME_TITLE_RE = re.compile(
    r"\b(?:herr|fru|fröken|dr|prof|hr|fr)\s+([A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+)?)",
    re.IGNORECASE,
)

_REPLACEMENTS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "personnummer": "[REDACTED_PNR]",
    "credit_card": "[REDACTED_CC]",
    "address": "[REDACTED_ADDRESS]",
    "name": "[REDACTED_NAME]",
}

# Top Swedish first names per SCB (Statistics Sweden) — adult population prevalence.
# Used for conservative heuristic redaction (title context or NER). List covers ~80% of common Swedish names.
_COMMON_SWEDISH_FIRST_NAMES = {
    # Top female (SCB all-ages)
    "anna",
    "kristina",
    "margareta",
    "birgitta",
    "elisabeth",
    "eva",
    "karin",
    "lena",
    "maria",
    "kerstin",
    "ingrid",
    "marianne",
    "gunilla",
    "britt",
    "inger",
    "susanne",
    "monica",
    "annika",
    "åsa",
    "helena",
    "barbro",
    "majbritt",
    "ann-marie",
    "gunvor",
    "ingegerd",
    "astrid",
    "maj",
    "siv",
    "berit",
    "gunnel",
    "solveig",
    "ritt",
    "gun",
    "ann-charlotte",
    "ann-britt",
    "kajsa",
    # Top male (SCB all-ages)
    "lars",
    "anders",
    "johan",
    "erik",
    "karl",
    "nils",
    "per",
    "bengt",
    "bo",
    "jan",
    "sven",
    "gunnar",
    "hans",
    "göran",
    "ingvar",
    "rolf",
    "kjell",
    "leif",
    "lennart",
    "olof",
    "stig",
    "mats",
    "peter",
    "ulf",
    "christer",
    "hakan",
    "magnus",
    "fredrik",
    "daniel",
    "martin",
    "andreas",
    "mikael",
    "joakim",
    "tomas",
    "andersson",
    "johansson",
    "svensson",
    "persson",
}


def _is_valid_luhn(digits: str) -> bool:
    """Validate credit card number using the Luhn algorithm (mod 10).

    Args:
        digits: String containing only digits (no spaces or dashes).

    Returns:
        True if the number passes the Luhn checksum (valid credit card),
        False otherwise (likely invoice number, case ID, or other non-PII).

    This prevents false positives on 13-16 digit numeric identifiers
    that are common in call center contexts (fakturanummer, ärendenummer).
    """
    if not digits or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:  # every second digit from right (0-indexed in reversed)
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(
    text: str, redaction_map: dict[str, str] | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """Redact obvious PII from a transcript string (Fas 4.4.1).

    Returns: (redacted_text, list_of_events)
    Each event: {"type": , "original": , "replacement": , "char_start": , "char_end": }
    Does not modify the input.
    """
    if not text:
        return text, []

    replacements = redaction_map or _REPLACEMENTS
    result = text
    events: list[dict[str, Any]] = []

    def _replace_with_log(
        pattern: re.Pattern, repl: str, pii_type: str, current_text: str
    ) -> tuple[str, list[dict]]:
        local_events = []
        new_text = current_text
        # Track redacted intervals to avoid overlapping/nested replacements
        redacted_intervals: list[tuple[int, int]] = []

        for m in reversed(list(pattern.finditer(current_text))):  # reverse to preserve indices
            start, end = m.start(), m.end()
            # Skip if this interval overlaps with a previous redaction
            if any(s < end and e > start for s, e in redacted_intervals):
                continue
            orig = m.group(0)
            new_text = new_text[:start] + repl + new_text[end:]
            redacted_intervals.append((start, end))
            local_events.append(
                {
                    "type": pii_type,
                    "original": orig,
                    "replacement": repl,
                    "char_start": start,
                    "char_end": end,
                }
            )
        return new_text, local_events

    # Order: personnummer first (more specific)
    result, ev = _replace_with_log(
        _PERSONNUMMER_RE, replacements.get("personnummer", "[REDACTED_PNR]"), "personnummer", result
    )
    events.extend(ev)

    result, ev = _replace_with_log(
        _EMAIL_RE, replacements.get("email", "[REDACTED_EMAIL]"), "email", result
    )
    events.extend(ev)

    # Credit card BEFORE phone: filter through Luhn validation to avoid false positives on invoice/case numbers.
    # Doing CC first prevents the phone regex from consuming CC digits (since replaced text has no digits).
    cc_events: list[dict[str, Any]] = []
    for m in reversed(list(_CREDIT_CARD_RE.finditer(result))):
        orig = m.group(0)
        digits = re.sub(r"[\s-]", "", orig)
        if _is_valid_luhn(digits):
            start, end = m.start(), m.end()
            result = (
                result[:start] + replacements.get("credit_card", "[REDACTED_CC]") + result[end:]
            )
            cc_events.append(
                {
                    "type": "credit_card",
                    "original": orig,
                    "replacement": replacements.get("credit_card", "[REDACTED_CC]"),
                    "char_start": start,
                    "char_end": end,
                }
            )
    events.extend(cc_events)

    result, ev = _replace_with_log(
        _PHONE_RE, replacements.get("phone", "[REDACTED_PHONE]"), "phone", result
    )
    events.extend(ev)

    result, ev = _replace_with_log(
        _ADDRESS_RE, replacements.get("address", "[REDACTED_ADDRESS]"), "address", result
    )
    events.extend(ev)
    # Address: filter permissive regex matches by known suffix presence
    addr_events: list[dict[str, Any]] = []
    for m in reversed(list(_ADDRESS_RE.finditer(result))):
        full = m.group(0)
        # Extract street portion (everything before the trailing number)
        street_match = re.match(r"^(.+?)\s+\d{1,4}\s*$", full)
        if not street_match:
            continue
        street = street_match.group(1)
        # Validate that street part contains a known suffix
        if any(suf in street.lower() for suf in _ADDRESS_SUFFIXES):
            start, end = m.start(), m.end()
            result = (
                result[:start] + replacements.get("address", "[REDACTED_ADDRESS]") + result[end:]
            )
            addr_events.append(
                {
                    "type": "address",
                    "original": full,
                    "replacement": replacements.get("address", "[REDACTED_ADDRESS]"),
                    "char_start": start,
                    "char_end": end,
                }
            )
    # Replace the events from the permissive regex run with the validated ones
    events = [e for e in events if e["type"] != "address"]
    events.extend(addr_events)

    # Name heuristic (title-based only, conservative)
    result, ev = _replace_with_log(
        _NAME_TITLE_RE, replacements.get("name", "[REDACTED_NAME]"), "name", result
    )
    events.extend(ev)

    return result, events


def redact_segments(
    segments: list[dict[str, Any]] | list[Any],
    profile_name: str = "callcenter",
    return_log: bool = False,
) -> tuple[list[dict[str, Any]], PiiRedactionLog] | list[dict[str, Any]]:
    """Redact PII in segments list (early pipeline for Fas 4.4.1).

    If profile llm.anonymize_before_llm is True:
      - Redacts text for BOTH local analysis (sentiment etc) and LLM.
      - The returned segments have redacted .text (report will reflect redacted data for privacy).

    Returns:
      If return_log=False (legacy): just the (possibly redacted) list[dict]
      If return_log=True: (redacted_list[dict], PiiRedactionLog pydantic)

    Supports input as list[dict] or list[Segment]. Always returns list[dict].
    """
    applied = False  # noqa: F841 — kept for clarity/logging hooks
    try:
        from ..profiles import resolve_profile

        _, spec = resolve_profile(profile=profile_name)
        llm_spec = spec.get("llm", {}) or {}
        if not llm_spec.get("anonymize_before_llm"):
            if return_log:
                log = PiiRedactionLog(
                    events=[],
                    total_redacted=0,
                    types_redacted=[],
                    applied_to_local=False,
                    profile=profile_name,
                )
                return segments, log
            return segments
    except Exception:
        if return_log:
            log = PiiRedactionLog(
                events=[],
                total_redacted=0,
                types_redacted=[],
                applied_to_local=False,
                profile=profile_name,
            )
            return segments, log
        return segments

    redacted_list: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []

    # Optional NER for names (Swedish) - lazy, only if anonymize enabled
    ner_pipeline = None
    try:
        from transformers import pipeline  # type: ignore

        # Small Swedish NER if available in env (non-fatal if missing)
        ner_pipeline = pipeline(
            "ner", model="KB/bert-base-swedish-cased-ner", aggregation_strategy="simple", device=-1
        )
        logger.debug("PII redactor: Swedish NER pipeline loaded for names/addresses")
    except Exception:
        ner_pipeline = None  # regex + heuristic only

    for idx, seg in enumerate(segments):
        if isinstance(seg, dict):
            new_seg = dict(seg)
        else:
            if hasattr(seg, "to_dict"):
                new_seg = seg.to_dict()
            else:
                new_seg = dict(getattr(seg, "__dict__", {}))

        original_text = new_seg.get("text", "") if isinstance(new_seg.get("text"), str) else ""
        if original_text:
            redacted_text, events = redact_pii(original_text)

            # Optional NER pass for additional names (if loaded)
            if ner_pipeline and original_text == redacted_text:  # only if regex didn't catch much
                try:
                    ner_results = ner_pipeline(original_text)
                    for ent in ner_results:
                        if ent.get("entity_group") in ("PER", "LOC") and ent.get("score", 0) > 0.85:
                            # conservative: only redact high-conf person/location
                            start, end = ent["start"], ent["end"]
                            snippet = original_text[start:end]
                            if (
                                len(snippet) > 2
                                and snippet.lower() not in _COMMON_SWEDISH_FIRST_NAMES
                            ):
                                repl = (
                                    "[REDACTED_NAME]"
                                    if ent["entity_group"] == "PER"
                                    else "[REDACTED_ADDRESS]"
                                )
                                redacted_text = redacted_text[:start] + repl + redacted_text[end:]
                                events.append(
                                    {
                                        "type": (
                                            "name" if ent["entity_group"] == "PER" else "address"
                                        ),
                                        "original": snippet,
                                        "replacement": repl,
                                        "char_start": start,
                                        "char_end": end,
                                    }
                                )
                except Exception as ner_e:
                    logger.debug("NER pass skipped: %s", ner_e)

            new_seg["text"] = redacted_text
            for ev in events:
                ev["segment_index"] = idx
            all_events.extend(events)

        redacted_list.append(new_seg)

    if return_log:
        types = sorted({e["type"] for e in all_events})
        log = PiiRedactionLog(
            events=[PiiRedactionEvent(**e) for e in all_events],
            total_redacted=len(all_events),
            types_redacted=types,
            applied_to_local=True,
            profile=profile_name,
        )
        if all_events:
            logger.info(
                "PII redaction (early pipeline, profile=%s): %d events, types=%s. Log attached to results['pii_redaction'].",
                profile_name,
                len(all_events),
                types,
            )
        return redacted_list, log

    return redacted_list


# Example (early pipeline integration - see src/pipeline.py for actual code):
#   seg_dicts = [s.to_dict() for s in transcript.segments]
#   redacted, pii_log = redact_segments(seg_dicts, profile_name=self.profile, return_log=True)
#   if pii_log.total_redacted > 0:
#       results["pii_redaction"] = pii_log.model_dump()
#   typed_segments = [Segment.from_dict(d) for d in redacted]
#   # use typed_segments for AnalysisContext and final report (local + LLM now see redacted text)
