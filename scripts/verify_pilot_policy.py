"""Verify conditional-pilot policy locks (local ASR, PII, Groq/Deepgram hygiene).

Exit codes:
  0 — all hard checks passed (warnings may still print)
  1 — one or more hard checks failed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def check_anonymize_default() -> tuple[bool, str]:
    from src.profiles import PROFILE_SPECS

    llm = PROFILE_SPECS.get("callcenter", {}).get("llm", {})
    if llm.get("anonymize_before_llm") is True:
        return True, "callcenter.anonymize_before_llm=True"
    return False, "callcenter.anonymize_before_llm is not True (required for pilot)"


def check_asr_schema_default() -> tuple[bool, str]:
    from src.install.config_schema import AsrDefaults

    default = AsrDefaults().provider
    if default == "local":
        return True, "AsrDefaults.provider default=local"
    return False, f"AsrDefaults.provider default={default!r} (expected 'local')"


def check_production_guards() -> tuple[bool, str]:
    prod = os.environ.get("API_PRODUCTION", "").lower() in {"1", "true", "yes"}
    if not prod:
        return True, "API_PRODUCTION not set — skipped auth/media guards (dev OK)"
    missing = []
    if not os.environ.get("SENTIMENT_API_KEY"):
        missing.append("SENTIMENT_API_KEY")
    if not os.environ.get("API_MEDIA_ROOT"):
        missing.append("API_MEDIA_ROOT")
    if missing:
        return False, f"API_PRODUCTION=true but missing: {', '.join(missing)}"
    return True, "API_PRODUCTION guards present (key + media root)"


def check_cloud_keys(*, strict: bool) -> tuple[bool, list[str]]:
    """Return (ok, messages). In strict+production, Groq/Deepgram keys fail."""
    messages: list[str] = []
    prod = os.environ.get("API_PRODUCTION", "").lower() in {"1", "true", "yes"}
    groq = bool(os.environ.get("GROQ_API_KEY"))
    deepgram = bool(os.environ.get("DEEPGRAM_API_KEY") or os.environ.get("CLOUD_STT_API_KEY"))

    if groq:
        msg = "GROQ_API_KEY is set — pilot policy: dev-only, not for customer PII"
        if strict and prod:
            return False, [f"FAIL: {msg}"]
        messages.append(f"WARN: {msg}")
    else:
        messages.append("OK: GROQ_API_KEY unset")

    if deepgram:
        msg = "Deepgram/CLOUD_STT key set — ensure provider=local for PII calls"
        if strict and prod:
            return False, [f"FAIL: {msg}"]
        messages.append(f"WARN: {msg}")
    else:
        messages.append("OK: DEEPGRAM/CLOUD_STT key unset")

    return True, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pilot policy configuration")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if Groq/Deepgram keys present while API_PRODUCTION=true",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Do not load .env from repo root",
    )
    args = parser.parse_args()
    if not args.no_dotenv:
        _load_dotenv()

    hard_ok = True
    print("=== Pilot policy verification ===")

    for check in (check_anonymize_default, check_asr_schema_default, check_production_guards):
        ok, msg = check()
        print(("OK: " if ok else "FAIL: ") + msg)
        hard_ok = hard_ok and ok

    cloud_ok, cloud_msgs = check_cloud_keys(strict=args.strict)
    for line in cloud_msgs:
        print(line)
    hard_ok = hard_ok and cloud_ok

    if hard_ok:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
