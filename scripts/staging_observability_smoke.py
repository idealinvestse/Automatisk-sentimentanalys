#!/usr/bin/env python3
"""Smoke test for Docker staging stack observability endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Staging observability smoke test")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--pipeline-calls", type=int, default=3)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SENTIMENT_API_KEY", ""),
        help="X-API-Key for authenticated endpoints (metrics, pipeline, status)",
    )
    args = parser.parse_args()
    base = args.api_base.rstrip("/")
    auth_headers = {"X-API-Key": args.api_key} if args.api_key else {}

    checks: list[tuple[str, bool, str]] = []

    code, body = _get(f"{base}/health")
    checks.append(("health", code == 200 and "ok" in body.lower(), f"status={code}"))

    code, body = _get(f"{base}/metrics", headers=auth_headers or None)
    checks.append(
        (
            "metrics",
            code == 200 and "http_requests_total" in body,
            f"status={code}, has_http_requests_total={'http_requests_total' in body}",
        )
    )

    code, body = _get(f"{base}/status/processes?limit=5", headers=auth_headers or None)
    checks.append(("status/processes", code == 200, f"status={code}"))

    code, body = _get(f"{base}/status/health/detail", headers=auth_headers or None)
    checks.append(("status/health/detail", code == 200, f"status={code}"))

    segment_payload = json.dumps(
        {
            "segments": [{"text": "Tack för hjälpen, det fungerade bra.", "speaker": "customer"}],
            "profile": "callcenter",
            "use_mistral_llm": False,
        }
    ).encode("utf-8")
    for i in range(args.pipeline_calls):
        req = urllib.request.Request(
            f"{base}/analyze_pipeline",
            data=segment_payload,
            headers={"Content-Type": "application/json", **auth_headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                ok = resp.status == 200
        except urllib.error.HTTPError as exc:
            ok = False
            print(f"pipeline call {i + 1} failed: {exc.code}", file=sys.stderr)
        checks.append((f"analyze_pipeline_{i + 1}", ok, "ok" if ok else "failed"))

    failed = [name for name, ok, _ in checks if not ok]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} ({detail})")

    if failed:
        print(f"\nFailed checks: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print("\nAll staging observability checks passed.")


if __name__ == "__main__":
    main()
