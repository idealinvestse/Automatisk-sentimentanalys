#!/usr/bin/env python3
"""Orchestrate pilot release gates (policy + L7/L8/L9 smoke).

Usage:
    python scripts/run_pilot_gates.py
    python scripts/run_pilot_gates.py --strict --skip-l8 --skip-l9
    python scripts/run_pilot_gates.py --device cpu --audio-pack sv_callcenter
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env or os.environ.copy())
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pilot release gates L7–L9 + policy")
    parser.add_argument("--strict", action="store_true", help="verify_pilot_policy --strict")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--audio-pack", default="sv_callcenter")
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--skip-l7", action="store_true")
    parser.add_argument("--skip-l8", action="store_true")
    parser.add_argument("--skip-l9", action="store_true")
    parser.add_argument(
        "--api-base",
        default=os.getenv("API_BASE", "http://localhost:8000"),
        help="Base URL for L9 staging smoke",
    )
    parser.add_argument("--api-key", default=os.getenv("SENTIMENT_API_KEY", ""))
    args = parser.parse_args()

    failures: list[str] = []

    if not args.skip_policy:
        policy_cmd = [sys.executable, "scripts/verify_pilot_policy.py"]
        if args.strict:
            policy_cmd.append("--strict")
        if _run(policy_cmd) != 0:
            failures.append("policy")

    if not args.skip_l7:
        l7 = [
            sys.executable,
            "-m",
            "src.evaluate",
            "audio",
            "smoke",
            "--device",
            args.device,
            "--pack",
            args.audio_pack,
            "--limit",
            "1",
        ]
        if _run(l7) != 0:
            failures.append("L7")

    if not args.skip_l8:
        l8 = [sys.executable, "-m", "src.evaluate", "llm-quality"]
        if _run(l8) != 0:
            failures.append("L8")

    if not args.skip_l9:
        l9 = [
            sys.executable,
            "scripts/staging_observability_smoke.py",
            "--api-base",
            args.api_base,
        ]
        if args.api_key:
            l9.extend(["--api-key", args.api_key])
        if _run(l9) != 0:
            failures.append("L9")

    if failures:
        print(f"\nFAIL: pilot gates failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nOK: pilot gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
