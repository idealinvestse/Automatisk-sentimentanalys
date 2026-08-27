"""Export DEMO_TRANSCRIPTS from Python to webui TypeScript.

Source of truth: app.services.data_services.DEMO_TRANSCRIPTS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "webui" / "src" / "lib" / "demo-transcripts.generated.json",
    )
    parser.add_argument("--check", action="store_true", help="Exit 1 if TS IDs drift")
    args = parser.parse_args()

    from app.services.data_services import DEMO_TRANSCRIPTS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(DEMO_TRANSCRIPTS, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.check:
        ts = (ROOT / "webui" / "src" / "lib" / "demo-transcripts.ts").read_text(encoding="utf-8")
        py_ids = [str(item["id"]) for item in DEMO_TRANSCRIPTS]
        missing = [i for i in py_ids if f'id: "{i}"' not in ts and f"id: '{i}'" not in ts]
        if missing:
            raise SystemExit(f"demo-transcripts.ts missing IDs: {missing}")
    print(f"Wrote {args.output} ({len(DEMO_TRANSCRIPTS)} transcripts)")


if __name__ == "__main__":
    main()
