"""Export FastAPI OpenAPI schema for webui type generation (no server required).

Usage:
  python scripts/export_openapi.py -o webui/openapi.json
  cd webui && npm run generate:types
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OpenAPI JSON for the Sentiment API")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "webui" / "openapi.json",
        help="Output path (default: webui/openapi.json)",
    )
    args = parser.parse_args()

    from src.api.app import create_app

    app = create_app()
    schema = app.openapi()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths = len(schema.get("paths") or {})
    print(f"OK: wrote {args.output} ({paths} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
