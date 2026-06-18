"""Fas 5 helper: Ingest YouTube downloaded files into the project data pipeline.

Usage:
    python scripts/ingest_youtube_data.py

Scans data/ingested/youtube/ and prepares a list for use in generate_testset.py
or prepare_callcenter_data.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def list_youtube_ingested(base_dir: Path = Path("data/ingested/youtube")) -> List[Dict]:
    """List all YouTube downloaded files with metadata."""
    if not base_dir.exists():
        return []

    files = []
    for json_file in base_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                meta = json.load(f)
            wav_file = base_dir / json_file.with_suffix(".wav").name
            if not wav_file.exists():
                # Try other formats
                for ext in [".m4a", ".webm", ".mp3"]:
                    candidate = base_dir / json_file.with_suffix(ext).name
                    if candidate.exists():
                        wav_file = candidate
                        break
            files.append({
                "json": str(json_file),
                "audio": str(wav_file) if wav_file.exists() else None,
                "title": meta.get("title", ""),
                "duration": meta.get("duration_seconds"),
                "youtube_id": meta.get("youtube_id"),
                "source_url": meta.get("source_url"),
            })
        except Exception:
            continue
    return files


def prepare_youtube_for_testset(output_file: Path = Path("data/youtube_ingested.jsonl")) -> None:
    """Export YouTube files as JSONL for easy import into testset generation."""
    files = list_youtube_ingested()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in files:
            if item["audio"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Exported {len(files)} YouTube files to {output_file}")


def main() -> None:
    prepare_youtube_for_testset()


if __name__ == "__main__":
    main()
