#!/usr/bin/env python3
"""Backup script for Swedish Sentiment Analysis production data.

Creates a timestamped tar.gz archive of:
  - outputs/ (pipeline reports, evaluation results)
  - .cache/alerting_state.json (persistent alerting state)
  - configs/ (LLM config, QA scorecards, alerting config, profiles)

Optionally saves Redis cache (if API_USE_REDIS_CACHE=true).

Usage:
  python scripts/backup.py [--output-dir /backups] [--keep 7] [--redis]

Cron example (daily at 02:00):
  0 2 * * * cd /app && python scripts/backup.py --output-dir /backups --keep 7 --redis

Exit codes:
  0 — success
  1 — partial failure (some paths missing, archive still created)
  2 — fatal failure (archive could not be created)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# Paths to back up (relative to project root)
BACKUP_PATHS = [
    "outputs/",
    ".cache/alerting_state.json",
    ".cache/aggregates/",
    "configs/",
]

# Paths that are optional (won't fail if missing)
OPTIONAL_PATHS = [
    ".cache/alerting_state.json",
    ".cache/aggregates/",
]

REDIS_SAVE_CMD = ["redis-cli", "BGSAVE"]


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    p = Path(__file__).resolve().parent.parent
    if (p / "pyproject.toml").is_file():
        return p
    # Fallback: current directory
    return Path.cwd()


def create_backup(
    project_root: Path,
    output_dir: Path,
    keep: int,
    use_redis: bool,
) -> int:
    """Create a timestamped backup archive. Returns exit code."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"sentiment-backup-{timestamp}.tar.gz"
    archive_path = output_dir / archive_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Redis BGSAVE before archiving (if enabled)
    if use_redis:
        redis_url = os.getenv("REDIS_URL")
        cmd = REDIS_SAVE_CMD + (["-u", redis_url] if redis_url else [])
        try:
            subprocess.run(cmd, check=True, timeout=30, capture_output=True)
            print(f"[backup] Redis BGSAVE triggered ({redis_url or 'default'})")
        except Exception as exc:
            print(f"[backup] WARNING: Redis BGSAVE failed: {exc}", file=sys.stderr)

    # Collect paths to archive
    paths_to_archive: list[tuple[str, Path]] = []
    missing_optional: list[str] = []
    missing_required: list[str] = []

    for rel_path in BACKUP_PATHS:
        full_path = project_root / rel_path
        if full_path.exists():
            paths_to_archive.append((rel_path, full_path))
        elif rel_path in OPTIONAL_PATHS:
            missing_optional.append(rel_path)
        else:
            missing_required.append(rel_path)

    if missing_required:
        for p in missing_required:
            print(f"[backup] ERROR: Required path missing: {p}", file=sys.stderr)

    if not paths_to_archive:
        print("[backup] FATAL: No paths to archive", file=sys.stderr)
        return 2

    # Create tar.gz
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            for rel_path, full_path in paths_to_archive:
                arcname = rel_path.rstrip("/")
                tar.add(str(full_path), arcname=arcname)
                print(f"[backup] Added: {rel_path}")
    except Exception as exc:
        print(f"[backup] FATAL: Failed to create archive: {exc}", file=sys.stderr)
        return 2

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"[backup] Created: {archive_path} ({size_mb:.1f} MB)")

    if missing_optional:
        for p in missing_optional:
            print(f"[backup] NOTE: Optional path skipped (not found): {p}")

    # Rotate old backups
    backups = sorted(output_dir.glob("sentiment-backup-*.tar.gz"))
    if len(backups) > keep:
        for old in backups[:-keep]:
            old.unlink()
            print(f"[backup] Rotated old backup: {old.name}")

    return 1 if missing_required else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup sentiment analysis production data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/backups"),
        help="Directory to store backup archives (default: /backups)",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=7,
        help="Number of recent backups to keep (default: 7)",
    )
    parser.add_argument(
        "--redis",
        action="store_true",
        help="Trigger Redis BGSAVE before archiving (for Redis cache backup)",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    print(f"[backup] Project root: {project_root}")
    print(f"[backup] Output dir: {args.output_dir}")
    print(f"[backup] Keep: {args.keep} archives")

    return create_backup(project_root, args.output_dir, args.keep, args.redis)


if __name__ == "__main__":
    sys.exit(main())
