"""Typer CLI for audio sample benchmarks (mounted under src.evaluate audio)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.table import Table

from .audio_catalog import load_catalog
from .audio_models import SampleFilter
from .audio_runner import run_scenario
from .audio_scenarios import SCENARIO_IDS

audio_app = typer.Typer(help="Audio sample benchmarks (samples/audio)")
console = Console()


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _default_output(scenario: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    return f"reports/audio_{scenario}_{ts}.json"


def _save_report(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    console.print(f"[green]Rapport sparad:[/green] {path}")


@audio_app.command("list")
def audio_list(
    pack: str | None = typer.Option(None, "--pack", help="Filter by pack id"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
    emotions: str | None = typer.Option(None, "--emotions", help="Comma-separated emotions"),
    actors: str | None = typer.Option(None, "--actors", help="Comma-separated actor ids"),
    limit: int | None = typer.Option(20, "--limit", help="Max rows to display"),
    audio_root: str | None = typer.Option(None, "--audio-root", help="Override samples/audio root"),
) -> None:
    """List catalogued audio samples."""
    catalog = load_catalog(audio_root)
    samples = catalog.discover(
        SampleFilter(
            pack_ids=[pack] if pack else None,
            tags=_split_csv(tags),
            emotions=_split_csv(emotions),
            actors=_split_csv(actors),
            limit=limit,
        )
    )
    table = Table(title=f"Audio samples ({len(samples)} shown)")
    table.add_column("Pack")
    table.add_column("Relative path")
    table.add_column("Emotion")
    table.add_column("Expected sentiment")
    for sample in samples:
        table.add_row(
            sample.pack_id,
            sample.relative_path,
            sample.metadata.emotion or "-",
            sample.expected_sentiment or "-",
        )
    console.print(table)


@audio_app.command("validate")
def audio_validate(
    audio_root: str | None = typer.Option(None, "--audio-root"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Validate manifest and on-disk sample packs."""
    catalog = load_catalog(audio_root)
    report = catalog.validate()
    if json_out:
        console.print_json(json.dumps(report.model_dump()))
        raise typer.Exit(code=0 if report.ok else 1)

    table = Table(title="Audio catalog validation")
    table.add_column("Pack")
    table.add_column("Active")
    table.add_column("Files")
    table.add_column("Parse failures")
    for pack_id, info in report.packs.items():
        table.add_row(
            pack_id,
            "yes" if info.get("active") else "no",
            str(info.get("file_count", 0)),
            str(info.get("parse_failures", 0)),
        )
    console.print(table)
    if report.errors:
        console.print("[red]Errors:[/red]")
        for err in report.errors:
            console.print(f"  - {err}")
    raise typer.Exit(code=0 if report.ok else 1)


@audio_app.command("smoke")
def audio_smoke(
    pack: str | None = typer.Option(None, "--pack"),
    device: str = typer.Option("cpu", "--device"),
    backend: str = typer.Option("faster", "--backend"),
    language: str | None = typer.Option(None, "--language"),
    output: str | None = typer.Option(None, "--output"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    audio_root: str | None = typer.Option(None, "--audio-root"),
) -> None:
    """Quick ASR smoke test on a small curated subset."""
    report = run_scenario(
        "smoke",
        audio_root=audio_root,
        pack_ids=[pack] if pack else None,
        device=device,
        backend=backend,
        language=language,
        dry_run=dry_run,
    )
    out = output or _default_output("smoke")
    _save_report(out, report.model_dump())
    if report.dry_run:
        console.print(f"[cyan]Smoke dry-run:[/cyan] {report.n_files} files selected")
    else:
        console.print(
            f"[cyan]Smoke complete:[/cyan] {report.summary.get('n_success', 0)}/{report.n_files} ok"
        )
    if report.errors:
        raise typer.Exit(code=1)


@audio_app.command("run")
def audio_run(
    scenario: str = typer.Option("asr", "--scenario", help=f"Scenario: {', '.join(SCENARIO_IDS)}"),
    pack: str | None = typer.Option(None, "--pack"),
    tags: str | None = typer.Option(None, "--tags"),
    emotions: str | None = typer.Option(None, "--emotions"),
    actors: str | None = typer.Option(None, "--actors"),
    limit: int | None = typer.Option(None, "--limit"),
    subset: str | None = typer.Option(None, "--subset"),
    device: str = typer.Option("cpu", "--device"),
    backend: str = typer.Option("faster", "--backend"),
    language: str | None = typer.Option(None, "--language"),
    output: str | None = typer.Option(None, "--output"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    audio_root: str | None = typer.Option(None, "--audio-root"),
) -> None:
    """Run a named audio benchmark scenario."""
    if scenario not in SCENARIO_IDS:
        console.print(f"[red]Unknown scenario: {scenario}[/red]")
        raise typer.Exit(code=1)

    report = run_scenario(
        scenario,
        audio_root=audio_root,
        pack_ids=[pack] if pack else None,
        tags=_split_csv(tags),
        emotions=_split_csv(emotions),
        actors=_split_csv(actors),
        limit=limit,
        subset=subset,
        device=device,
        backend=backend,
        language=language,
        dry_run=dry_run,
    )
    out = output or _default_output(scenario)
    _save_report(out, report.model_dump())
    console.print(f"[cyan]Scenario '{scenario}' complete[/cyan] in {report.duration_s}s")
    for key, value in report.summary.items():
        console.print(f"  {key}: {value}")
    if report.errors:
        raise typer.Exit(code=1)
