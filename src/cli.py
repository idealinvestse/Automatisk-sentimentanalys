

@app.command("scan-openrouter-models")
def scan_openrouter_models_cmd(
    output: str = typer.Option(
        "data/openrouter_models_catalog.json",
        "--output", "-o",
        help="Sökväg att spara katalogen till (används av dashboard och client)",
    ),
    show_top: int = typer.Option(
        10,
        "--show-top", "-n",
        help="Visa de N billigaste modellerna i tabell",
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    """Scanna OpenRouter efter alla tillgängliga modeller och spara katalog med kostnad + kort info.

    Detta använder model_catalog.py och gör det enkelt att hålla pricing uppdaterad.
    Används av openrouter_client för dynamisk PRICING.
    """
    from src.llm.model_catalog import fetch_openrouter_models_catalog, load_catalog
    from rich.table import Table

    setup_logging(log_level)
    console.print("[cyan]Scannar OpenRouter efter alla modeller...[/cyan]")

    try:
        catalog = fetch_openrouter_models_catalog(output_path=output)
    except Exception as e:
        console.print(f"[red]Kunde inte scanna: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✅ Sparade {catalog['count']} modeller till {output}[/green]")
    console.print(f"Scannad: {catalog['scanned_at']}")

    # Visa de billigaste (prompt) modellerna
    models = sorted(
        catalog["models"],
        key=lambda m: m["pricing"]["prompt_per_million_usd"] or 999,
    )[:show_top]

    table = Table(title=f"Top {show_top} billigaste modeller (prompt per miljon tokens)")
    table.add_column("ID / Namn", style="cyan")
    table.add_column("Prompt $/M", justify="right")
    table.add_column("Completion $/M", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Kort info", style="dim")

    for m in models:
        p = m["pricing"]
        desc = m["description"][:80] + "..." if len(m["description"]) > 80 else m["description"]
        table.add_row(
            m["id"],
            f"{p['prompt_per_million_usd']:.4f}",
            f"{p['completion_per_million_usd']:.4f}",
            str(m.get("context_length", "-")) or "-",
            desc,
        )

    console.print(table)
    console.print("[yellow]Använd katalogen i dashboard eller openrouter_client för dynamisk pricing.[/yellow]")


if __name__ == "__main__":
    app()
