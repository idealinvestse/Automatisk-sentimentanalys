# Test Lab / LLM Inställningar
# Lägger till knapp för att scanna OpenRouter + visa model catalog status

import contextlib
import json
from pathlib import Path

from nicegui import ui


def create_llm_model_settings() -> None:
    """LLM Model Catalog sektion i Test Lab eller Settings.

    Knapp som scannar OpenRouter, sparar katalog och visar info.
    Kan användas för att hålla pricing uppdaterad i openrouter_client.
    """
    with ui.card().classes("w-full"):
        ui.label("🤖 OpenRouter Model Catalog").classes("text-lg font-bold")
        ui.label("Scanna alla tillgängliga modeller + hämta live-pricing och info").classes(
            "text-sm text-gray-600"
        )

        status_label = ui.label("Katalog ej laddad än").classes("text-sm")

        def refresh_catalog():
            from src.llm.model_catalog import fetch_openrouter_models_catalog

            try:
                cat = fetch_openrouter_models_catalog()
                status_label.set_text(
                    f"✅ {cat['count']} modeller | Senast: {cat['scanned_at'][:16]} | "
                    f"Sparad till data/openrouter_models_catalog.json"
                )
                ui.notify(
                    "Model catalog uppdaterad! openrouter_client använder nu dynamisk pricing.",
                    type="positive",
                )
            except Exception as e:
                ui.notify(f"Fel vid scan: {e}", type="negative")

        ui.button(
            "🔄 Scanna OpenRouter modeller + uppdatera pricing",
            on_click=refresh_catalog,
            color="primary",
        ).classes("w-full mt-2")

        ui.button(
            "Visa senaste katalog (topp 5 billigaste)",
            on_click=lambda: show_top_models(),
        ).classes("w-full mt-1")

        def show_top_models():
            p = Path("data/openrouter_models_catalog.json")
            if not p.exists():
                ui.notify("Ingen katalog än – tryck på scanna först", type="warning")
                return
            with open(p) as f:
                cat = json.load(f)
            models = sorted(
                cat["models"], key=lambda m: m["pricing"]["prompt_per_million_usd"] or 999
            )[:5]
            with ui.dialog() as dlg, ui.card():
                ui.label("Top 5 billigaste modeller").classes("text-lg")
                for m in models:
                    p = m["pricing"]
                    ui.label(
                        f"{m['id']} | Prompt: ${p['prompt_per_million_usd']:.4f}/M | Completion: ${p['completion_per_million_usd']:.4f}/M"
                    ).classes("text-sm")
                ui.button("Stäng", on_click=dlg.close)
            dlg.open()


# Auto-registrera i Test Lab om importerad
with contextlib.suppress(Exception):
    create_llm_model_settings()  # Om körs utan NiceGUI context
