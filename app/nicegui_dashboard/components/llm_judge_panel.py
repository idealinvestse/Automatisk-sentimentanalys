        # Header with stats + filter
        with ui.row().classes("items-center justify-between w-full mb-2"):
            ui.markdown(f"**{total}** segment bedömdes | **{changed_count}** ändrade").classes("text-sm text-gray-600")

            with ui.row().classes("items-center gap-1"):
                filter_mode = ui.toggle(
                    ["Alla", "Endast ändrade"],
                    value="Alla",
                    on_change=lambda e: refresh_panel.refresh(),
                ).props("dense toggle-color=primary").classes("text-sm")

                ui.icon("help_outline", size="xs").classes("text-grey cursor-help").tooltip(
                    "Visar endast segment där LLM:en ändrade den ursprungliga sentiment-bedömningen (t.ex. neutral → negative)."
                )