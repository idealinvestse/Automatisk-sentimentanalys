        ui.label("Strukturerade insikter (LLM + Fas4)").classes("text-subtitle2 q-mt-md")
        with ui.expansion("Sammanfattning & agentbedömning", icon="insights").classes("w-full"):
            ui.markdown(_build_insights_markdown(report))

        # LLM Judge panel (TASK-02)
        render_llm_judge_panel(report.get("results", {}).get("llm_judge"))

        if on_back: