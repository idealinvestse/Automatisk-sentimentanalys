        # Fas 3 viz: LLM-judge breakdown (only if verdicts or demo data present)
        from app.nicegui_dashboard.components.llm_judge_panel import render_llm_judge_panel
        render_llm_judge_panel(report.get("results", {}).get("llm_judge"))