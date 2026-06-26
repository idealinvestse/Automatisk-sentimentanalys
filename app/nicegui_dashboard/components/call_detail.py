from app.nicegui_dashboard.components.llm_judge_panel import render_llm_judge_panel
from app.nicegui_dashboard.components.virtual_transcript import (
    render_timeline,
    render_transcript_panel,
    scroll_transcript_to_index,
)
from app.nicegui_dashboard.services.qa_display import qa_chip_color
from app.nicegui_dashboard.services.transcript_virtualizer import filter_segments_with_index
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import enrich_segments_with_sentiment, get_overall_sentiment