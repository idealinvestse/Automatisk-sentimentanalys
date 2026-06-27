# EmpathyScoringAnalyzer (New – TASK-B1)
from .base import Analyzer

@register_analyzer("empathy")
class EmpathyScoringAnalyzer(Analyzer):
    requires = ["sentiment", "negation"]
    def analyze(self, ctx):
        # Logic: Empati per segment + trajectory
        # Combines polite terms, negation handling, active listening cues
        score = calculate_empathy(ctx.segments)
        return {"empathy_score": score, "tips": ["Använd mer validering i steg 2"], "evidence": [...]}