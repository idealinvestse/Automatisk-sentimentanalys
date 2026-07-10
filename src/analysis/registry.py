# Updated registry for INSIGHT-02 - LLM priority

from src.analysis.llm_judge import LLMJudgeAnalyzer

# ... existing code ...

# Set LLM as primary for key aspects
ANALYZERS = {
    'sentiment': LLMJudgeAnalyzer(priority=1),
    'emotion': LLMJudgeAnalyzer(priority=1),
    'intent': LLMJudgeAnalyzer(priority=1),
    # heuristics as fallback
}
