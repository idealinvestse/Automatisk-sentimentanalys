"""Analysis module that registers and manages text and conversation analyzers."""

from __future__ import annotations

from .base import Analyzer
from .insights import InsightsAnalyzer
from .intent import IntentAnalyzer
from .predictive import PredictiveAnalyzer
from .registry import get_registered_analyzers, run_analyzers

# Importing the adapters triggers decorator registration
from .sentiment import SentimentAnalyzer
from .summary import SummaryAnalyzer
from .topics import TopicAnalyzer

__all__ = [
    "Analyzer",
    "get_registered_analyzers",
    "run_analyzers",
    "SentimentAnalyzer",
    "IntentAnalyzer",
    "TopicAnalyzer",
    "SummaryAnalyzer",
    "InsightsAnalyzer",
    "PredictiveAnalyzer",
]
