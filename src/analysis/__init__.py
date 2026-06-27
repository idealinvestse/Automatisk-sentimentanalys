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
from .aspect import AspectAnalyzer  # Task 1.5
from .emotion import EmotionAnalyzer  # Task 2.1
from .role_classifier import RoleAnalyzer  # Task 2.2
from .trajectory import TrajectoryAnalyzer  # Task 2.3
from .llm_judge import LLMJudgeAnalyzer  # Task 2.4
from .spoken_normalizer import SpokenNormalizerAnalyzer  # Task 2.5

# New high-value analyzers (added 2026-06-27)
from .customer_effort import CustomerEffortScoreAnalyzer
from .compliance_risk import ComplianceRiskAnalyzer
from .active_listening import ActiveListeningBehaviorAnalyzer

# Revenue & outcome focused (added 2026-06-27)
from .upsell_opportunity import UpsellOpportunityDetector
from .resolution_probability import ResolutionProbabilityPredictor

# Journey & dialect (added 2026-06-27)
from .multi_turn_journey import MultiTurnJourneyMapper
from .dialect_sensitivity import DialectSensitivityAnalyzer

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
    "AspectAnalyzer",
    "EmotionAnalyzer",
    "RoleAnalyzer",
    "TrajectoryAnalyzer",
    "LLMJudgeAnalyzer",
    "SpokenNormalizerAnalyzer",
    "CustomerEffortScoreAnalyzer",
    "ComplianceRiskAnalyzer",
    "ActiveListeningBehaviorAnalyzer",
    "UpsellOpportunityDetector",
    "ResolutionProbabilityPredictor",
    "MultiTurnJourneyMapper",
    "DialectSensitivityAnalyzer",
]
