# Edge AI module for Fas 5 - On-device Swedish Call Center Analysis

from .contracts import EdgeAnalysisResult, EdgeSegmentResult
from .local_inference import analyze_segments_offline, analyze_text_offline

__all__ = [
    "EdgeAnalysisResult",
    "EdgeSegmentResult",
    "analyze_text_offline",
    "analyze_segments_offline",
]
