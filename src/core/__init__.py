"""Core module containing shared data structures, configuration, device utilities, and custom errors."""

from .audio import resolve_audio_paths
from .serialization import (
    map_results_to_segment_dicts,
    score_dict,
    single_label_distribution,
    texts_from_segments,
    top_label,
    utc_now_iso,
)

__all__ = [
    "resolve_audio_paths",
    "utc_now_iso",
    "score_dict",
    "single_label_distribution",
    "top_label",
    "texts_from_segments",
    "map_results_to_segment_dicts",
]
