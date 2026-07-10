"""Quality evaluation scaffolding (MQM + preference gate)."""

from .mqm import (
    MqmAnnotation,
    MqmError,
    PreferenceGateResult,
    PreferencePair,
    evaluate_preference_gate,
)

__all__ = [
    "MqmAnnotation",
    "MqmError",
    "PreferenceGateResult",
    "PreferencePair",
    "evaluate_preference_gate",
]
