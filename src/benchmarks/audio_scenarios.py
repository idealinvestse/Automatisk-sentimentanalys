"""Predefined audio benchmark scenarios and sample selection."""

from __future__ import annotations

from .audio_catalog import AudioCatalog
from .audio_models import AudioSample, SampleFilter, ScenarioId

SCENARIO_IDS: tuple[ScenarioId, ...] = (
    "catalog",
    "smoke",
    "asr",
    "pipeline",
    "sentiment_chain",
    "language_sanity",
)

SCENARIO_DEFAULTS: dict[ScenarioId, dict[str, object]] = {
    "catalog": {"subset": None, "limit": None, "requires_ml": False},
    "smoke": {"subset": "smoke_subset", "limit": 3, "requires_ml": True},
    "asr": {"subset": "emotion_coverage", "limit": 8, "requires_ml": True},
    "pipeline": {"subset": "smoke_subset", "limit": 2, "requires_ml": True},
    "sentiment_chain": {"subset": "emotion_coverage", "limit": 8, "requires_ml": True},
    "language_sanity": {"subset": "one_per_pack", "limit": None, "requires_ml": True},
}


def resolve_samples(
    catalog: AudioCatalog,
    scenario: ScenarioId,
    *,
    pack_ids: list[str] | None = None,
    tags: list[str] | None = None,
    emotions: list[str] | None = None,
    actors: list[str] | None = None,
    limit: int | None = None,
    subset: str | None = None,
) -> list[AudioSample]:
    defaults = SCENARIO_DEFAULTS[scenario]
    effective_subset = subset if subset is not None else defaults.get("subset")  # type: ignore[arg-type]
    effective_limit = limit if limit is not None else defaults.get("limit")  # type: ignore[arg-type]

    if scenario == "catalog":
        return catalog.discover(
            SampleFilter(
                pack_ids=pack_ids,
                tags=tags,
                emotions=emotions,
                actors=actors,
                limit=effective_limit if isinstance(effective_limit, int) else None,
            )
        )

    return catalog.discover(
        SampleFilter(
            pack_ids=pack_ids,
            tags=tags,
            emotions=emotions,
            actors=actors,
            limit=effective_limit if isinstance(effective_limit, int) else None,
            subset=effective_subset if isinstance(effective_subset, str) else None,
        )
    )


def scenario_requires_ml(scenario: ScenarioId) -> bool:
    return bool(SCENARIO_DEFAULTS[scenario]["requires_ml"])
