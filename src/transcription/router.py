"""ASR provider routing (local default, cloud opt-in)."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Literal

from ..core.errors import TranscriptionError
from ..core.models import Transcript
from .error_codes import AsrErrorCode
from .factory import get_transcriber
from .postprocess import filter_hallucinations

ProviderName = Literal["local", "cloud"]

_VALID_PROVIDERS = frozenset({"local", "cloud"})


def resolve_asr_provider(provider: str | None) -> ProviderName:
    """Resolve ASR provider; defaults to local for None or empty string."""
    if provider is None or str(provider).strip() == "":
        return "local"
    normalized = str(provider).strip().lower()
    if normalized not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown ASR provider '{provider}'. Supported: 'local', 'cloud'"
        )
    return normalized  # type: ignore[return-value]


class AsrRouter:
    """Route transcription to local or cloud ASR engines."""

    def transcribe(
        self,
        audio_path: str,
        *,
        provider: str = "local",
        backend: str = "faster",
        model_name: str = "kb-whisper-large",
        device: str = "auto",
        cloud_fallback_local: bool = False,
        cloud_provider: str = "deepgram",
        **kwargs: object,
    ) -> Transcript:
        resolved = resolve_asr_provider(provider)

        if resolved == "cloud":
            raise TranscriptionError(
                "Cloud STT (Deepgram) is not configured in this build. "
                "Use provider='local' or enable cloud-stt (see Task 6).",
                error_code=AsrErrorCode.CLOUD_AUTH,
            )

        transcriber = get_transcriber(
            backend=backend, model_name=model_name, device=device
        )
        transcript = transcriber.transcribe(audio_path, **kwargs)  # type: ignore[arg-type]
        transcript = filter_hallucinations(transcript)
        return replace(transcript, provider=resolved)


@lru_cache(maxsize=1)
def get_asr_router() -> AsrRouter:
    """Return a cached :class:`AsrRouter` instance."""
    return AsrRouter()


def transcribe_with_router(audio_path: str, **kwargs: object) -> Transcript:
    """Convenience wrapper around :meth:`AsrRouter.transcribe`."""
    return get_asr_router().transcribe(audio_path, **kwargs)
