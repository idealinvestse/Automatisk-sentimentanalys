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
_CLOUD_FALLBACK_ERRORS = frozenset({AsrErrorCode.CLOUD_TIMEOUT, AsrErrorCode.CLOUD_QUOTA})


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
            try:
                transcript = self._transcribe_cloud(
                    audio_path,
                    cloud_provider=cloud_provider,
                    **kwargs,
                )
            except TranscriptionError as exc:
                if cloud_fallback_local and exc.error_code in _CLOUD_FALLBACK_ERRORS:
                    transcript = self._transcribe_local(
                        audio_path,
                        backend=backend,
                        model_name=model_name,
                        device=device,
                        **kwargs,
                    )
                    return replace(transcript, provider="local")
                raise
            transcript = filter_hallucinations(transcript)
            return replace(transcript, provider=resolved)

        transcript = self._transcribe_local(
            audio_path,
            backend=backend,
            model_name=model_name,
            device=device,
            **kwargs,
        )
        return replace(transcript, provider=resolved)

    def _transcribe_local(
        self,
        audio_path: str,
        *,
        backend: str,
        model_name: str,
        device: str,
        **kwargs: object,
    ) -> Transcript:
        transcriber = get_transcriber(
            backend=backend, model_name=model_name, device=device
        )
        transcript = transcriber.transcribe(audio_path, **kwargs)  # type: ignore[arg-type]
        return filter_hallucinations(transcript)

    def _transcribe_cloud(
        self,
        audio_path: str,
        *,
        cloud_provider: str,
        **kwargs: object,
    ) -> Transcript:
        if cloud_provider != "deepgram":
            raise TranscriptionError(
                f"Unsupported cloud ASR provider '{cloud_provider}'. "
                "Only 'deepgram' is available.",
                error_code=AsrErrorCode.CLOUD_AUTH,
            )
        from .cloud_deepgram import DeepgramTranscriber

        return DeepgramTranscriber().transcribe(audio_path, **kwargs)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_asr_router() -> AsrRouter:
    """Return a cached :class:`AsrRouter` instance."""
    return AsrRouter()


def transcribe_with_router(audio_path: str, **kwargs: object) -> Transcript:
    """Convenience wrapper around :meth:`AsrRouter.transcribe`."""
    return get_asr_router().transcribe(audio_path, **kwargs)
