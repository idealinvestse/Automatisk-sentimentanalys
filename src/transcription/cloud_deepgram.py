"""Deepgram cloud ASR adapter (opt-in via AsrRouter)."""

from __future__ import annotations

import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import httpx

from ..core.errors import TranscriptionError
from ..core.models import Segment, Transcript, Word
from .error_codes import AsrErrorCode

logger = logging.getLogger(__name__)

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-2"
_REQUEST_TIMEOUT_S = 120.0
_MAX_ATTEMPTS = 4
_BACKOFF_S = (0.5, 1.0, 2.0)
_LOW_CONFIDENCE_THRESHOLD = 0.60


def _resolve_api_key() -> str | None:
    return os.environ.get("DEEPGRAM_API_KEY") or os.environ.get("CLOUD_STT_API_KEY")


def map_deepgram_response(data: dict[str, Any]) -> list[Segment]:
    """Map a Deepgram listen API JSON body to :class:`Segment` objects."""
    channels = (data.get("results") or {}).get("channels") or []
    if not channels:
        return []

    alternatives = (channels[0].get("alternatives") or [])
    if not alternatives:
        return []

    alt = alternatives[0]
    text = str(alt.get("transcript") or "").strip()
    raw_words = alt.get("words") or []
    words = [
        Word(
            start=float(w.get("start", 0.0)),
            end=float(w.get("end", 0.0)),
            word=str(w.get("word", "")),
            prob=float(w.get("confidence", 0.0)),
        )
        for w in raw_words
    ]

    if not text and not words:
        return []

    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    avg_conf = sum(w.prob for w in words) / len(words) if words else None
    confidence = avg_conf
    low_confidence = avg_conf is not None and avg_conf < _LOW_CONFIDENCE_THRESHOLD

    return [
        Segment(
            start=start,
            end=end,
            text=text,
            words=words,
            avg_confidence=avg_conf,
            confidence=confidence,
            low_confidence=low_confidence,
        )
    ]


class DeepgramTranscriber:
    """Cloud ASR via Deepgram pre-recorded REST API."""

    def transcribe(
        self,
        audio_path: str,
        language: str = "sv",
        **_kwargs: object,
    ) -> Transcript:
        api_key = _resolve_api_key()
        if not api_key:
            raise TranscriptionError(
                "Deepgram API key missing. Set DEEPGRAM_API_KEY or CLOUD_STT_API_KEY.",
                error_code=AsrErrorCode.CLOUD_AUTH,
            )

        logger.info("asr_cloud_egress=true provider=deepgram")

        path = Path(audio_path)
        if not path.is_file():
            raise TranscriptionError(
                f"Audio file not found: {audio_path}",
                error_code=AsrErrorCode.PREPROCESS_FAILED,
            )

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        params = {
            "model": DEEPGRAM_MODEL,
            "language": language,
            "punctuate": "true",
            "words": "true",
        }
        headers = {"Authorization": f"Token {api_key}"}

        t0 = time.time()
        data = self._post_with_retries(path, content_type, params, headers)
        segments = map_deepgram_response(data)
        duration = (data.get("metadata") or {}).get("duration")
        processing_time = time.time() - t0

        return Transcript(
            model=DEEPGRAM_MODEL,
            backend="deepgram",
            language=language,
            duration=float(duration) if duration is not None else None,
            processing_time=processing_time,
            segments=segments,
        )

    def _post_with_retries(
        self,
        path: Path,
        content_type: str,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        last_error: TranscriptionError | None = None

        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                time.sleep(_BACKOFF_S[attempt - 1])

            try:
                with path.open("rb") as audio_file:
                    response = httpx.post(
                        DEEPGRAM_LISTEN_URL,
                        params=params,
                        headers=headers,
                        content=audio_file,
                        timeout=_REQUEST_TIMEOUT_S,
                    )
            except httpx.TimeoutException as exc:
                last_error = TranscriptionError(
                    "Deepgram request timed out.",
                    error_code=AsrErrorCode.CLOUD_TIMEOUT,
                )
                if attempt < _MAX_ATTEMPTS - 1:
                    continue
                raise last_error from exc
            except httpx.TransportError as exc:
                last_error = TranscriptionError(
                    f"Deepgram transport error: {exc}",
                    error_code=AsrErrorCode.CLOUD_TIMEOUT,
                )
                if attempt < _MAX_ATTEMPTS - 1:
                    continue
                raise last_error from exc

            if response.status_code in (401, 403):
                raise TranscriptionError(
                    "Deepgram authentication failed.",
                    error_code=AsrErrorCode.CLOUD_AUTH,
                )

            if response.status_code == 429:
                last_error = TranscriptionError(
                    "Deepgram rate limit exceeded.",
                    error_code=AsrErrorCode.CLOUD_QUOTA,
                )
                if attempt < _MAX_ATTEMPTS - 1:
                    continue
                raise last_error

            if response.status_code >= 500:
                last_error = TranscriptionError(
                    f"Deepgram server error ({response.status_code}).",
                    error_code=AsrErrorCode.CLOUD_TIMEOUT,
                )
                if attempt < _MAX_ATTEMPTS - 1:
                    continue
                raise last_error

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise TranscriptionError(
                    f"Deepgram request failed ({response.status_code}).",
                    error_code=AsrErrorCode.CLOUD_AUTH,
                ) from exc

            return response.json()

        if last_error is not None:
            raise last_error
        raise TranscriptionError(
            "Deepgram request failed after retries.",
            error_code=AsrErrorCode.CLOUD_TIMEOUT,
        )
