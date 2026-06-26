"""Optional audio preprocessing for ASR.

Provides valbar brusreducering före transkription för bättre WER på bullriga
callcenter-inspelningar (t.ex. bakgrundsljud, lågfrekvent brus från kontor).

Stödjer:
- High-pass filter via ffmpeg (redan i Dockerfile) – tar bort lågfrekvent
  rumble (<80-100 Hz) som stör voice activity och ordigenkänning.
- Noise reduction via noisereduce (valfritt paket) – stationär brusreducering.

Användning:
    from .preprocess import maybe_preprocess

    prep = maybe_preprocess("noisy_call.wav", preprocess=True)
    try:
        transcriber.transcribe(prep.path, ...)
    finally:
        prep.cleanup()

Integreras i alla backends via ``preprocess=True`` i transcribe().

Designval:
- Använder ffmpeg för highpass eftersom det är systemberoende som redan finns
  och är mycket pålitligt för audio pipelines.
- noisereduce är extra (kräver ``pip install noisereduce``).
- Returnerar en :class:`PreprocessHandle` med sökväg till (temp) wav-fil.
- Temporära filer städas via ``handle.cleanup()`` eller context manager –
  anroparen (ASR-backend) ansvarar för cleanup i ``finally`` efter transkribering.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np

logger = logging.getLogger(__name__)

PreprocessMode = Literal["off", "basic", "callcenter"]

_MODE_ALIASES: dict[str, PreprocessMode] = {
    "": "off",
    "off": "off",
    "false": "off",
    "none": "off",
    "0": "off",
    "basic": "basic",
    "true": "basic",
    "legacy": "basic",
    "v1": "basic",
    "1": "basic",
    "callcenter": "callcenter",
    "call_center": "callcenter",
    "cc": "callcenter",
    "v2": "callcenter",
}


@dataclass
class PreprocessHandle:
    """Handle to preprocessed audio; call ``cleanup()`` when ASR is done."""

    path: str
    _temp_paths: list[str] = field(default_factory=list)
    _original_path: str = ""

    def cleanup(self) -> None:
        """Remove temporary WAV files created during preprocessing."""
        for p in self._temp_paths:
            if not p or p == self._original_path:
                continue
            if os.path.exists(p):
                try:
                    os.unlink(p)
                    logger.debug("Removed preprocessed temp file: %s", p)
                except OSError as err:
                    logger.warning("Could not remove temp file %s: %s", p, err)
        self._temp_paths.clear()

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()


def _numpy_to_wav(audio: np.ndarray, output_path: str, sr: int = 16000) -> None:
    """Write float32 mono audio to WAV using ffmpeg pipe (no extra Python audio deps)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-loglevel",
        "error",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate(input=audio.astype(np.float32).tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to encode WAV with ffmpeg: {proc.stderr.read().decode()}")


def _cleanup_paths(paths: list[str], *, exclude: str) -> None:
    for p in paths:
        if p and p != exclude and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


def preprocess_audio(
    audio_path: str,
    highpass: bool = True,
    noise_reduction: bool = True,
    highpass_freq: int = 100,
) -> PreprocessHandle:
    """Apply optional preprocessing and return a handle to cleaned audio (WAV @16k mono).

    Args:
        audio_path: Input audio file.
        highpass: Apply high-pass filter via ffmpeg.
        noise_reduction: Apply noisereduce if the package is installed.
        highpass_freq: Cutoff frequency for high-pass (Hz).

    Returns:
        PreprocessHandle whose ``path`` is the audio to feed to ASR.
        Caller must call ``handle.cleanup()`` (or use ``finally``) after transcription.
    """
    current_path = audio_path
    tmp_files: list[str] = []

    try:
        if highpass:
            fd, highpass_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_files.append(highpass_path)

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                current_path,
                "-af",
                f"highpass=f={highpass_freq}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-loglevel",
                "error",
                highpass_path,
            ]
            logger.debug("Applying high-pass filter: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("High-pass ffmpeg failed: %s", result.stderr)
                highpass_path = current_path
            else:
                current_path = highpass_path

        if noise_reduction:
            try:
                import noisereduce as nr  # type: ignore

                from faster_whisper.audio import decode_audio  # type: ignore

                audio = decode_audio(current_path)
                logger.debug("Applying noisereduce (stationary) on %d samples", len(audio))

                reduced = nr.reduce_noise(y=audio, sr=16000, stationary=True, prop_decrease=0.8)

                fd, nr_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                tmp_files.append(nr_path)

                _numpy_to_wav(reduced, nr_path)
                current_path = nr_path

                logger.info("Noise reduction applied successfully")

            except ImportError:
                logger.info(
                    "noisereduce not installed – skipping noise reduction. "
                    "Install with: pip install noisereduce (optional for Task 1.4)"
                )
            except Exception as e:
                logger.warning("Noise reduction failed, continuing without: %s", e)

        temps_to_clean = [p for p in tmp_files if p != audio_path and os.path.exists(p)]
        return PreprocessHandle(
            path=current_path,
            _temp_paths=temps_to_clean,
            _original_path=audio_path,
        )

    except Exception as e:
        _cleanup_paths(tmp_files, exclude=audio_path)
        raise RuntimeError(f"Preprocessing failed: {e}") from e


def normalize_preprocess_mode(
    *,
    preprocess: bool = False,
    preprocess_mode: str | PreprocessMode | None = None,
) -> PreprocessMode:
    """Resolve preprocessing mode from explicit mode or legacy boolean flag."""
    if preprocess_mode is not None:
        key = str(preprocess_mode).strip().lower()
        resolved = _MODE_ALIASES.get(key)
        if resolved is not None:
            return resolved
        logger.warning("Unknown preprocess_mode '%s'; defaulting to off", preprocess_mode)
        return "off"
    return "basic" if preprocess else "off"


def maybe_preprocess_for_mode(
    audio_path: str,
    mode: PreprocessMode | str,
) -> PreprocessHandle:
    """Return a handle to preprocessed audio for the given mode."""
    resolved = normalize_preprocess_mode(preprocess_mode=str(mode))
    if resolved == "off":
        return PreprocessHandle(path=audio_path, _temp_paths=[], _original_path=audio_path)
    if resolved == "callcenter":
        from .preprocess_v2 import preprocess_audio_callcenter

        return preprocess_audio_callcenter(audio_path)
    return preprocess_audio(audio_path, highpass=True, noise_reduction=True)


def maybe_preprocess(audio_path: str, preprocess: bool = False) -> PreprocessHandle:
    """Return a handle to preprocessed audio, or the original path if disabled."""
    return maybe_preprocess_for_mode(
        audio_path,
        normalize_preprocess_mode(preprocess=preprocess),
    )


def prepare_asr_audio(
    audio_path: str,
    *,
    preprocess: bool = False,
    preprocess_mode: str | PreprocessMode | None = None,
) -> tuple[PreprocessHandle, PreprocessMode]:
    """Resolve mode, preprocess audio, and fall back to original on failure."""
    mode = normalize_preprocess_mode(preprocess=preprocess, preprocess_mode=preprocess_mode)
    if mode == "off":
        return (
            PreprocessHandle(path=audio_path, _temp_paths=[], _original_path=audio_path),
            mode,
        )
    try:
        handle = maybe_preprocess_for_mode(audio_path, mode)
        logger.info("Preprocessing enabled | mode=%s | path=%s", mode, handle.path)
        return handle, mode
    except Exception as exc:
        logger.warning("Preprocessing failed (mode=%s), falling back to original audio: %s", mode, exc)
        return (
            PreprocessHandle(path=audio_path, _temp_paths=[], _original_path=audio_path),
            "off",
        )


@contextmanager
def managed_preprocess(
    audio_path: str,
    *,
    preprocess: bool = False,
    highpass: bool = True,
    noise_reduction: bool = True,
) -> Iterator[str]:
    """Context manager that yields ASR audio path and cleans up temps on exit."""
    if preprocess:
        handle = preprocess_audio(
            audio_path,
            highpass=highpass,
            noise_reduction=noise_reduction,
        )
    else:
        handle = PreprocessHandle(path=audio_path, _temp_paths=[], _original_path=audio_path)
    try:
        yield handle.path
    finally:
        handle.cleanup()