"""Optional audio preprocessing for ASR.

Provides valbar brusreducering före transkription för bättre WER på bullriga
callcenter-inspelningar (t.ex. bakgrundsljud, lågfrekvent brus från kontor).

Stödjer:
- High-pass filter via ffmpeg (redan i Dockerfile) – tar bort lågfrekvent
  rumble (<80-100 Hz) som stör voice activity och ordigenkänning.
- Noise reduction via noisereduce (valfritt paket) – stationär brusreducering.

Användning:
    from .preprocess import preprocess_audio
    clean_path = preprocess_audio("noisy_call.wav", highpass=True, noise_reduction=True)

Integreras i alla backends via `preprocess=True` i transcribe().

Designval:
- Använder ffmpeg för highpass eftersom det är systemberoende som redan finns
  och är mycket pålitligt för audio pipelines.
- noisereduce är extra (kräver `pip install noisereduce`).
- Returnerar alltid en (temp) wav-fil som kan matas direkt till
  faster-whisper / whisperx / transformers.
- Temporära filer lämnas (OS städar så småningom); för produktion kan man
  använda NamedTemporaryFile med delete=False + explicit cleanup.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _numpy_to_wav(audio: np.ndarray, output_path: str, sr: int = 16000) -> None:
    """Write float32 mono audio to WAV using ffmpeg pipe (no extra Python audio deps)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "f32le",
        "-ar", str(sr),
        "-ac", "1",
        "-i", "pipe:0",
        "-loglevel", "error",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate(input=audio.astype(np.float32).tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to encode WAV with ffmpeg: {proc.stderr.read().decode()}")


def preprocess_audio(
    audio_path: str,
    highpass: bool = True,
    noise_reduction: bool = True,
    highpass_freq: int = 100,
) -> str:
    """Apply optional preprocessing and return path to cleaned audio (WAV @16k mono).

    Args:
        audio_path: Input audio file.
        highpass: Apply high-pass filter via ffmpeg.
        noise_reduction: Apply noisereduce if the package is installed.
        highpass_freq: Cutoff frequency for high-pass (Hz).

    Returns:
        Path to a (temporary) WAV file with preprocessing applied.
        Caller is responsible for cleanup if desired.
    """
    current_path = audio_path
    tmp_files: list[str] = []

    try:
        # 1. High-pass filter (always via ffmpeg for robustness)
        if highpass:
            fd, highpass_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_files.append(highpass_path)

            cmd = [
                "ffmpeg", "-y",
                "-i", current_path,
                "-af", f"highpass=f={highpass_freq}",
                "-ar", "16000",
                "-ac", "1",
                "-loglevel", "error",
                highpass_path,
            ]
            logger.debug("Applying high-pass filter: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("High-pass ffmpeg failed: %s", result.stderr)
                # fall back to original
                highpass_path = current_path
            current_path = highpass_path

        # 2. Noise reduction (optional, requires noisereduce)
        if noise_reduction:
            try:
                import noisereduce as nr  # type: ignore

                # Decode current (possibly highpassed) audio
                from faster_whisper.audio import decode_audio  # type: ignore

                audio = decode_audio(current_path)
                logger.debug("Applying noisereduce (stationary) on %d samples", len(audio))

                reduced = nr.reduce_noise(y=audio, sr=16000, stationary=True, prop_decrease=0.8)

                # Encode back to temp wav
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

        return current_path

    except Exception as e:
        # Cleanup any temps we created on error
        for p in tmp_files:
            if p != audio_path and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass
        raise RuntimeError(f"Preprocessing failed: {e}") from e


def maybe_preprocess(audio_path: str, preprocess: bool = False) -> str:
    """Convenience wrapper: if preprocess=True then run default preprocessing."""
    if not preprocess:
        return audio_path
    return preprocess_audio(audio_path, highpass=True, noise_reduction=True)
