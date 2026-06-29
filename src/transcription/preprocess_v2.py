"""Call-center optimised audio preprocessing for ASR (Transcription v2, Task A-1).

Extends :mod:`preprocess` with a chain tuned for Swedish telephone / VoIP
recordings:

1. Resample to 16 kHz mono (ffmpeg)
2. Bandpass 100–3400 Hz (telephone bandwidth; removes rumble and high hiss)
3. Dynamic loudness normalization (dynaudnorm – evens agent/customer levels)
4. Optional stationary noise reduction (noisereduce, gentler than v1)

Returns the same :class:`~preprocess.PreprocessHandle` contract as v1.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from .preprocess import PreprocessHandle, _cleanup_paths, _numpy_to_wav

logger = logging.getLogger(__name__)

# Gentler than v1 (0.8) to preserve narrowband speech intelligibility.
_NOISE_REDUCE_PROP_DECREASE = 0.65

# Single-pass ffmpeg filter chain for call-center narrowband audio.
_FFMPEG_CALLCENTER_AF = "highpass=f=100,lowpass=f=3400,dynaudnorm=f=75:g=15:p=0.95"


def preprocess_audio_callcenter(
    audio_path: str,
    *,
    noise_reduction: bool = True,
) -> PreprocessHandle:
    """Apply call-center preprocessing and return a handle to cleaned audio."""
    current_path = audio_path
    tmp_files: list[str] = []

    try:
        fd, bandpass_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_files.append(bandpass_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            current_path,
            "-af",
            _FFMPEG_CALLCENTER_AF,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-loglevel",
            "error",
            bandpass_path,
        ]
        logger.debug("Applying call-center bandpass+normalize: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Call-center ffmpeg preprocess failed: %s. Falling back to original audio.",
                result.stderr,
            )
            bandpass_path = current_path
        else:
            current_path = bandpass_path

        if noise_reduction:
            try:
                import noisereduce as nr  # type: ignore
                from faster_whisper.audio import decode_audio  # type: ignore

                audio = decode_audio(current_path)
                logger.debug(
                    "Applying call-center noisereduce on %d samples (prop=%.2f)",
                    len(audio),
                    _NOISE_REDUCE_PROP_DECREASE,
                )
                reduced = nr.reduce_noise(
                    y=audio,
                    sr=16000,
                    stationary=True,
                    prop_decrease=_NOISE_REDUCE_PROP_DECREASE,
                )

                fd, nr_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                tmp_files.append(nr_path)
                _numpy_to_wav(reduced, nr_path)
                current_path = nr_path
                logger.info("Call-center noise reduction applied")

            except ImportError:
                logger.info(
                    "noisereduce not installed – skipping call-center noise reduction. "
                    "Install with: pip install noisereduce"
                )
            except Exception as exc:
                logger.warning("Call-center noise reduction failed, continuing without: %s", exc)

        temps_to_clean = [p for p in tmp_files if p != audio_path and os.path.exists(p)]
        return PreprocessHandle(
            path=current_path,
            _temp_paths=temps_to_clean,
            _original_path=audio_path,
        )

    except Exception as exc:
        _cleanup_paths(tmp_files, exclude=audio_path)
        raise RuntimeError(f"Call-center preprocessing failed: {exc}") from exc
