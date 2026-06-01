"""Device normalization utilities for PyTorch, Transformers, and Faster-Whisper."""

from __future__ import annotations

import torch


def normalize_device_spec(
    device: int | str | torch.device | None,
) -> tuple[int | torch.device, str]:
    """Normalize device specification for Hugging Face Transformers.

    Returns:
        (device_arg_for_pipeline, device_key_for_cache)
        where device_arg_for_pipeline is an integer (e.g., 0 for GPU, -1 for CPU)
        or a torch.device instance (e.g., MPS).
    """
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        if torch.cuda.is_available():
            return 0, "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return -1, "cpu"

    if isinstance(device, int):
        if device >= 0 and torch.cuda.is_available():
            if device < torch.cuda.device_count():
                return device, f"cuda:{device}"
            return 0, "cuda:0"
        return -1, "cpu"

    if isinstance(device, torch.device):
        if device.type == "cuda" and torch.cuda.is_available():
            idx = device.index if device.index is not None else 0
            if idx < torch.cuda.device_count():
                return idx, f"cuda:{idx}"
            return 0, "cuda:0"
        if (
            device.type == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return device, "mps"
        return -1, "cpu"

    if isinstance(device, str):
        d = device.strip().lower()
        if d == "cpu":
            return -1, "cpu"
        if d.startswith("cuda"):
            idx = 0
            if ":" in d:
                try:
                    idx = int(d.split(":", 1)[1])
                except ValueError:
                    idx = 0
            if torch.cuda.is_available():
                if idx < torch.cuda.device_count():
                    return idx, f"cuda:{idx}"
                return 0, "cuda:0"
            return -1, "cpu"
        if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return -1, "cpu"

    return -1, "cpu"


def device_arg_from_key(key: str) -> int | torch.device:
    """Reconstruct transformers device argument from cache key string."""
    if key == "cpu":
        return -1
    if key == "mps":
        return torch.device("mps")
    if key.startswith("cuda:"):
        try:
            idx = int(key.split(":", 1)[1])
            return idx
        except ValueError:
            return 0
    return -1


def normalize_device_for_asr(device: str | None = "auto") -> tuple[str, int | None]:
    """Normalize device specification for Faster-Whisper or Transformers ASR.

    Returns:
        (device_string, cuda_index_or_none)
        where device_string is 'cuda', 'mps', or 'cpu'.
    """
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return "cuda", 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", None
        return "cpu", None

    d = str(device).lower().strip()
    if d.startswith("cuda"):
        idx = 0
        if ":" in d:
            try:
                idx = int(d.split(":", 1)[1])
            except ValueError:
                idx = 0
        if torch.cuda.is_available() and idx < torch.cuda.device_count():
            return "cuda", idx
        if torch.cuda.is_available():
            return "cuda", 0
        return "cpu", None
    if d == "mps":
        return "mps", None
    return "cpu", None
