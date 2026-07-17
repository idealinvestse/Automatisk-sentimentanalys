import pytest

from src.transcription.oom_fallback import (
    OomFallbackResult,
    is_cuda_oom_error,
    transcribe_with_oom_fallback,
)


class FakeCudaOom(RuntimeError):
    def __init__(self) -> None:
        super().__init__("CUDA out of memory. Tried to allocate 512 MiB")


def test_is_cuda_oom_detects_message():
    assert is_cuda_oom_error(FakeCudaOom()) is True
    assert is_cuda_oom_error(ValueError("bad audio")) is False


def test_fallback_on_oom():
    calls: list[str] = []

    def fn(model: str) -> str:
        calls.append(model)
        if model == "kb-whisper-large":
            raise FakeCudaOom()
        return "ok"

    result = transcribe_with_oom_fallback(transcribe_fn=fn)
    assert isinstance(result, OomFallbackResult)
    assert result.value == "ok"
    assert result.model_used == "kb-whisper-medium"
    assert result.fell_back is True
    assert calls == ["kb-whisper-large", "kb-whisper-medium"]


def test_no_fallback_when_disabled():
    def fn(model: str) -> str:
        raise FakeCudaOom()

    with pytest.raises(RuntimeError, match="out of memory"):
        transcribe_with_oom_fallback(transcribe_fn=fn, allow_fallback=False)


def test_non_oom_propagates():
    def fn(model: str) -> str:
        raise ValueError("corrupt wav")

    with pytest.raises(ValueError, match="corrupt wav"):
        transcribe_with_oom_fallback(transcribe_fn=fn)
