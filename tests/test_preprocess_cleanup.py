"""Tests for preprocess temp-file cleanup (Task 1.4 disk leak fix)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.transcription.preprocess import (
    PreprocessHandle,
    maybe_preprocess,
    preprocess_audio,
)


def test_preprocess_handle_cleanup_removes_temps(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")

    temp1 = tmp_path / "temp1.wav"
    temp2 = tmp_path / "temp2.wav"
    temp1.write_bytes(b"t1")
    temp2.write_bytes(b"t2")

    handle = PreprocessHandle(
        path=str(temp2),
        _temp_paths=[str(temp1), str(temp2)],
        _original_path=str(audio),
    )
    handle.cleanup()

    assert not temp1.exists()
    assert not temp2.exists()
    assert audio.exists()


def test_preprocess_handle_cleanup_idempotent(tmp_path):
    missing = tmp_path / "gone.wav"
    handle = PreprocessHandle(path=str(missing), _temp_paths=[str(missing)])
    handle.cleanup()
    handle.cleanup()


def test_maybe_preprocess_noop_does_not_delete_original(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")
    handle = maybe_preprocess(str(audio), preprocess=False)
    assert handle.path == str(audio)
    handle.cleanup()
    assert audio.exists()


@patch("src.transcription.preprocess.subprocess.run")
def test_preprocess_audio_highpass_creates_and_cleans_temp(mock_run, tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")

    created: list[str] = []

    def fake_run(cmd, **kwargs):
        out_path = cmd[-1]
        created.append(out_path)
        with open(out_path, "wb") as f:
            f.write(b"wav")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    handle = preprocess_audio(str(audio), highpass=True, noise_reduction=False)
    assert handle.path in created
    assert os.path.exists(handle.path)

    handle.cleanup()
    for p in created:
        assert not os.path.exists(p)


@patch("src.transcription.preprocess.subprocess.run")
def test_preprocess_audio_error_cleans_partial_temps(mock_run, tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")

    def fail_run(cmd, **kwargs):
        out_path = cmd[-1]
        with open(out_path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("ffmpeg boom")

    mock_run.side_effect = fail_run

    with pytest.raises(RuntimeError, match="Preprocessing failed"):
        preprocess_audio(str(audio), highpass=True, noise_reduction=False)

    temps = list(tmp_path.glob("*.wav"))
    assert temps == [audio]