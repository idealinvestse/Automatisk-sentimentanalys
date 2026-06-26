"""Tests for Transcription v2 call-center preprocess + VAD (Task A-1)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.transcription.factory import resolve_preprocess_mode
from src.transcription.preprocess import (
    normalize_preprocess_mode,
    maybe_preprocess_for_mode,
    prepare_asr_audio,
)
from src.transcription.preprocess_v2 import preprocess_audio_callcenter
from src.transcription.vad_callcenter import (
    CALLCENTER_VAD_PARAMETERS,
    get_callcenter_vad_parameters,
    vad_options_for_mode,
)


class TestNormalizePreprocessMode:
    def test_legacy_boolean_basic(self):
        assert normalize_preprocess_mode(preprocess=True) == "basic"
        assert normalize_preprocess_mode(preprocess=False) == "off"

    def test_explicit_modes(self):
        assert normalize_preprocess_mode(preprocess_mode="callcenter") == "callcenter"
        assert normalize_preprocess_mode(preprocess_mode="basic") == "basic"
        assert normalize_preprocess_mode(preprocess_mode="off") == "off"

    def test_aliases(self):
        assert normalize_preprocess_mode(preprocess_mode="v2") == "callcenter"
        assert normalize_preprocess_mode(preprocess_mode="cc") == "callcenter"
        assert normalize_preprocess_mode(preprocess_mode="legacy") == "basic"

    def test_unknown_mode_defaults_off(self):
        assert normalize_preprocess_mode(preprocess_mode="unknown") == "off"


class TestResolvePreprocessMode:
    def test_callcenter_profile_upgrades_basic_preprocess(self):
        assert (
            resolve_preprocess_mode(preprocess=True, profile="callcenter")
            == "callcenter"
        )

    def test_explicit_mode_overrides_profile(self):
        assert (
            resolve_preprocess_mode(
                preprocess=True,
                preprocess_mode="basic",
                profile="callcenter",
            )
            == "basic"
        )


class TestCallcenterVad:
    def test_parameters_are_copy(self):
        params = get_callcenter_vad_parameters()
        params["threshold"] = 0.0
        assert CALLCENTER_VAD_PARAMETERS["threshold"] == 0.35

    def test_vad_options_only_for_callcenter(self):
        assert vad_options_for_mode("callcenter") is not None
        assert vad_options_for_mode("basic") is None
        assert vad_options_for_mode("off") is None
        assert vad_options_for_mode("callcenter", vad_enabled=False) is None


@patch("src.transcription.preprocess_v2.subprocess.run")
def test_preprocess_audio_callcenter_ffmpeg_chain(mock_run, tmp_path):
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

    handle = preprocess_audio_callcenter(str(audio), noise_reduction=False)
    cmd = mock_run.call_args.args[0]
    af_index = cmd.index("-af") + 1
    assert "lowpass=f=3400" in cmd[af_index]
    assert "dynaudnorm" in cmd[af_index]
    assert handle.path in created

    handle.cleanup()
    for path in created:
        assert not os.path.exists(path)


def test_maybe_preprocess_for_mode_callcenter_delegates(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")

    with patch(
        "src.transcription.preprocess_v2.preprocess_audio_callcenter"
    ) as mock_cc:
        mock_cc.return_value = MagicMock(path=str(audio), cleanup=MagicMock())
        handle = maybe_preprocess_for_mode(str(audio), "callcenter")
        mock_cc.assert_called_once_with(str(audio))
        assert handle.path == str(audio)


def test_prepare_asr_audio_falls_back_on_failure(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")

    with patch(
        "src.transcription.preprocess.maybe_preprocess_for_mode",
        side_effect=RuntimeError("boom"),
    ):
        handle, mode = prepare_asr_audio(str(audio), preprocess_mode="callcenter")

    assert mode == "off"
    assert handle.path == str(audio)