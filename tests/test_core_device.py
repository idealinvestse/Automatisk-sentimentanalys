"""Tests for src.core.device normalization helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.core.device import device_arg_from_key, normalize_device_for_asr, normalize_device_spec


class TestNormalizeDeviceSpec:
    def test_mps_auto_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.core.device.torch.cuda.is_available", lambda: False)
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        monkeypatch.setattr("src.core.device.torch.backends.mps", mock_mps)
        device_arg, key = normalize_device_spec("auto")
        assert key == "mps"
        assert isinstance(device_arg, torch.device)

    def test_cuda_index_out_of_range_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.core.device.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("src.core.device.torch.cuda.device_count", lambda: 1)
        device_arg, key = normalize_device_spec(5)
        assert device_arg == 0
        assert key == "cuda:0"

    def test_torch_device_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.core.device.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("src.core.device.torch.cuda.device_count", lambda: 2)
        device_arg, key = normalize_device_spec(torch.device("cuda:1"))
        assert device_arg == 1
        assert key == "cuda:1"

    def test_string_cuda_invalid_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.core.device.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("src.core.device.torch.cuda.device_count", lambda: 1)
        device_arg, key = normalize_device_spec("cuda:not-a-number")
        assert device_arg == 0
        assert key == "cuda:0"

    def test_unknown_type_falls_back_to_cpu(self) -> None:
        device_arg, key = normalize_device_spec(object())  # type: ignore[arg-type]
        assert device_arg == -1
        assert key == "cpu"


class TestDeviceArgFromKey:
    def test_mps_key(self) -> None:
        assert device_arg_from_key("mps") == torch.device("mps")

    def test_cuda_invalid_index(self) -> None:
        assert device_arg_from_key("cuda:bad") == 0


class TestNormalizeDeviceForAsr:
    def test_mps_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.core.device.torch.cuda.is_available", lambda: False)
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        monkeypatch.setattr("src.core.device.torch.backends.mps", mock_mps)
        device, idx = normalize_device_for_asr("auto")
        assert device == "mps"
        assert idx is None

    def test_mps_explicit(self) -> None:
        device, idx = normalize_device_for_asr("mps")
        assert device == "mps"
        assert idx is None

    def test_unknown_device_string(self) -> None:
        device, idx = normalize_device_for_asr("tpu")
        assert device == "cpu"
        assert idx is None
