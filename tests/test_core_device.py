"""Tests for src.core.device normalization helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.core.device import device_arg_from_key, normalize_device_for_asr, normalize_device_spec


class TestNormalizeDeviceSpec:
    def test_cpu_string(self) -> None:
        device_arg, key = normalize_device_spec("cpu")
        assert device_arg == -1
        assert key == "cpu"

    def test_int_negative_returns_cpu(self) -> None:
        device_arg, key = normalize_device_spec(-1)
        assert device_arg == -1
        assert key == "cpu"

    def test_torch_device_cpu_fallback(self) -> None:
        device_arg, key = normalize_device_spec(torch.device("cpu"))
        assert device_arg == -1
        assert key == "cpu"

    def test_cuda_string_without_gpu(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device_arg, key = normalize_device_spec("cuda:0")
        assert device_arg == -1
        assert key == "cpu"

    def test_cuda_string_invalid_index(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        device_arg, key = normalize_device_spec("cuda:not-a-number")
        assert key in ("cuda:0", "cpu")

    def test_int_cuda_with_gpu(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        device_arg, key = normalize_device_spec(1)
        assert device_arg == 1
        assert key == "cuda:1"

    def test_torch_cuda_device(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        dev = torch.device("cuda:1")
        device_arg, key = normalize_device_spec(dev)
        assert device_arg == 1
        assert key == "cuda:1"

    def test_mps_string_when_available(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        mps = MagicMock()
        mps.is_available.return_value = True
        monkeypatch.setattr(torch.backends, "mps", mps)
        device_arg, key = normalize_device_spec("mps")
        assert key == "mps"
        assert isinstance(device_arg, torch.device)

    def test_unknown_type_returns_cpu(self) -> None:
        device_arg, key = normalize_device_spec(object())  # type: ignore[arg-type]
        assert device_arg == -1
        assert key == "cpu"


class TestDeviceArgFromKey:
    def test_mps(self) -> None:
        assert device_arg_from_key("mps") == torch.device("mps")

    def test_cuda_invalid_index(self) -> None:
        assert device_arg_from_key("cuda:bad") == 0


class TestNormalizeDeviceForAsr:
    def test_cuda_invalid_index_falls_back(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        device, idx = normalize_device_for_asr("cuda:9")
        assert device == "cuda"
        assert idx == 0

    def test_cuda_unavailable_returns_cpu(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device, idx = normalize_device_for_asr("cuda:0")
        assert device == "cpu"
        assert idx is None

    def test_mps_string(self) -> None:
        device, idx = normalize_device_for_asr("mps")
        assert device == "mps"
        assert idx is None
