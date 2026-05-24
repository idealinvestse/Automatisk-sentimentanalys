"""Tests for sentiment module (unit tests without model loading)."""
from __future__ import annotations

from src.sentiment import (
    DEFAULT_MODEL,
    _device_arg_from_key,
    _normalize_device_spec,
    normalize_label,
)


class TestNormalizeLabel:
    def test_label_0(self):
        assert normalize_label("label_0") == "negativ"
        assert normalize_label("LABEL_0") == "negativ"

    def test_label_1(self):
        assert normalize_label("label_1") == "neutral"

    def test_label_2(self):
        assert normalize_label("label_2") == "positiv"

    def test_english_names(self):
        assert normalize_label("negative") == "negativ"
        assert normalize_label("neg") == "negativ"
        assert normalize_label("neutral") == "neutral"
        assert normalize_label("positive") == "positiv"
        assert normalize_label("pos") == "positiv"

    def test_swedish_names(self):
        assert normalize_label("negativ") == "negativ"
        assert normalize_label("neutral") == "neutral"
        assert normalize_label("positiv") == "positiv"

    def test_unknown_passthrough(self):
        assert normalize_label("unknown") == "unknown"


class TestDeviceNormalization:
    def test_auto(self):
        device_arg, key = _normalize_device_spec("auto")
        assert isinstance(key, str)
        assert key in ("cpu", "cuda:0", "mps")

    def test_cpu(self):
        device_arg, key = _normalize_device_spec("cpu")
        assert device_arg == -1
        assert key == "cpu"

    def test_cuda_explicit(self):
        device_arg, key = _normalize_device_spec("cuda:0")
        # May return cpu if CUDA not available
        assert key in ("cuda:0", "cpu")

    def test_none_means_auto(self):
        device_arg, key = _normalize_device_spec(None)
        assert isinstance(key, str)

    def test_int_device(self):
        device_arg, key = _normalize_device_spec(-1)
        assert key == "cpu"


class TestDeviceArgFromKey:
    def test_cpu(self):
        assert _device_arg_from_key("cpu") == -1

    def test_cuda(self):
        result = _device_arg_from_key("cuda:0")
        assert result == 0

    def test_unknown(self):
        assert _device_arg_from_key("unknown") == -1


class TestDefaults:
    def test_default_model_is_set(self):
        assert DEFAULT_MODEL == "cardiffnlp/twitter-xlm-roberta-base-sentiment"
