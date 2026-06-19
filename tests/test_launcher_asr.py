"""Tests for launcher ASR management helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from launcher.asr_manager import asr_status_for_config, format_asr_report_lines, run_asr_setup
from src.install.asr_assets import AsrAssetReport
from src.install.config_schema import UserConfig


def test_format_asr_report_lines():
    report = AsrAssetReport()
    report.add("model_faster", True, "ok", "detail")
    lines = format_asr_report_lines(report)
    assert len(lines) == 1
    assert lines[0][0] is True
    assert "detail" in lines[0][1]


@patch("launcher.asr_manager.ensure_asr_assets")
@patch("launcher.asr_manager.resolve_python")
def test_run_asr_setup_uses_config(mock_py, mock_ensure):
    mock_py.return_value = MagicMock()
    mock_ensure.return_value = AsrAssetReport()
    cfg = UserConfig()
    cfg.paths.app_root = "."
    cfg.asr.model = "kb-whisper-large"
    cfg.asr.language = "sv"
    cfg.asr.revision = "strict"

    run_asr_setup(cfg, install_packages=True, download_models=False)

    mock_ensure.assert_called_once()
    kwargs = mock_ensure.call_args.kwargs
    assert kwargs["model"] == "kb-whisper-large"
    assert kwargs["language"] == "sv"
    assert kwargs["revision"] == "strict"
    assert kwargs["install_packages"] is True
    assert kwargs["download_models"] is False


@patch("launcher.asr_manager.collect_asr_status")
def test_asr_status_for_config(mock_collect):
    from src.install.asr_assets import AsrStatus

    mock_collect.return_value = AsrStatus(
        faster_whisper_installed=True,
        whisperx_installed=False,
        huggingface_hub_installed=True,
        model_name="KBLab/kb-whisper-large",
        hf_cache_dir="/tmp/hf",
        kb_model_cached=False,
    )
    status = asr_status_for_config(UserConfig())
    assert status.whisperx_installed is False