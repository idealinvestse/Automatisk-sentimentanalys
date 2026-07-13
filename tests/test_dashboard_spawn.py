"""Tests for launcher.dashboard_spawn (Next.js webui process entry)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from launcher import dashboard_spawn


def test_resolve_dashboard_ui_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHBOARD_UI", raising=False)
    assert dashboard_spawn.resolve_dashboard_ui() == "webui"


def test_dev_mode_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_DEV_MODE", "1")
    assert dashboard_spawn._dev_mode() is True
    monkeypatch.setenv("SENTIMENT_DEV_MODE", "0")
    assert dashboard_spawn._dev_mode() is False


def test_main_dev_mode_runs_npm_dev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_DEV_MODE", "1")
    monkeypatch.setenv("PORT", "3456")
    monkeypatch.delenv("DASHBOARD_UI", raising=False)

    with (
        patch.object(dashboard_spawn, "_repo_root", return_value=tmp_path),
        patch.object(dashboard_spawn.shutil, "which", return_value="npm"),
        patch.object(dashboard_spawn.subprocess, "call", return_value=0) as mock_call,
        pytest.raises(SystemExit) as exc,
    ):
        dashboard_spawn.main()

    assert exc.value.code == 0
    assert mock_call.call_args.args[0][:3] == ["npm", "run", "dev"]


def test_main_prod_builds_when_next_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("SENTIMENT_DEV_MODE", raising=False)
    monkeypatch.setenv("PORT", "3000")

    calls: list[list[str]] = []

    def fake_call(cmd: list[str], **kwargs: object) -> int:
        calls.append(list(cmd))
        return 0

    with (
        patch.object(dashboard_spawn, "_repo_root", return_value=tmp_path),
        patch.object(dashboard_spawn.shutil, "which", return_value="npm"),
        patch.object(dashboard_spawn.subprocess, "call", side_effect=fake_call),
        pytest.raises(SystemExit) as exc,
    ):
        dashboard_spawn.main()

    assert exc.value.code == 0
    assert calls[0][:3] == ["npm", "run", "build"]
    assert calls[1][:3] == ["npm", "run", "start"]
