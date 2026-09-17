"""Contract tests for customer registry and filename-based customer identity.

Customer identity resolved from the original audio filename is *routing
context* — it selects organization configuration, never authentication or
authorization (see docs/PILOT_RUNBOOK.md, decision R03).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from src.api.call_store import CallStore
from src.customers import (
    AmbiguousCustomerIdError,
    CustomerPolicyError,
    CustomerProfile,
    CustomerRegistry,
    DisabledCustomerError,
    InvalidCustomerRegistryError,
    MissingCustomerIdError,
    UnknownCustomerError,
    clamp_execution_policy,
    extract_customer_ids,
    get_customer_registry,
    normalize_customer_id,
    resolve_customer,
)

PATTERN = r"kund-(?P<customer_id>[0-9a-z]+)"


def _profile(**overrides) -> CustomerProfile:
    base = {
        "display_name": "Testkund AB",
        "analyzer_profile": "callcenter",
        "qa_scorecard": "standard_support_v1",
        "asr": {"provider": "local", "allow_cloud_fallback": False},
        "llm": {
            "enabled": True,
            "provider_allowlist": ["lmstudio"],
            "anonymize_before_llm": True,
        },
    }
    base.update(overrides)
    return CustomerProfile(**base)


def _registry(**overrides) -> CustomerRegistry:
    base = {
        "mode": "required",
        "id_patterns": [PATTERN],
        "customers": {"0042": _profile()},
    }
    base.update(overrides)
    return CustomerRegistry(**base)


class TestNormalizeCustomerId:
    def test_strip_and_lowercase(self) -> None:
        assert normalize_customer_id("  ABC-123 ") == "abc-123"

    def test_empty(self) -> None:
        assert normalize_customer_id("") == ""


class TestCustomerErrorCodes:
    def test_stable_codes(self) -> None:
        assert MissingCustomerIdError("x.wav").error_code == "missing_customer_id"
        assert AmbiguousCustomerIdError("x.wav", ["a", "b"]).error_code == "ambiguous_customer_id"
        assert UnknownCustomerError("9999").error_code == "unknown_customer"
        assert DisabledCustomerError("0042").error_code == "disabled_customer"
        assert InvalidCustomerRegistryError("bad").error_code == "invalid_customer_registry"
        assert CustomerPolicyError("widen").error_code == "customer_policy_violation"


class TestExtractCustomerIds:
    def test_single_match_normalized(self) -> None:
        assert extract_customer_ids("kund-0042_samtal.wav", [PATTERN]) == ["0042"]

    def test_case_insensitive_match(self) -> None:
        assert extract_customer_ids("KUND-0042_samtal.WAV", [PATTERN]) == ["0042"]

    def test_uses_basename_only(self) -> None:
        # Customer ID in a directory component must not leak into identity.
        ids = extract_customer_ids("kund-9999/inspelning.wav", [PATTERN])
        assert ids == []

    def test_multiple_patterns_union(self) -> None:
        patterns = [PATTERN, r"org-(?P<customer_id>[0-9a-z]+)"]
        assert extract_customer_ids("org-abc1_call.wav", patterns) == ["abc1"]

    def test_distinct_ids_preserved_in_order(self) -> None:
        ids = extract_customer_ids("kund-0042_kund-0077.wav", [PATTERN])
        assert ids == ["0042", "0077"]

    def test_same_id_repeated_is_single(self) -> None:
        ids = extract_customer_ids("kund-0042_kund-0042.wav", [PATTERN])
        assert ids == ["0042"]

    def test_no_match_returns_empty(self) -> None:
        assert extract_customer_ids("random_audio.wav", [PATTERN]) == []

    def test_invalid_pattern_raises(self) -> None:
        with pytest.raises(InvalidCustomerRegistryError):
            extract_customer_ids("x.wav", ["kund-(?P<broken>"])


class TestRegistryValidation:
    def test_required_mode_requires_patterns(self) -> None:
        with pytest.raises(ValidationError):
            CustomerRegistry(mode="required", id_patterns=[], customers={})

    def test_optional_mode_requires_patterns(self) -> None:
        with pytest.raises(ValidationError):
            CustomerRegistry(mode="optional", id_patterns=[], customers={})

    def test_disabled_mode_allows_empty(self) -> None:
        reg = CustomerRegistry()
        assert reg.mode == "disabled"
        assert reg.id_patterns == []

    def test_pattern_must_have_named_group(self) -> None:
        with pytest.raises(ValidationError):
            CustomerRegistry(
                mode="required",
                id_patterns=[r"kund-\d+"],
                customers={},
            )

    def test_pattern_must_compile(self) -> None:
        with pytest.raises(ValidationError):
            CustomerRegistry(
                mode="required",
                id_patterns=["kund-(?P<customer_id>["],
                customers={},
            )

    def test_analyzer_profile_must_exist(self) -> None:
        with pytest.raises(ValidationError):
            _profile(analyzer_profile="nonexistent_profile")

    def test_customer_keys_normalized(self) -> None:
        reg = CustomerRegistry(
            mode="required",
            id_patterns=[PATTERN],
            customers={"CUST-42": _profile()},
        )
        assert "cust-42" in reg.customers

    def test_normalized_key_collision_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CustomerRegistry(
                mode="required",
                id_patterns=[PATTERN],
                customers={"cust-42": _profile(), " CUST-42 ": _profile()},
            )

    def test_alias_resolves(self) -> None:
        reg = CustomerRegistry(
            mode="required",
            id_patterns=[PATTERN],
            customers={"0042": _profile(aliases=["testbolaget"])},
        )
        ctx = resolve_customer("kund-testbolaget_samtal.wav", reg)
        assert ctx is not None
        assert ctx.customer_id == "0042"

    def test_alias_collision_with_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CustomerRegistry(
                mode="required",
                id_patterns=[PATTERN],
                customers={
                    "0042": _profile(aliases=["0077"]),
                    "0077": _profile(),
                },
            )

    def test_duplicate_alias_across_customers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CustomerRegistry(
                mode="required",
                id_patterns=[PATTERN],
                customers={
                    "0042": _profile(aliases=["acme"]),
                    "0077": _profile(aliases=["ACME"]),
                },
            )

    def test_enabled_llm_requires_allowlist(self) -> None:
        with pytest.raises(ValidationError):
            _profile(llm={"enabled": True, "provider_allowlist": []})

    def test_disabled_llm_allows_empty_allowlist(self) -> None:
        profile = _profile(llm={"enabled": False, "provider_allowlist": []})
        assert profile.llm.enabled is False


class TestResolveCustomer:
    def test_disabled_mode_returns_none(self) -> None:
        reg = CustomerRegistry()  # disabled
        assert resolve_customer("kund-0042_samtal.wav", reg) is None
        assert resolve_customer("kund-unknown_x.wav", reg) is None

    def test_optional_missing_id_returns_none(self) -> None:
        reg = _registry(mode="optional")
        assert resolve_customer("no_identifier.wav", reg) is None

    def test_required_missing_id_raises(self) -> None:
        with pytest.raises(MissingCustomerIdError):
            resolve_customer("no_identifier.wav", _registry())

    def test_required_none_filename_raises(self) -> None:
        with pytest.raises(MissingCustomerIdError):
            resolve_customer(None, _registry())

    def test_unknown_id_raises_required(self) -> None:
        with pytest.raises(UnknownCustomerError) as exc:
            resolve_customer("kund-9999_samtal.wav", _registry())
        assert "9999" in str(exc.value)

    def test_unknown_id_raises_optional(self) -> None:
        # Controlled mode never falls back to a generic default.
        with pytest.raises(UnknownCustomerError):
            resolve_customer("kund-9999_samtal.wav", _registry(mode="optional"))

    def test_ambiguous_id_raises(self) -> None:
        with pytest.raises(AmbiguousCustomerIdError) as exc:
            resolve_customer("kund-0042_kund-0077.wav", _registry())
        assert exc.value.candidates == ["0042", "0077"]

    def test_disabled_customer_raises(self) -> None:
        reg = _registry(customers={"0042": _profile(enabled=False)})
        with pytest.raises(DisabledCustomerError):
            resolve_customer("kund-0042_samtal.wav", reg)

    def test_context_fields(self) -> None:
        ctx = resolve_customer("kund-0042_samtal.wav", _registry(version=3))
        assert ctx is not None
        assert ctx.customer_id == "0042"
        assert ctx.display_name == "Testkund AB"
        assert ctx.analyzer_profile == "callcenter"
        assert ctx.qa_scorecard == "standard_support_v1"
        assert ctx.asr.provider == "local"
        assert ctx.asr.allow_cloud_fallback is False
        assert ctx.llm.provider_allowlist == ["lmstudio"]
        assert ctx.registry_version == 3
        assert ctx.config_fingerprint
        assert ctx.source_filename == "kund-0042_samtal.wav"

    def test_context_uses_canonical_key_not_alias(self) -> None:
        reg = _registry(customers={"0042": _profile(aliases=["acme"])})
        ctx = resolve_customer("kund-acme_call.wav", reg)
        assert ctx is not None
        assert ctx.customer_id == "0042"

    def test_context_is_frozen(self) -> None:
        ctx = resolve_customer("kund-0042_samtal.wav", _registry())
        assert ctx is not None
        with pytest.raises(ValidationError):
            ctx.customer_id = "0077"

    def test_fingerprint_stable_and_config_sensitive(self) -> None:
        reg = _registry()
        a = resolve_customer("kund-0042_a.wav", reg)
        b = resolve_customer("kund-0042_b.wav", reg)
        assert a is not None and b is not None
        assert a.config_fingerprint == b.config_fingerprint

        reg2 = _registry(customers={"0042": _profile(analyzer_profile="support")})
        c = resolve_customer("kund-0042_a.wav", reg2)
        assert c is not None
        assert c.config_fingerprint != a.config_fingerprint


class TestUserConfigIntegration:
    def test_user_config_default_customers_disabled(self) -> None:
        from src.install.config_schema import UserConfig

        cfg = UserConfig()
        assert cfg.customers.mode == "disabled"

    def test_merge_configs_parses_customers(self) -> None:
        from src.install.user_config import merge_configs

        cfg = merge_configs(
            {},
            {
                "customers": {
                    "mode": "required",
                    "id_patterns": [PATTERN],
                    "customers": {"0042": {"display_name": "Testkund AB"}},
                }
            },
        )
        assert cfg.customers.mode == "required"
        assert "0042" in cfg.customers.customers

    def test_invalid_customer_config_rejected_at_load(self) -> None:
        from src.install.user_config import merge_configs

        with pytest.raises(ValidationError):
            merge_configs(
                {},
                {"customers": {"mode": "required", "id_patterns": [], "customers": {}}},
            )


# ---------------------------------------------------------------------------
# API wiring: /upload and /transcribe resolve customer from original filename
# ---------------------------------------------------------------------------

_AUDIO = b"RIFF" + (56).to_bytes(4, "little") + b"WAVE" + b"\x00" * 48


def _write_user_config(tmp_path: Path, registry: dict) -> Path:
    cfg_path = tmp_path / "user_config.yaml"
    cfg_path.write_text(yaml.safe_dump({"customers": registry}), encoding="utf-8")
    return cfg_path


@pytest.fixture
def customer_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the API at a temp user config with a required-mode registry."""

    def _setup(registry: dict):
        cfg_path = _write_user_config(tmp_path, registry)
        monkeypatch.setenv("SENTIMENT_USER_CONFIG", str(cfg_path))
        monkeypatch.setenv("API_MEDIA_ROOT", str(tmp_path / "media"))
        monkeypatch.setenv("API_STATE_DIR", str(tmp_path / "state"))
        (tmp_path / "media").mkdir(exist_ok=True)
        from src.api.settings import get_api_settings

        get_api_settings.cache_clear()
        get_customer_registry.cache_clear()

    yield _setup
    get_customer_registry.cache_clear()
    from src.api.settings import get_api_settings

    get_api_settings.cache_clear()


class TestUploadCustomerResolution:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        return TestClient(create_app())

    def test_upload_returns_customer_ref(self, customer_env) -> None:
        customer_env(
            {
                "mode": "required",
                "id_patterns": [PATTERN],
                "customers": {"0042": {"display_name": "Testkund AB"}},
            }
        )
        client = self._client()
        with patch(
            "src.api.routers.transcription.validate_audio_path",
            side_effect=lambda p: p,
        ):
            r = client.post(
                "/upload",
                files={"file": ("kund-0042_samtal.wav", io.BytesIO(_AUDIO), "audio/wav")},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["customer"]["customer_id"] == "0042"
        assert body["customer"]["display_name"] == "Testkund AB"

    def test_upload_unknown_customer_rejected_and_not_saved(
        self, customer_env, tmp_path: Path
    ) -> None:
        customer_env(
            {
                "mode": "required",
                "id_patterns": [PATTERN],
                "customers": {"0042": {"display_name": "Testkund AB"}},
            }
        )
        client = self._client()
        r = client.post(
            "/upload",
            files={"file": ("kund-9999_samtal.wav", io.BytesIO(_AUDIO), "audio/wav")},
        )
        assert r.status_code == 422
        assert r.json()["error_code"] == "unknown_customer"
        uploads = tmp_path / "media" / "uploads"
        leftover = list(uploads.glob("*")) if uploads.is_dir() else []
        assert leftover == []

    def test_upload_missing_id_rejected_in_required_mode(self, customer_env) -> None:
        customer_env(
            {
                "mode": "required",
                "id_patterns": [PATTERN],
                "customers": {"0042": {"display_name": "Testkund AB"}},
            }
        )
        client = self._client()
        r = client.post(
            "/upload",
            files={"file": ("samtal.wav", io.BytesIO(_AUDIO), "audio/wav")},
        )
        assert r.status_code == 422
        assert r.json()["error_code"] == "missing_customer_id"

    def test_upload_disabled_mode_unchanged(self, customer_env, tmp_path: Path) -> None:
        customer_env({"mode": "disabled", "id_patterns": [], "customers": {}})
        client = self._client()
        with patch(
            "src.api.routers.transcription.validate_audio_path",
            side_effect=lambda p: p,
        ):
            r = client.post(
                "/upload",
                files={"file": ("samtal.wav", io.BytesIO(_AUDIO), "audio/wav")},
            )
        assert r.status_code == 200, r.text
        assert r.json()["customer"] is None


class TestTranscribeCustomerResolution:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        return TestClient(create_app())

    def _registry_env(self, customer_env) -> None:
        customer_env(
            {
                "mode": "required",
                "id_patterns": [PATTERN],
                "customers": {"0042": {"display_name": "Testkund AB"}},
            }
        )

    def test_transcribe_uses_original_filename(
        self, customer_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._registry_env(customer_env)
        media_root = tmp_path / "media"
        audio = media_root / "uploads" / "ab12cd34ef56_samtal.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(_AUDIO)

        captured: dict = {}

        def _fake_helper(**kwargs):
            captured.update(kwargs)
            return {"text": "hej", "segments": [{"text": "hej"}]}

        client = self._client()
        with patch(
            "src.api.routers.transcription.transcribe_helper",
            side_effect=_fake_helper,
        ):
            r = client.post(
                "/transcribe",
                json={
                    "audio_path": str(audio),
                    "original_filename": "kund-0042_samtal.wav",
                },
            )
        assert r.status_code == 200, r.text
        assert captured["audio_path"] == str(audio)

    def test_transcribe_unknown_customer_rejected(self, customer_env, tmp_path: Path) -> None:
        self._registry_env(customer_env)
        audio = tmp_path / "media" / "uploads" / "ab12cd34ef56_samtal.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(_AUDIO)

        client = self._client()
        with patch(
            "src.api.routers.transcription.transcribe_helper",
            side_effect=AssertionError("transcribe_helper must not run"),
        ):
            r = client.post(
                "/transcribe",
                json={
                    "audio_path": str(audio),
                    "original_filename": "kund-9999_samtal.wav",
                },
            )
        assert r.status_code == 422
        assert r.json()["error_code"] == "unknown_customer"

    def test_transcribe_job_meta_carries_customer(self, customer_env, tmp_path: Path) -> None:
        self._registry_env(customer_env)
        audio = tmp_path / "media" / "uploads" / "ab12cd34ef56_samtal.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(_AUDIO)

        client = self._client()
        with patch(
            "src.api.routers.transcription.transcribe_helper",
            return_value={"text": "hej", "segments": [{"text": "hej"}]},
        ):
            r = client.post(
                "/transcribe",
                headers={"X-Transcription-Job-Id": "job-cust-1"},
                json={
                    "audio_path": str(audio),
                    "original_filename": "kund-0042_samtal.wav",
                },
            )
        assert r.status_code == 200, r.text
        job = client.get("/transcription/jobs/job-cust-1")
        assert job.status_code == 200
        meta = job.json()["meta"]
        assert meta["customer"]["customer_id"] == "0042"
        assert meta["customer"]["registry_version"] >= 1


class TestOtherIntakePathsCustomerGate:
    """All audio intake paths must honor the registry — required mode must not
    be bypassable via batch, analyze or scan endpoints."""

    def _client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        return TestClient(create_app())

    def _registry_env(self, customer_env) -> None:
        customer_env(
            {
                "mode": "required",
                "id_patterns": [PATTERN],
                "customers": {"0042": {"display_name": "Testkund AB"}},
            }
        )

    def _write_audio(self, media_root: Path, name: str) -> Path:
        path = media_root / "uploads" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_AUDIO)
        return path

    def test_batch_transcribe_unknown_file_fails_per_item(
        self, customer_env, tmp_path: Path
    ) -> None:
        self._registry_env(customer_env)
        media_root = tmp_path / "media"
        good = self._write_audio(media_root, "kund-0042_a.wav")
        bad = self._write_audio(media_root, "kund-9999_b.wav")

        client = self._client()
        with patch(
            "src.api.routers.transcription.transcribe_helper",
            return_value={"text": "hej", "segments": [{"text": "hej"}]},
        ):
            r = client.post(
                "/batch_transcribe",
                json={"audio_paths": [str(good), str(bad)]},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] == 1
        assert body["failed"] == 1
        errors = {i["file"]: i["error"] for i in body["items"] if i["error"]}
        assert str(bad) in errors
        assert "9999" in errors[str(bad)]

    def test_analyze_conversation_unknown_rejected(self, customer_env, tmp_path: Path) -> None:
        self._registry_env(customer_env)
        audio = self._write_audio(tmp_path / "media", "kund-9999_x.wav")

        client = self._client()
        with patch(
            "src.api.routers.conversation.run_analyze_conversation",
            side_effect=AssertionError("must not run for unknown customer"),
        ):
            r = client.post(
                "/analyze_conversation",
                json={
                    "audio_path": str(audio),
                    "original_filename": "kund-9999_x.wav",
                },
            )
        assert r.status_code == 422

    def test_analyze_conversation_known_injects_customer_meta(
        self, customer_env, tmp_path: Path
    ) -> None:
        self._registry_env(customer_env)
        audio = self._write_audio(tmp_path / "media", "kund-0042_x.wav")

        from src.api.schemas import AnalyzeConversationResponse

        fake = AnalyzeConversationResponse(
            transcript={"segments": []},
            segment_sentiments=[],
            meta={},
            timestamp="2026-01-01T00:00:00Z",
        )
        client = self._client()
        with patch(
            "src.api.routers.conversation.run_analyze_conversation",
            return_value=fake,
        ):
            r = client.post(
                "/analyze_conversation",
                json={
                    "audio_path": str(audio),
                    "original_filename": "kund-0042_x.wav",
                },
            )
        assert r.status_code == 200, r.text
        assert r.json()["meta"]["customer"]["customer_id"] == "0042"

    def test_scan_process_rejects_unknown_files(self, customer_env, tmp_path: Path) -> None:
        self._registry_env(customer_env)
        incoming = tmp_path / "media" / "incoming"
        incoming.mkdir(parents=True)
        (incoming / "kund-0042_a.wav").write_bytes(_AUDIO)
        (incoming / "kund-9999_b.wav").write_bytes(_AUDIO)

        client = self._client()
        with patch(
            "src.api.routers.scan.transcribe_helper",
            return_value={"text": "hej", "segments": [{"text": "hej"}]},
        ):
            r = client.post(
                "/scan_process",
                json={
                    "directory": str(incoming),
                    "pattern": "*.wav",
                    "operation": "transcribe",
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] == 1
        assert body["failed"] == 1
        failed_names = [Path(i["file"]).name for i in body["items"] if not i["ok"]]
        assert failed_names == ["kund-9999_b.wav"]


class TestCustomerExecutionPolicy:
    def test_no_customer_keeps_request(self) -> None:
        policy = clamp_execution_policy(
            None,
            requested_asr_provider="local",
            requested_llm_enabled=True,
            requested_llm_provider="openrouter",
            requested_profile="sales",
        )
        assert policy.analyzer_profile == "sales"
        assert policy.llm_enabled is True
        assert policy.llm_provider == "openrouter"

    def test_customer_profile_and_scorecard_win(self) -> None:
        ctx = resolve_customer("kund-0042_x.wav", _registry())
        policy = clamp_execution_policy(ctx, requested_profile="sales")
        assert policy.analyzer_profile == "callcenter"
        assert policy.qa_scorecard == "standard_support_v1"
        assert policy.customer_id == "0042"

    def test_request_cannot_widen_asr_to_cloud(self) -> None:
        ctx = resolve_customer("kund-0042_x.wav", _registry())
        with pytest.raises(CustomerPolicyError, match="cloud"):
            clamp_execution_policy(ctx, requested_asr_provider="cloud")

    def test_request_cannot_enable_disallowed_llm(self) -> None:
        ctx = resolve_customer("kund-0042_x.wav", _registry())
        with pytest.raises(CustomerPolicyError, match="openrouter"):
            clamp_execution_policy(
                ctx,
                requested_llm_enabled=True,
                requested_llm_provider="openrouter",
            )

    def test_request_may_use_allowlisted_llm(self) -> None:
        ctx = resolve_customer("kund-0042_x.wav", _registry())
        policy = clamp_execution_policy(
            ctx,
            requested_llm_enabled=True,
            requested_llm_provider="lmstudio",
        )
        assert policy.llm_enabled is True
        assert policy.llm_provider == "lmstudio"


class TestFailClosedAndPipelineWiring:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        return TestClient(create_app())

    def test_transcribe_empty_speech_is_422(
        self, customer_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        customer_env({"mode": "disabled", "id_patterns": [], "customers": {}})
        audio = tmp_path / "media" / "uploads" / "silent.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(_AUDIO)
        monkeypatch.setenv("API_STATE_DIR", str(tmp_path / "state"))
        from src.api.settings import get_api_settings

        get_api_settings.cache_clear()
        client = self._client()
        with patch(
            "src.api.routers.transcription.transcribe_helper",
            return_value={"text": "", "segments": []},
        ):
            r = client.post("/transcribe", json={"audio_path": str(audio)})
        assert r.status_code == 422
        assert r.json()["error_code"] == "asr_empty_transcript"

    def test_analyze_pipeline_uses_customer_profile_and_rejects_llm_widen(
        self, customer_env, tmp_path: Path
    ) -> None:
        customer_env(
            {
                "mode": "required",
                "id_patterns": [PATTERN],
                "customers": {
                    "0042": {
                        "display_name": "Testkund AB",
                        "analyzer_profile": "callcenter",
                        "qa_scorecard": "standard_support_v1",
                        "llm": {"enabled": True, "provider_allowlist": ["lmstudio"]},
                    }
                },
            }
        )
        client = self._client()
        widen = client.post(
            "/analyze_pipeline",
            json={
                "segments": [{"text": "hej", "start": 0, "end": 1}],
                "original_filename": "kund-0042_samtal.wav",
                "use_mistral_llm": True,
                "provider": "groq",
            },
        )
        assert widen.status_code == 422
        assert widen.json()["error_code"] == "customer_policy_violation"

        missing = client.post(
            "/analyze_pipeline",
            json={"segments": [{"text": "hej", "start": 0, "end": 1}]},
        )
        assert missing.status_code == 422
        assert missing.json()["error_code"] == "missing_customer_id"

        unknown_fas4 = client.post(
            "/agent_performance/Agent-1",
            json={
                "segments_list": [[{"text": "hej"}]],
                "agent_id": "Agent-1",
                "original_filename": "kund-9999_samtal.wav",
            },
        )
        assert unknown_fas4.status_code == 422
        assert unknown_fas4.json()["error_code"] == "unknown_customer"

        unknown_compare = client.post(
            "/analyze_pipeline/compare",
            json={
                "segments": [{"text": "hej"}],
                "models": ["mistralai/mistral-small-3.1-24b-instruct"],
                "original_filename": "kund-9999_samtal.wav",
            },
        )
        assert unknown_compare.status_code == 422
        assert unknown_compare.json()["error_code"] == "unknown_customer"

        captured: dict = {}

        def _fake_analyze(self, segments, selected=None):
            captured["profile"] = self.profile
            captured["qa_scorecard"] = self.qa_scorecard
            from src.core.models import CallAnalysisReport

            return CallAnalysisReport(
                segments=segments,
                sentiment_results=[],
                intent_results=[],
                summary={},
                topics={},
                insights={},
                risks={},
                processing_time_s=0.0,
                results={"qa": {"scorecard_name": self.qa_scorecard}},
                llm={},
            )

        with patch("src.pipeline.CallAnalysisPipeline.analyze_segments", _fake_analyze):
            ok = client.post(
                "/analyze_pipeline",
                json={
                    "segments": [{"text": "hej", "start": 0, "end": 1}],
                    "original_filename": "kund-0042_samtal.wav",
                    "profile": "sales",
                },
            )
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert captured["profile"] == "callcenter"
        assert captured["qa_scorecard"] == "standard_support_v1"
        assert body["customer"]["customer_id"] == "0042"
        assert body["call_id"]
        assert body["persisted"] is True

    def test_analyze_conversation_empty_speech_is_422(
        self, customer_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        customer_env({"mode": "disabled", "id_patterns": [], "customers": {}})
        audio = tmp_path / "media" / "uploads" / "silent.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(_AUDIO)
        monkeypatch.setenv("API_STATE_DIR", str(tmp_path / "state"))
        from src.api.settings import get_api_settings

        get_api_settings.cache_clear()
        client = self._client()
        with patch(
            "src.api.services.conversation.transcribe_helper",
            return_value={"text": "", "segments": []},
        ):
            r = client.post("/analyze_conversation", json={"audio_path": str(audio)})
        assert r.status_code == 422
        assert r.json()["error_code"] == "asr_empty_transcript"
        store = CallStore(tmp_path / "state")
        failed = [doc for doc in store.list(limit=20) if doc.get("status") == "failed"]
        assert failed
        assert failed[0]["provenance"]["fail_reason"] == "asr_empty_transcript"
