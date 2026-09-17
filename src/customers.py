"""Customer registry and filename-based customer identity resolution.

The customer identifier embedded in the original audio filename is *routing
context*: it selects which organization's configuration applies to the call
(analyzer profile, QA scorecard, ASR/LLM policy). It is not authentication
or authorization and must never be treated as proof of user identity.

Registry lives in ``user_config.yaml`` under the ``customers:`` key (see
``src/install/config_schema.py``). Modes:

- ``disabled`` — resolution not applied (backward compatible default).
- ``optional`` — a filename without an ID is processed uncontrolled, but a
  present ID must resolve to a known, enabled customer.
- ``required`` — every call must resolve to a known, enabled customer.

In controlled modes (optional/required) an unknown, ambiguous or disabled
identity is a hard error — never a silent fallback to a generic profile.
"""

from __future__ import annotations

import hashlib
import logging
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .profiles import PROFILE_SPECS

logger = logging.getLogger(__name__)

CUSTOMER_ID_GROUP = "customer_id"


class CustomerResolutionError(Exception):
    """Base class for customer identity resolution failures."""

    error_code = "customer_resolution_error"


class MissingCustomerIdError(CustomerResolutionError):
    """No customer identifier could be extracted from the filename."""

    error_code = "missing_customer_id"

    def __init__(self, filename: str | None) -> None:
        self.filename = filename
        super().__init__(f"No customer identifier in filename {filename!r}")


class AmbiguousCustomerIdError(CustomerResolutionError):
    """Multiple distinct customer identifiers found in the filename."""

    error_code = "ambiguous_customer_id"

    def __init__(self, filename: str | None, candidates: list[str]) -> None:
        self.filename = filename
        self.candidates = list(candidates)
        super().__init__(f"Ambiguous customer identity in filename {filename!r}: {self.candidates}")


class UnknownCustomerError(CustomerResolutionError):
    """The extracted identifier does not match any registered customer."""

    error_code = "unknown_customer"

    def __init__(self, customer_id: str) -> None:
        self.customer_id = customer_id
        super().__init__(f"Unknown customer identifier {customer_id!r}")


class DisabledCustomerError(CustomerResolutionError):
    """The resolved customer exists but is disabled."""

    error_code = "disabled_customer"

    def __init__(self, customer_id: str) -> None:
        self.customer_id = customer_id
        super().__init__(f"Customer {customer_id!r} is disabled")


class InvalidCustomerRegistryError(CustomerResolutionError):
    """Registry configuration is invalid at resolve time."""

    error_code = "invalid_customer_registry"


class CustomerPolicyError(CustomerResolutionError):
    """Request tried to widen a customer's allowed processing policy."""

    error_code = "customer_policy_violation"


class CustomerMode(StrEnum):
    """How strictly filename-based customer identity is enforced."""

    disabled = "disabled"
    optional = "optional"
    required = "required"


class CustomerAsrPolicy(BaseModel):
    """ASR routing policy for one customer organization."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["local", "cloud"] = "local"
    allow_cloud_fallback: bool = False
    cloud_provider: Literal["deepgram"] | None = None

    @model_validator(mode="after")
    def _cloud_provider_when_needed(self) -> CustomerAsrPolicy:
        if (self.provider == "cloud" or self.allow_cloud_fallback) and not self.cloud_provider:
            raise ValueError("cloud_provider is required for cloud ASR or cloud fallback")
        return self


class CustomerLlmPolicy(BaseModel):
    """LLM routing policy for one customer organization.

    ``provider_allowlist`` is the *complete* set of allowed providers for the
    customer (including local ones such as ``lmstudio``). When ``enabled`` is
    true the allowlist must be non-empty — there is no implicit allow-all.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    provider_allowlist: list[str] = Field(default_factory=list)
    anonymize_before_llm: bool = True
    cost_budget_per_call: float = 0.08
    max_attempts: int = 3

    @field_validator("provider_allowlist")
    @classmethod
    def _normalize_allowlist(cls, v: list[str]) -> list[str]:
        return [p.strip().lower() for p in v if p and p.strip()]

    @model_validator(mode="after")
    def _enabled_requires_allowlist(self) -> CustomerLlmPolicy:
        if self.enabled and not self.provider_allowlist:
            raise ValueError("provider_allowlist must name at least one provider when llm.enabled")
        return self


class CustomerProfile(BaseModel):
    """Per-organization configuration selected by the filename identifier."""

    display_name: str = ""
    enabled: bool = True
    analyzer_profile: str = "callcenter"
    qa_scorecard: str | None = None
    aliases: list[str] = Field(default_factory=list)
    asr: CustomerAsrPolicy = Field(default_factory=CustomerAsrPolicy)
    llm: CustomerLlmPolicy = Field(default_factory=CustomerLlmPolicy)

    @field_validator("analyzer_profile")
    @classmethod
    def _known_analyzer_profile(cls, v: str) -> str:
        name = (v or "").strip().lower()
        if name not in PROFILE_SPECS:
            raise ValueError(f"unknown analyzer_profile {v!r}")
        return name

    @field_validator("qa_scorecard")
    @classmethod
    def _normalize_scorecard(cls, v: str | None) -> str | None:
        if v is None:
            return None
        name = v.strip()
        return name or None

    @field_validator("aliases")
    @classmethod
    def _normalize_aliases(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for alias in v:
            norm = normalize_customer_id(alias)
            if norm and norm not in out:
                out.append(norm)
        return out


class CustomerRegistry(BaseModel):
    """Versioned customer registry loaded from user configuration."""

    version: int = 1
    mode: CustomerMode = CustomerMode.disabled
    id_patterns: list[str] = Field(default_factory=list)
    customers: dict[str, CustomerProfile] = Field(default_factory=dict)

    @field_validator("id_patterns")
    @classmethod
    def _validate_patterns(cls, v: list[str]) -> list[str]:
        for pattern in v:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid id_pattern {pattern!r}: {exc}") from exc
            if CUSTOMER_ID_GROUP not in compiled.groupindex:
                raise ValueError(
                    f"id_pattern must contain a named group (?P<{CUSTOMER_ID_GROUP}>...)"
                )
        return v

    @field_validator("customers")
    @classmethod
    def _normalize_customer_keys(cls, v: dict[str, CustomerProfile]) -> dict[str, CustomerProfile]:
        out: dict[str, CustomerProfile] = {}
        for key, profile in v.items():
            norm = normalize_customer_id(key)
            if not norm:
                raise ValueError("customer id must not be empty")
            if norm in out:
                raise ValueError(f"duplicate customer id after normalization: {norm!r}")
            out[norm] = profile
        return out

    @model_validator(mode="after")
    def _consistent(self) -> CustomerRegistry:
        if self.mode != CustomerMode.disabled and not self.id_patterns:
            raise ValueError("id_patterns are required when mode is not 'disabled'")
        seen_aliases: set[str] = set()
        for key, profile in self.customers.items():
            for alias in profile.aliases:
                if alias in self.customers and alias != key:
                    raise ValueError(f"alias {alias!r} collides with a customer id")
                if alias in seen_aliases:
                    raise ValueError(f"duplicate alias {alias!r}")
                seen_aliases.add(alias)
        return self

    def fingerprint(self) -> str:
        """Short content hash for provenance/frozen-config references."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def find(self, raw_id: str) -> tuple[str, CustomerProfile] | None:
        """Look up a customer by identifier or alias. Returns canonical key."""
        norm = normalize_customer_id(raw_id)
        if not norm:
            return None
        direct = self.customers.get(norm)
        if direct is not None:
            return norm, direct
        for key, profile in self.customers.items():
            if norm in profile.aliases:
                return key, profile
        return None


class CustomerContext(BaseModel):
    """Frozen per-job customer configuration resolved at intake time."""

    model_config = ConfigDict(frozen=True)

    customer_id: str
    display_name: str
    analyzer_profile: str
    qa_scorecard: str | None = None
    asr: CustomerAsrPolicy
    llm: CustomerLlmPolicy
    registry_version: int
    config_fingerprint: str
    source_filename: str


def normalize_customer_id(raw: str) -> str:
    """Canonical form for customer identifiers and aliases."""
    return (raw or "").strip().lower()


@lru_cache(maxsize=64)
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def extract_customer_ids(filename: str, id_patterns: list[str]) -> list[str]:
    """Extract distinct normalized customer IDs from a filename's basename.

    Only the basename is matched — directory components never contribute to
    identity. Returns IDs in order of first appearance, deduplicated.
    """
    name = Path(filename or "").name
    found: list[str] = []
    for raw in id_patterns:
        try:
            pattern = _compile_pattern(raw)
        except re.error as exc:
            raise InvalidCustomerRegistryError(f"invalid id_pattern {raw!r}") from exc
        if CUSTOMER_ID_GROUP not in pattern.groupindex:
            raise InvalidCustomerRegistryError(
                f"id_pattern must contain a named group (?P<{CUSTOMER_ID_GROUP}>...)"
            )
        for match in pattern.finditer(name):
            norm = normalize_customer_id(match.group(CUSTOMER_ID_GROUP) or "")
            if norm and norm not in found:
                found.append(norm)
    return found


def resolve_customer(filename: str | None, registry: CustomerRegistry) -> CustomerContext | None:
    """Resolve the frozen customer context for a call from its filename.

    Returns ``None`` in disabled mode, and in optional mode when the filename
    carries no identifier. Raises a :class:`CustomerResolutionError` subclass
    for missing (required), ambiguous, unknown or disabled identities.
    """
    if registry.mode == CustomerMode.disabled:
        return None
    ids = extract_customer_ids(filename or "", registry.id_patterns)
    if not ids:
        if registry.mode == CustomerMode.required:
            raise MissingCustomerIdError(filename)
        return None
    if len(ids) > 1:
        raise AmbiguousCustomerIdError(filename, ids)
    found = registry.find(ids[0])
    if found is None:
        raise UnknownCustomerError(ids[0])
    key, profile = found
    if not profile.enabled:
        raise DisabledCustomerError(key)
    return CustomerContext(
        customer_id=key,
        display_name=profile.display_name or key,
        analyzer_profile=profile.analyzer_profile,
        qa_scorecard=profile.qa_scorecard,
        asr=profile.asr,
        llm=profile.llm,
        registry_version=registry.version,
        config_fingerprint=registry.fingerprint(),
        source_filename=Path(filename or "").name,
    )


@lru_cache
def get_customer_registry() -> CustomerRegistry:
    """Customer registry from the installed user configuration.

    Process-cached: registry changes require an API restart to take effect.
    A misconfigured registry raises — it must never silently downgrade to
    ``disabled`` in a controlled installation.
    """
    from .install.user_config import load_user_config

    return load_user_config().customers


DEFAULT_QA_SCORECARD = "standard_support_v1"


class CustomerExecutionPolicy(BaseModel):
    """Clamped execution parameters for one call.

    Request parameters may only *narrow* a customer's permissions. Widening
    ASR/LLM access is a hard error.
    """

    model_config = ConfigDict(frozen=True)

    analyzer_profile: str
    qa_scorecard: str
    asr_provider: Literal["local", "cloud"]
    cloud_fallback_local: bool
    llm_enabled: bool
    llm_provider: str | None = None
    anonymize_before_llm: bool = True
    customer_id: str | None = None
    config_fingerprint: str | None = None


def clamp_execution_policy(
    customer: CustomerContext | None,
    *,
    requested_asr_provider: str = "local",
    requested_cloud_fallback: bool = False,
    requested_llm_enabled: bool = False,
    requested_llm_provider: str | None = None,
    requested_profile: str | None = None,
) -> CustomerExecutionPolicy:
    """Apply customer policy as a ceiling over request parameters.

    When no customer is resolved (registry disabled / optional without ID),
    the request is used unchanged. A resolved customer always selects the
    analyzer profile and QA scorecard. ASR/LLM requests that exceed the
    customer allowlist raise :class:`CustomerPolicyError`.
    """
    req_asr = (requested_asr_provider or "local").strip().lower()
    if req_asr not in {"local", "cloud"}:
        req_asr = "local"
    req_llm = (requested_llm_provider or "").strip().lower() or None
    fallback = bool(requested_cloud_fallback)
    llm_on = bool(requested_llm_enabled)

    if customer is None:
        profile = (requested_profile or "callcenter").strip().lower() or "callcenter"
        return CustomerExecutionPolicy(
            analyzer_profile=profile,
            qa_scorecard=DEFAULT_QA_SCORECARD,
            asr_provider=req_asr,  # type: ignore[arg-type]
            cloud_fallback_local=fallback,
            llm_enabled=llm_on,
            llm_provider=req_llm,
            anonymize_before_llm=True,
        )

    if req_asr == "cloud" and customer.asr.provider != "cloud":
        raise CustomerPolicyError(
            f"ASR provider 'cloud' is not allowed for customer {customer.customer_id!r}"
        )
    if fallback and not customer.asr.allow_cloud_fallback:
        raise CustomerPolicyError(
            f"ASR cloud fallback is not allowed for customer {customer.customer_id!r}"
        )
    asr_provider: Literal["local", "cloud"] = (
        "cloud" if req_asr == "cloud" and customer.asr.provider == "cloud" else "local"
    )

    if llm_on and not customer.llm.enabled:
        raise CustomerPolicyError(f"LLM is not enabled for customer {customer.customer_id!r}")
    if llm_on:
        allow = set(customer.llm.provider_allowlist)
        if not req_llm or req_llm not in allow:
            raise CustomerPolicyError(
                f"LLM provider {req_llm!r} is not in the allowlist for customer "
                f"{customer.customer_id!r}"
            )

    return CustomerExecutionPolicy(
        analyzer_profile=customer.analyzer_profile,
        qa_scorecard=customer.qa_scorecard or DEFAULT_QA_SCORECARD,
        asr_provider=asr_provider,
        cloud_fallback_local=fallback and customer.asr.allow_cloud_fallback,
        llm_enabled=llm_on,
        llm_provider=req_llm if llm_on else None,
        anonymize_before_llm=customer.llm.anonymize_before_llm,
        customer_id=customer.customer_id,
        config_fingerprint=customer.config_fingerprint,
    )
