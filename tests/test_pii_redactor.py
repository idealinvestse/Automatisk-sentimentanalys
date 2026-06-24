"""Tests for PII redaction hardening (Luhn, expanded Swedish names, addresses)."""

import pytest
from src.llm.pii_redactor import (
    _is_valid_luhn,
    redact_pii,
    redact_segments,
    _CREDIT_CARD_RE,
    _ADDRESS_RE,
    _COMMON_SWEDISH_FIRST_NAMES,
)


class TestLuhnValidation:
    """Luhn checksum validation for credit cards."""

    def test_valid_luhn_visa(self):
        """Valid Visa test card passes Luhn."""
        assert _is_valid_luhn("4111111111111111") is True

    def test_valid_luhn_mastercard(self):
        """Valid Mastercard test card passes Luhn."""
        assert _is_valid_luhn("5555555555554444") is True

    def test_invalid_luhn_invoice_number(self):
        """13-digit invoice number without Luhn should fail (no false positive)."""
        assert _is_valid_luhn("1234567890123") is False

    def test_invalid_luhn_case_id(self):
        """16-digit case number without Luhn should fail."""
        assert _is_valid_luhn("9876543210987654") is False

    def test_luhn_with_spaces(self):
        """Luhn ignores spaces (cleaned before validation)."""
        digits = "4111 1111 1111 1111".replace(" ", "")
        assert _is_valid_luhn(digits) is True

    def test_luhn_empty_or_non_digit(self):
        """Non-digit input returns False."""
        assert _is_valid_luhn("") is False
        assert _is_valid_luhn("not-a-number") is False


class TestCreditCardRedaction:
    """Credit card redaction only on valid Luhn."""

    def test_redacts_valid_cc_only(self):
        """Should redact 4111111111111111 but not invoice numbers."""
        text = "Kort: 4111111111111111, faktura 1234567890123"
        redacted, events = redact_pii(text)
        assert "[REDACTED_CC]" in redacted
        assert "1234567890123" in redacted  # invoice not redacted
        cc_events = [e for e in events if e["type"] == "credit_card"]
        assert len(cc_events) == 1

    def test_no_false_positive_on_fakturanummer(self):
        """Fakturanummer with 13-16 digits must NOT trigger redaction."""
        text = "Fakturanummer 1234567890123456, kundnummer 9876543210987654"
        redacted, events = redact_pii(text)
        assert "[REDACTED_CC]" not in redacted
        assert "1234567890123456" in redacted
        cc_events = [e for e in events if e["type"] == "credit_card"]
        assert len(cc_events) == 0


class TestSwedishNameExpansion:
    """Expanded name list (SCB top ~50) with title heuristic."""

    @pytest.mark.parametrize("name", ["Anna", "Erik", "Lars", "Karl", "Maria", "Anders"])
    def test_common_swedish_names_in_set(self, name):
        """Top Swedish first names are present in the heuristic set."""
        assert name.lower() in _COMMON_SWEDISH_FIRST_NAMES

    def test_title_prefix_triggers_name_redaction(self):
        """'Herr Anna' or 'Dr Erik' triggers name heuristic."""
        text = "Tala med herr Erik om ärendet."
        redacted, events = redact_pii(text)
        name_events = [e for e in events if e["type"] == "name"]
        assert len(name_events) >= 1
        assert "[REDACTED_NAME]" in redacted


class TestSwedishAddressExpansion:
    """Swedish street suffix expansion (gatan, vägen, torget, etc.)."""

    @pytest.mark.parametrize(
        "address",
        [
            "Storgatan 5",
            "Drottningvägen 12",
            "Brunkebergstorg 3",
            "Kungsgatan 42",
            "Vasaplatsen 8",
        ],
    )
    def test_swedish_street_addresses_match(self, address):
        """Swedish addresses with expanded suffixes are matched."""
        # Extract the street part that _ADDRESS_RE should catch
        match = _ADDRESS_RE.search(address)
        assert match is not None, f"Address '{address}' should match _ADDRESS_RE"


class TestIdempotency:
    """Redaction is idempotent (running twice yields same result)."""

    def test_redact_twice_is_noop(self):
        """Redacting already-redacted text produces identical output."""
        original = "Ring 070-123 45 67 eller maila test@example.com"
        once, _ = redact_pii(original)
        twice, _ = redact_pii(once)
        assert once == twice
        assert "[REDACTED_PHONE]" in twice
        assert "[REDACTED_EMAIL]" in twice


class TestPersonnummer:
    """Personnummer (Swedish SSN) patterns with/without delimiter."""

    def test_personnummer_with_dash(self):
        """19850101-1234 matches."""
        text = "Personnummer: 19850101-1234"
        redacted, events = redact_pii(text)
        pnr_events = [e for e in events if e["type"] == "personnummer"]
        assert len(pnr_events) == 1
        assert "[REDACTED_PNR]" in redacted

    def test_personnummer_without_dash(self):
        """198501011234 matches."""
        text = "Personnummer 198501011234"
        redacted, events = redact_pii(text)
        pnr_events = [e for e in events if e["type"] == "personnummer"]
        assert len(pnr_events) == 1
        assert "[REDACTED_PNR]" in redacted


@pytest.fixture
def sample_segments():
    return [
        {"text": "Kund 4111111111111111 ringde, faktura 1234567890123"},
        {"text": "Storgatan 5, herr Erik"},
    ]


def test_redact_segments_with_profile(sample_segments, monkeypatch):
    """redact_segments honors profile llm.anonymize_before_llm flag."""
    # Mock profile resolution to enable redaction
    import src.llm.pii_redactor as pr

    def fake_resolve(*args, **kwargs):
        return None, {"llm": {"anonymize_before_llm": True}}

    monkeypatch.setattr(pr, "resolve_profile", fake_resolve)
    redacted, log = redact_segments(sample_segments, profile_name="callcenter", return_log=True)
    assert log.total_redacted > 0
    assert any("4111111111111111" not in seg["text"] for seg in redacted)
