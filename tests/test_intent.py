"""Tests for intent classification module."""

from __future__ import annotations

from src.intent import CALL_CENTER_INTENTS, IntentClassifier


class TestIntentDefinitions:
    def test_ten_intents(self):
        assert len(CALL_CENTER_INTENTS) == 10

    def test_all_have_required_keys(self):
        for _name, data in CALL_CENTER_INTENTS.items():
            assert "id" in data
            assert "display_sv" in data
            assert "keywords" in data
            assert "priority" in data

    def test_unique_ids(self):
        ids = [d["id"] for d in CALL_CENTER_INTENTS.values()]
        assert len(ids) == len(set(ids))


class TestIntentClassifier:
    def setup_method(self):
        self.clf = IntentClassifier(backend="heuristic")

    def test_classify_account_update(self):
        intent, conf = self.clf.classify("Jag vill ändra min adress")
        assert intent == "account_update"
        assert 0 <= conf <= 1

    def test_classify_billing(self):
        intent, conf = self.clf.classify("Min faktura är för hög, varför kostar det så mycket?")
        assert intent == "billing_inquiry"

    def test_classify_technical(self):
        intent, conf = self.clf.classify("Produkten fungerar inte")
        assert intent == "technical_support"

    def test_classify_complaint(self):
        intent, conf = self.clf.classify("Det här är helt oacceptabelt och dåligt!")
        assert intent == "complaint"

    def test_classify_cancellation(self):
        intent, conf = self.clf.classify("Jag vill avsluta mitt abonnemang")
        assert intent == "cancellation"

    def test_classify_other(self):
        intent, conf = self.clf.classify("Hej hej")
        assert intent == "other"

    def test_classify_batch(self):
        texts = ["Jag vill ändra adress", "Min faktura är fel", "Hej"]
        results = self.clf.classify_batch(texts)
        assert len(results) == 3
        assert results[0][0] == "account_update"

    def test_get_intent_info(self):
        info = self.clf.get_intent_info("billing_inquiry")
        assert info is not None
        assert info["display_sv"] == "Fakturafråga"

    def test_list_intents(self):
        intents = self.clf.list_intents()
        assert len(intents) == 10
