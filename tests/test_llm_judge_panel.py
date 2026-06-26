    def test_filter_changed_only_logic(self):
        """Test that we can correctly identify how many verdicts would be shown after 'Endast ändrade' filter."""
        verdicts = [
            {"original_sentiment": "positive", "judge_label": "negative"},   # changed
            {"original_sentiment": "negative", "judge_label": "negative"},   # not changed
            {"original_sentiment": "neutral", "judge_label": "positive"},    # changed
            {"original_sentiment": "positive", "judge_label": "positive"},    # not changed
        ]

        changed_count = sum(1 for v in verdicts if _is_changed(v))
        assert changed_count == 2

        # Simulate what the panel would show with filter = "Endast ändrade"
        filtered = [v for v in verdicts if _is_changed(v)]
        assert len(filtered) == 2