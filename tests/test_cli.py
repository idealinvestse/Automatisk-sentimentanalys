"""Basic CLI tests using Typer's CliRunner (covers argument handling, profiles, lexicon paths, error exits)."""

from __future__ import annotations

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_sentiment_help_runs():
    result = runner.invoke(app, ["sentiment", "--help"])
    assert result.exit_code == 0
    assert "Kör svensk sentimentanalys" in result.output or "sentiment" in result.output.lower()


def test_sentiment_single_text(monkeypatch):
    def fake_analyze_smart(texts, **kwargs):
        # Return (results, meta) as analyze_smart does; simulate profile auto + no lexicon here
        res = [{"label": "positiv", "score": 0.92} for _ in texts]
        meta = {"profile": kwargs.get("profile") or "default", "model": "fake"}
        return res, meta

    monkeypatch.setattr("src.sentiment.analyze_smart", fake_analyze_smart)
    monkeypatch.setattr("src.cli.analyze_smart", fake_analyze_smart)

    result = runner.invoke(app, ["sentiment", "--text", "Det här var bra!"])
    # With fake, should succeed (exit 0); covers CLI parsing + call to analyze_smart path
    assert result.exit_code == 0
    assert "positiv" in result.output or "Profil" in result.output


def test_sentiment_requires_one_source():
    result = runner.invoke(app, ["sentiment"])
    assert result.exit_code != 0
    assert "Ange en källa" in result.output or "text" in result.output.lower()


def test_sentiment_csv_input(monkeypatch, tmp_path):
    csvp = tmp_path / "in.csv"
    csvp.write_text("text\nBra service\nDålig support\n", encoding="utf-8")

    def fake_analyze_smart(texts, **kwargs):
        res = [{"label": "positiv", "score": 0.8}, {"label": "negativ", "score": 0.75}]
        meta = {"profile": kwargs.get("profile") or "default", "model": "fake"}
        return res, meta

    monkeypatch.setattr("src.sentiment.analyze_smart", fake_analyze_smart)
    monkeypatch.setattr("src.cli.analyze_smart", fake_analyze_smart)

    result = runner.invoke(app, ["sentiment", "--csv-file", str(csvp), "--text-column", "text"])
    assert result.exit_code == 0
    assert (
        "positiv" in result.output
        or "negativ" in result.output
        or "csv" in (result.output or "").lower()
    )


def test_sentiment_with_lexicon_file(monkeypatch, tmp_path):
    lexp = tmp_path / "lex.csv"
    lexp.write_text("term,polarity\nbra,0.8\n", encoding="utf-8")

    called = {}

    def fake_analyze_smart(texts, **kwargs):
        # Simulate analyze_smart: returns (scores, meta) and records lexicon args as the CLI now passes them through
        called["lex"] = kwargs.get("lexicon_file")
        called["w"] = kwargs.get("lexicon_weight")
        res = [
            [
                {"label": "positiv", "score": 0.6},
                {"label": "neutral", "score": 0.3},
                {"label": "negativ", "score": 0.1},
            ]
            for _ in texts
        ]
        meta = {
            "profile": kwargs.get("profile") or "forum",
            "model": "fake",
            "lexicon_file": kwargs.get("lexicon_file"),
            "lexicon_weight": kwargs.get("lexicon_weight"),
        }
        return res, meta

    monkeypatch.setattr("src.sentiment.analyze_smart", fake_analyze_smart)
    monkeypatch.setattr("src.cli.analyze_smart", fake_analyze_smart)
    # Patch blend at lexicon source too (in case any direct path or future), though analyze_smart handles blend
    monkeypatch.setattr("src.lexicon.blend_results_with_lexicon", lambda t, r, lf, w: r)

    result = runner.invoke(
        app,
        [
            "sentiment",
            "--text",
            "Tack bra",
            "--lexicon-file",
            str(lexp),
            "--lexicon-weight",
            "0.25",
            "--return-all-scores",
        ],
    )
    assert result.exit_code == 0
    assert called.get("w") == 0.25
    assert called.get("lex") == str(lexp)


def test_transcribe_help():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
