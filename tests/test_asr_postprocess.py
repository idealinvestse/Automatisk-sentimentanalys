from src.core.models import Segment, Transcript
from src.transcription.postprocess import filter_hallucinations


def _t(texts):
    return Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=10.0,
        processing_time=1.0,
        segments=[Segment(start=i, end=i + 1, text=x) for i, x in enumerate(texts)],
    )


def test_drops_thanks_for_watching():
    out = filter_hallucinations(_t(["Hej", "Thanks for watching", "Hejdå"]))
    assert [s.text for s in out.segments] == ["Hej", "Hejdå"]
    assert any("hallucination_dropped" in w for w in out.warnings)


def test_drops_swedish_ghost():
    out = filter_hallucinations(_t(["Tack för att ni tittade"]))
    assert out.segments == []


def test_drops_repetition_loops():
    out = filter_hallucinations(_t(["ja ja ja ja ja"]))
    assert out.segments == []


def test_keeps_normal_swedish():
    out = filter_hallucinations(_t(["Jag vill ha hjälp med fakturan"]))
    assert len(out.segments) == 1
