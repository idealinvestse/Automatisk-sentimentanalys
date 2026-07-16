from src.core.models import Transcript
from src.transcription.error_codes import AsrErrorCode


def test_error_codes_are_stable_strings():
    assert AsrErrorCode.CLOUD_AUTH == "CLOUD_AUTH"
    assert AsrErrorCode.CHUNK_FAILED == "CHUNK_FAILED"


def test_transcript_warnings_roundtrip():
    t = Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        warnings=["chunk_retry:3"],
        provider="local",
    )
    d = t.to_dict()
    assert d["warnings"] == ["chunk_retry:3"]
    assert d["provider"] == "local"
    assert Transcript.from_dict(d).warnings == ["chunk_retry:3"]
