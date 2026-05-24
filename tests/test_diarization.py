"""Tests for diarization module."""

from __future__ import annotations

from src.diarization import DiarizationPipeline, DiarizationResult, SpeakerSegment


class TestSpeakerSegment:
    def test_creation(self):
        s = SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_0")
        assert s.start == 0.0
        assert s.end == 5.0
        assert s.speaker == "SPEAKER_0"
        assert s.confidence == 1.0


class TestDiarizationResult:
    def test_empty(self):
        r = DiarizationResult()
        assert r.num_speakers == 0
        assert r.segments == []
        assert r.speakers == []

    def test_speaker_timeline(self):
        segs = [
            SpeakerSegment(0.0, 3.0, "SPEAKER_0"),
            SpeakerSegment(3.0, 6.0, "SPEAKER_1"),
            SpeakerSegment(6.0, 8.0, "SPEAKER_0"),
        ]
        r = DiarizationResult(
            segments=segs, num_speakers=2, speakers=["SPEAKER_0", "SPEAKER_1"], audio_duration_s=8.0
        )
        timeline = r.speaker_timeline("SPEAKER_0")
        assert len(timeline) == 2
        assert timeline[0] == (0.0, 3.0)
        assert timeline[1] == (6.0, 8.0)

    def test_speaker_ratio(self):
        segs = [
            SpeakerSegment(0.0, 4.0, "SPEAKER_0"),
            SpeakerSegment(4.0, 10.0, "SPEAKER_1"),
        ]
        r = DiarizationResult(
            segments=segs,
            num_speakers=2,
            speakers=["SPEAKER_0", "SPEAKER_1"],
            audio_duration_s=10.0,
        )
        assert r.speaker_ratio("SPEAKER_0") == 0.4
        assert r.speaker_ratio("SPEAKER_1") == 0.6

    def test_to_dict(self):
        segs = [SpeakerSegment(0.0, 2.0, "SPEAKER_0")]
        r = DiarizationResult(
            segments=segs,
            num_speakers=1,
            speakers=["SPEAKER_0"],
            backend="heuristic",
            processing_time_s=0.5,
        )
        d = r.to_dict()
        assert d["num_speakers"] == 1
        assert d["backend"] == "heuristic"
        assert len(d["segments"]) == 1


class TestDiarizationPipeline:
    def test_init_heuristic(self):
        dp = DiarizationPipeline(backend="heuristic")
        assert dp.backend == "heuristic"
        assert dp._pipeline is None

    def test_assign_speakers_to_segments(self):
        dp = DiarizationPipeline()
        asr_segs = [
            {"start": 0.0, "end": 5.0, "text": "Hej"},
            {"start": 5.0, "end": 10.0, "text": "Hej själv"},
        ]
        diar_segs = [
            SpeakerSegment(0.0, 4.0, "SPEAKER_0"),
            SpeakerSegment(4.0, 10.0, "SPEAKER_1"),
        ]
        diar = DiarizationResult(
            segments=diar_segs, num_speakers=2, speakers=["SPEAKER_0", "SPEAKER_1"]
        )
        result = dp.assign_speakers_to_segments(asr_segs, diar)
        assert result[0]["speaker"] == "SPEAKER_0"
        assert result[1]["speaker"] == "SPEAKER_1"

    def test_assign_speakers_empty_diarization(self):
        dp = DiarizationPipeline()
        asr_segs = [{"start": 0.0, "end": 5.0, "text": "Hej"}]
        diar = DiarizationResult()
        result = dp.assign_speakers_to_segments(asr_segs, diar)
        assert result[0]["speaker"] == "UNKNOWN"
