from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str


@dataclass
class AsrWord:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class AsrResult:
    text: str
    segments: list[AsrSegment]
    words: list[AsrWord]  # empty for qwen_asr path
    method: str           # 'whisper' | 'qwen_asr'
    model: str

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'segments': [
                {'start': s.start, 'end': s.end, 'text': s.text}
                for s in self.segments
            ],
            'words': [
                {'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability}
                for w in self.words
            ],
            'method': self.method,
            'model': self.model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AsrResult':
        segments = [
            AsrSegment(float(s['start']), float(s['end']), s['text'])
            for s in data.get('segments', [])
        ]
        words = [
            AsrWord(float(w['start']), float(w['end']), w['word'], float(w.get('probability', 1.0)))
            for w in data.get('words', [])
        ]
        return cls(
            text=data.get('text', ''),
            segments=segments,
            words=words,
            method=data.get('method', 'whisper'),
            model=data.get('model', ''),
        )


@dataclass
class DiarizationResult:
    segments: list  # list[SpeakerSegment]
    method: str
    params: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            'method': self.method,
            'segments': [
                {'start': s.start, 'end': s.end, 'speaker': s.speaker}
                for s in self.segments
            ],
            'params': self.params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DiarizationResult':
        from .diarization import SpeakerSegment
        segments = [
            SpeakerSegment(start=float(s['start']), end=float(s['end']), speaker=str(s['speaker']))
            for s in data.get('segments', [])
        ]
        return cls(
            segments=segments,
            method=data.get('method', ''),
            params=data.get('params', {}),
        )


@dataclass
class Replica:
    speaker: str
    role: str   # 'К' | 'Т'
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {
            'speaker': self.speaker,
            'role': self.role,
            'text': self.text,
            'start': self.start,
            'end': self.end,
        }


@dataclass
class AlignmentResult:
    """Stage 3a result — UNMERGED segments with predicted roles."""
    segments: list[Replica]

    def to_dict(self) -> dict:
        return {'segments': [r.to_dict() for r in self.segments]}

    @classmethod
    def from_dict(cls, data: dict) -> 'AlignmentResult':
        segments = [
            Replica(
                speaker=s['speaker'],
                role=s['role'],
                text=s['text'],
                start=float(s['start']),
                end=float(s['end']),
            )
            for s in data.get('segments', [])
        ]
        return cls(segments=segments)


@dataclass
class LlmResult:
    segments: list[Replica]  # segments with corrected roles (before merge)
    analysis: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'segments': [r.to_dict() for r in self.segments],
            'analysis': self.analysis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LlmResult':
        segments = [
            Replica(
                speaker=s['speaker'],
                role=s['role'],
                text=s['text'],
                start=float(s['start']),
                end=float(s['end']),
            )
            for s in data.get('segments', [])
        ]
        return cls(segments=segments, analysis=data.get('analysis'))
