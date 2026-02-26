from typing import Dict, List, Optional, Tuple

from .diarization import SpeakerSegment


def assign_speakers_to_spans(
    spans: List[Tuple[float, float]],
    diarization_segments: List[SpeakerSegment],
) -> List[str]:
    diarization_segments = sorted(diarization_segments, key=lambda s: s.start)
    assigned: List[str] = []
    idx = 0
    last_speaker = diarization_segments[0].speaker if diarization_segments else "SPEAKER_00"

    for start, end in spans:
        while idx < len(diarization_segments) and diarization_segments[idx].end <= start:
            idx += 1

        overlaps: Dict[str, float] = {}
        j = idx
        while j < len(diarization_segments) and diarization_segments[j].start < end:
            dseg = diarization_segments[j]
            overlap = max(0.0, min(end, dseg.end) - max(start, dseg.start))
            if overlap > 0:
                overlaps[dseg.speaker] = overlaps.get(dseg.speaker, 0.0) + overlap
            j += 1

        if overlaps:
            speaker = max(overlaps.items(), key=lambda item: item[1])[0]
        else:
            speaker = last_speaker
        assigned.append(speaker)
        last_speaker = speaker

    return assigned


def assign_speakers_to_segments(
    whisper_segments: List[Dict],
    diarization_segments: List[SpeakerSegment],
) -> List[str]:
    spans = [
        (float(seg.get("start", 0.0)), float(seg.get("end", seg.get("start", 0.0))))
        for seg in whisper_segments
    ]
    return assign_speakers_to_spans(spans, diarization_segments)


def build_replicas(
    whisper_segments: List[Dict],
    diarization_segments: List[SpeakerSegment],
) -> List[Dict[str, str]]:
    speakers = assign_speakers_to_segments(whisper_segments, diarization_segments)
    replicas: List[Dict[str, str]] = []

    for seg, speaker in zip(whisper_segments, speakers):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if not replicas or replicas[-1]["speaker"] != speaker:
            replicas.append({"speaker": speaker, "text": text})
        else:
            replicas[-1]["text"] = f"{replicas[-1]['text']} {text}"

    return replicas


def build_replicas_from_words(
    words: List[Dict[str, float]],
    diarization_segments: List[SpeakerSegment],
    smooth_min_words: int,
) -> List[Dict[str, str]]:
    if not words:
        return []
    spans = [(w["start"], w["end"]) for w in words]
    speakers = assign_speakers_to_spans(spans, diarization_segments)
    speakers = smooth_word_speakers(speakers, min_words=smooth_min_words)

    replicas: List[Dict[str, str]] = []
    current_speaker: Optional[str] = None
    tokens: List[str] = []

    for word, speaker in zip(words, speakers):
        token = word["word"]
        if current_speaker is None:
            current_speaker = speaker
        if speaker != current_speaker:
            text = "".join(tokens).strip()
            if text:
                replicas.append({"speaker": current_speaker, "text": text})
            tokens = []
            current_speaker = speaker
        tokens.append(token)

    text = "".join(tokens).strip()
    if text and current_speaker is not None:
        replicas.append({"speaker": current_speaker, "text": text})

    return replicas


def smooth_word_speakers(
    speakers: List[str],
    min_words: int = 2,
) -> List[str]:
    if min_words <= 1 or len(speakers) < 3:
        return speakers
    smoothed = speakers[:]
    runs: List[Tuple[int, int, str]] = []
    start = 0
    current = speakers[0]
    for idx, speaker in enumerate(speakers[1:], start=1):
        if speaker != current:
            runs.append((start, idx - 1, current))
            start = idx
            current = speaker
    runs.append((start, len(speakers) - 1, current))

    for i in range(1, len(runs) - 1):
        run_start, run_end, run_speaker = runs[i]
        run_len = run_end - run_start + 1
        prev_speaker = runs[i - 1][2]
        next_speaker = runs[i + 1][2]
        if run_len <= min_words and prev_speaker == next_speaker:
            for idx in range(run_start, run_end + 1):
                smoothed[idx] = prev_speaker

    return smoothed
