from typing import Dict, List, Optional, Tuple

from .diarization import SpeakerSegment


def assign_segments_without_merge(
    whisper_segments: List[Dict],
    diarization_segments: List[SpeakerSegment],
) -> List[Dict]:
    """Assign speakers to whisper segments without merging adjacent same-speaker entries."""
    speakers = assign_speakers_to_segments(whisper_segments, diarization_segments)
    result = []
    for seg, speaker in zip(whisper_segments, speakers):
        text = seg.get('text', '').strip()
        if not text:
            continue
        result.append({
            'speaker': speaker,
            'text': text,
            'start': float(seg.get('start', 0.0)),
            'end': float(seg.get('end', 0.0)),
        })
    return result


def assign_words_without_merge(
    words: List[Dict],
    whisper_segments: List[Dict],
    diarization_segments: List[SpeakerSegment],
    smooth_min_words: int,
) -> List[Dict]:
    """Assign speakers at word level, smooth, then group back into whisper segments."""
    if not words:
        return []
    spans = [(w['start'], w['end']) for w in words]
    speakers = assign_speakers_to_spans(spans, diarization_segments)
    speakers = smooth_word_speakers(speakers, min_words=smooth_min_words)

    word_spk_pairs = list(zip(words, speakers))
    result = []
    for seg in whisper_segments:
        seg_start = float(seg.get('start', 0.0))
        seg_end = float(seg.get('end', seg_start))
        text = seg.get('text', '').strip()
        if not text:
            continue

        seg_speakers = [
            spk for w, spk in word_spk_pairs
            if w['start'] >= seg_start - 0.05 and w['end'] <= seg_end + 0.05
        ]

        if seg_speakers:
            counts: Dict[str, int] = {}
            for spk in seg_speakers:
                counts[spk] = counts.get(spk, 0) + 1
            speaker = max(counts, key=lambda k: counts[k])
        else:
            speaker = assign_speakers_to_segments([seg], diarization_segments)[0]

        result.append({
            'speaker': speaker,
            'text': text,
            'start': seg_start,
            'end': seg_end,
        })
    return result


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
        seg_start = float(seg.get('start', 0.0))
        seg_end = float(seg.get('end', seg_start))
        if not replicas or replicas[-1]['speaker'] != speaker:
            replicas.append({
                'speaker': speaker,
                'text': text,
                'start': seg_start,
                'end': seg_end,
            })
        else:
            replicas[-1]['text'] = f"{replicas[-1]['text']} {text}"
            replicas[-1]['end'] = seg_end

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
    replica_start: float = 0.0
    replica_end: float = 0.0

    for word, speaker in zip(words, speakers):
        token = word['word']
        if current_speaker is None:
            current_speaker = speaker
            replica_start = word['start']
        if speaker != current_speaker:
            text = ''.join(tokens).strip()
            if text:
                replicas.append({
                    'speaker': current_speaker,
                    'text': text,
                    'start': replica_start,
                    'end': replica_end,
                })
            tokens = []
            current_speaker = speaker
            replica_start = word['start']
        tokens.append(token)
        replica_end = word['end']

    text = ''.join(tokens).strip()
    if text and current_speaker is not None:
        replicas.append({
            'speaker': current_speaker,
            'text': text,
            'start': replica_start,
            'end': replica_end,
        })

    return replicas


def _build_runs(speakers: List[str]) -> List[Tuple[int, int, str]]:
    runs: List[Tuple[int, int, str]] = []
    start = 0
    current = speakers[0]
    for idx, speaker in enumerate(speakers[1:], start=1):
        if speaker != current:
            runs.append((start, idx - 1, current))
            start = idx
            current = speaker
    runs.append((start, len(speakers) - 1, current))
    return runs


def smooth_word_speakers(
    speakers: List[str],
    min_words: int = 2,
) -> List[str]:
    if min_words <= 1 or len(speakers) < 3:
        return speakers
    smoothed = speakers[:]

    # First pass: sandwich smoothing — runs <= min_words between matching neighbors
    runs = _build_runs(smoothed)
    for i in range(1, len(runs) - 1):
        run_start, run_end, _ = runs[i]
        run_len = run_end - run_start + 1
        prev_speaker = runs[i - 1][2]
        next_speaker = runs[i + 1][2]
        if run_len <= min_words and prev_speaker == next_speaker:
            for idx in range(run_start, run_end + 1):
                smoothed[idx] = prev_speaker

    # Second pass: remove single-word outliers regardless of sandwich context.
    # Assign the word to the speaker of the longer neighboring run.
    changed = True
    while changed:
        changed = False
        runs = _build_runs(smoothed)
        for i, (run_start, run_end, _) in enumerate(runs):
            if run_end - run_start + 1 != 1:
                continue
            prev_len = runs[i - 1][1] - runs[i - 1][0] + 1 if i > 0 else 0
            next_len = runs[i + 1][1] - runs[i + 1][0] + 1 if i < len(runs) - 1 else 0
            if prev_len == 0 and next_len == 0:
                continue
            if prev_len >= next_len:
                replacement = runs[i - 1][2]
            else:
                replacement = runs[i + 1][2]
            if smoothed[run_start] != replacement:
                smoothed[run_start] = replacement
                changed = True

    return smoothed
