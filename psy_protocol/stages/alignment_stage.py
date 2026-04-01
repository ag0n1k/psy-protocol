import logging
from pathlib import Path
from typing import Callable, List

from ..alignment import assign_segments_without_merge, assign_words_without_merge
from ..io_utils import load_json, save_json, save_text
from ..models import AlignmentResult, AsrResult, DiarizationResult, Replica
from ..replica_postprocess import merge_adjacent_by_role
from ..roles import map_speakers_to_roles, parse_speaker_map
from ..text_postprocess import postprocess_replica_text
from .diarization_stage import run_diarization


def _make_replicas_with_roles(raw_segments: List[dict], opts) -> List[Replica]:
    """Assign roles to raw speaker-labeled segments and return Replica list."""
    explicit_map = parse_speaker_map(opts.speaker_map)
    role_map = map_speakers_to_roles(raw_segments, explicit_map)
    result = []
    for seg in raw_segments:
        speaker = seg['speaker']
        role = role_map.get(speaker, 'К')
        result.append(Replica(
            speaker=speaker,
            role=role,
            text=seg['text'],
            start=seg['start'],
            end=seg['end'],
        ))
    return result


def run_alignment(
    asr_result: AsrResult,
    diarization_result: DiarizationResult,
    opts,
) -> AlignmentResult:
    """Stage 3a: align ASR output with diarization. Returns UNMERGED segments with predicted roles."""
    diar_segs = diarization_result.segments
    whisper_segs_dicts = [
        {'start': s.start, 'end': s.end, 'text': s.text}
        for s in asr_result.segments
    ]
    words = [
        {'start': w.start, 'end': w.end, 'word': w.word}
        for w in asr_result.words
    ]

    if words:
        logging.info('Alignment: using word timestamps (%d words)', len(words))
        raw_segments = assign_words_without_merge(
            words,
            whisper_segs_dicts,
            diar_segs,
            smooth_min_words=opts.word_smooth_min_words,
        )
    else:
        if opts.word_timestamps:
            logging.warning('Alignment: no word timestamps available, falling back to segments')
        raw_segments = assign_segments_without_merge(whisper_segs_dicts, diar_segs)

    for seg in raw_segments:
        seg['text'] = postprocess_replica_text(seg['text'])
    raw_segments = [s for s in raw_segments if s['text']]
    logging.info('Alignment: %d unmerged segments after postprocessing', len(raw_segments))

    segments = _make_replicas_with_roles(raw_segments, opts)
    return AlignmentResult(segments=segments)


def run_qwen_combined(
    audio_path: Path,
    opts,
    cache_dir: Path,
    emit: Callable,
) -> AlignmentResult:
    """Qwen-ASR combined path: diarize → per-segment transcription → AlignmentResult."""
    from ..qwen_transcribe import transcribe_per_diarization

    force = opts.force_whisper or opts.force_diarization
    transcript_json_path = cache_dir / 'transcript.json'
    transcript_txt_path = cache_dir / 'transcript.txt'
    alignment_result_path = cache_dir / 'alignment_result.json'

    diarization_result = run_diarization(audio_path, opts, cache_dir, emit)

    qwen_segments_path = cache_dir / 'qwen_segments.json'
    if qwen_segments_path.exists() and not force:
        replicas = load_json(qwen_segments_path)
        emit('whisper', 95.0, 'Qwen replicas loaded from cache')
        logging.info('Qwen: replicas loaded from cache %s', qwen_segments_path)
    else:
        emit('whisper', 10.0, 'Qwen per-segment transcription started')
        logging.info('Qwen: per-segment transcription started')
        replicas = transcribe_per_diarization(
            str(audio_path),
            diarization_result.segments,
            opts.qwen_asr_model,
            tmp_dir=str(cache_dir),
            language=opts.qwen_asr_language,
            progress_callback=lambda p: emit('whisper', 10.0 + p * 0.85, f'Qwen {int(p)}%'),
        )
        save_json(qwen_segments_path, replicas)
        logging.info('Qwen: per-segment transcription done, replicas=%d', len(replicas))
        emit('whisper', 95.0, 'Qwen per-segment transcription done')

    full_text = ' '.join(r['text'] for r in replicas)
    if not transcript_txt_path.exists() or force:
        save_text(transcript_txt_path, full_text)
    if not transcript_json_path.exists() or force:
        save_json(transcript_json_path, {'text': full_text, 'segments': replicas})

    for replica in replicas:
        replica['text'] = postprocess_replica_text(replica['text'])
    replicas = [r for r in replicas if r['text']]
    logging.info('Qwen: replicas after postprocessing: %d', len(replicas))

    segments = _make_replicas_with_roles(replicas, opts)
    result = AlignmentResult(segments=segments)
    save_json(alignment_result_path, result.to_dict())
    logging.info('Alignment result saved %s', alignment_result_path)
    return result


def merge_segments_to_replicas(segments: List[Replica]) -> List[Replica]:
    """Stage 3b: merge adjacent same-role segments into replicas."""
    if not segments:
        return []
    dicts = [s.to_dict() for s in segments]
    merged_dicts = merge_adjacent_by_role(dicts)
    return [
        Replica(
            speaker=d['speaker'],
            role=d['role'],
            text=d['text'],
            start=float(d['start']),
            end=float(d['end']),
        )
        for d in merged_dicts
    ]
