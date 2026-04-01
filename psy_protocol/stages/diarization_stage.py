import logging
from pathlib import Path
from typing import Callable, List

from ..diarization import (
    SpeakerSegment,
    cluster_segments_by_embeddings,
    diarize_audio_mlx,
    load_audio,
    post_process_diarization,
)
from ..io_utils import load_json, save_json
from ..models import DiarizationResult


def _serialize_segments(segments: List[SpeakerSegment]) -> list:
    return [{'start': s.start, 'end': s.end, 'speaker': s.speaker} for s in segments]


def _deserialize_segments(data: list) -> List[SpeakerSegment]:
    return [
        SpeakerSegment(start=float(s['start']), end=float(s['end']), speaker=str(s['speaker']))
        for s in data
    ]


def _is_diarization_cache_valid(payload: dict, opts) -> bool:
    params = payload.get('params', {})
    if params.get('diarization_method') != opts.diarization_method:
        return False
    if params.get('num_speakers') != opts.max_speakers:
        return False
    if params.get('merge_gap') != opts.merge_gap:
        return False
    if params.get('sandwich_max_duration') != opts.sandwich_max_duration:
        return False
    return (
        payload.get('method') == 'embedding_clustering_v4'
        and params.get('silence_threshold') == opts.silence_threshold
        and params.get('min_segment_duration') == opts.min_segment_duration
        and params.get('embedding_model') == opts.speaker_embedding_model
        and params.get('embedding_device') == opts.speaker_embedding_device
    )


def _run_mlx_diarization(
    opts,
    processed_audio_path: Path,
    transcript_dir: Path,
    save_path: Path,
    emit: Callable,
) -> List[SpeakerSegment]:
    raw_path = transcript_dir / 'diarization.json'
    raw_segments = None

    if raw_path.exists() and not opts.force_diarization:
        raw_payload = load_json(raw_path)
        raw_params = raw_payload.get('params', {})
        if (
            raw_params.get('diarization_method') == opts.diarization_method
            and raw_params.get('silence_threshold') == opts.silence_threshold
            and raw_params.get('min_segment_duration') == opts.min_segment_duration
            and raw_params.get('chunk_size') == opts.chunk_size
            and raw_params.get('overlap') == opts.overlap
        ):
            logging.info('Diarization: raw segments from cache %s', raw_path)
            raw_segments = _deserialize_segments(raw_payload.get('segments', []))
            emit('diarization', 86.0, 'Loaded raw diarization segments')
        else:
            logging.info('Diarization: raw cache params mismatch, re-running segmentation')

    if raw_segments is None:
        logging.info('Diarization: running MLX segmentation')
        emit('diarization', 86.0, 'Running diarization segmentation')
        raw_segments = diarize_audio_mlx(
            str(processed_audio_path),
            opts.diarization_model,
            min_duration=opts.min_segment_duration,
            chunk_size=opts.chunk_size,
            overlap=opts.overlap,
            silence_threshold=opts.silence_threshold,
        )
        save_json(raw_path, {
            'segments': _serialize_segments(raw_segments),
            'params': {
                'diarization_method': opts.diarization_method,
                'silence_threshold': opts.silence_threshold,
                'min_segment_duration': opts.min_segment_duration,
                'chunk_size': opts.chunk_size,
                'overlap': opts.overlap,
            },
        })
        logging.info('Diarization: raw segments saved %s', raw_path)

    waveform, sample_rate = load_audio(str(processed_audio_path))
    logging.info('Diarization: raw segments count %d', len(raw_segments))
    emit('diarization', 90.0, 'Clustering speaker segments')
    segments = cluster_segments_by_embeddings(
        waveform,
        sample_rate,
        raw_segments,
        num_speakers=opts.max_speakers,
        embedding_model_id=opts.speaker_embedding_model,
        embedding_device=opts.speaker_embedding_device,
    )
    logging.info('Diarization: segments after clustering %d', len(segments))
    params = {
        'diarization_method': opts.diarization_method,
        'num_speakers': opts.max_speakers,
        'merge_gap': opts.merge_gap,
        'sandwich_max_duration': opts.sandwich_max_duration,
        'silence_threshold': opts.silence_threshold,
        'min_segment_duration': opts.min_segment_duration,
        'embedding_model': opts.speaker_embedding_model,
        'embedding_device': opts.speaker_embedding_device,
    }
    segments = post_process_diarization(
        segments,
        merge_gap=opts.merge_gap,
        sandwich_max_duration=opts.sandwich_max_duration,
    )
    logging.info('Diarization: segments after post-processing %d', len(segments))
    save_json(save_path, {
        'method': 'embedding_clustering_v4',
        'segments': _serialize_segments(segments),
        'params': params,
    })
    logging.info('Diarization: post-processing saved %s', save_path)
    emit('diarization', 92.0, 'Diarization post-processing saved')
    return segments, params


def run_diarization(
    audio_path: Path,
    opts,
    cache_dir: Path,
    emit: Callable,
) -> DiarizationResult:
    """Run diarization with caching. Returns DiarizationResult."""
    diarization_post_path = cache_dir / 'diarization_post.json'
    emit('diarization', 82.0, 'Starting diarization')
    logging.info('Diarization method: %s', opts.diarization_method)

    if diarization_post_path.exists() and not opts.force_diarization:
        payload = load_json(diarization_post_path)
        if _is_diarization_cache_valid(payload, opts):
            logging.info('Diarization: loading from cache %s', diarization_post_path)
            emit('diarization', 92.0, 'Diarization loaded from cache')
            segments = _deserialize_segments(payload.get('segments', []))
            return DiarizationResult(
                segments=segments,
                method=payload.get('method', 'embedding_clustering_v4'),
                params=payload.get('params', {}),
            )

    segments, params = _run_mlx_diarization(
        opts, audio_path, cache_dir, diarization_post_path, emit,
    )
    return DiarizationResult(
        segments=segments,
        method='embedding_clustering_v4',
        params=params,
    )
