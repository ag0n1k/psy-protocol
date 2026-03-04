import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from .alignment import build_replicas, build_replicas_from_words
from .config import (
    DEFAULT_DIARIZATION_METHOD,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_EMBEDDING_MIN_DURATION,
    DEFAULT_MERGE_ADJACENT_ROLES,
    DEFAULT_QWEN_ASR_LANGUAGE,
    DEFAULT_QWEN_ASR_MODEL,
    DEFAULT_SPEAKER_EMBEDDING_DEVICE,
    DEFAULT_SPEAKER_EMBEDDING_MODEL,
    DEFAULT_TRANSCRIPTION_METHOD,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WORD_PROB_THRESHOLD,
    DEFAULT_WORD_SMOOTH_MIN_WORDS,
)
from .diarization import (
    SpeakerSegment,
    cluster_segments_by_embeddings,
    diarize_audio_mlx,
    load_audio,
    post_process_diarization,
)
from .docx_writer import create_docx
from .io_utils import load_json, save_json, save_text
from .roles import map_speakers_to_roles, parse_speaker_map
from .replica_postprocess import merge_adjacent_by_role
from .text_outputs import save_dialogue_txt, save_sentences_txt, save_timed_dialogue_txt
from .text_postprocess import postprocess_replica_text
from .audio_preprocess import preprocess_audio
from .qwen_transcribe import transcribe_audio_qwen, transcribe_per_diarization
from .whisper_transcribe import extract_words, transcribe_audio


@dataclass
class ProcessingOptions:
    output_docx: Optional[Union[str, Path]] = None
    transcript_dir: Union[str, Path] = "transcripts"
    whisper_model: str = DEFAULT_WHISPER_MODEL
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    speaker_map: Optional[str] = None
    fio: str = ""
    group: str = ""
    date: str = ""
    topic: str = ""
    task: str = ""
    min_segment_duration: float = 0.5
    silence_threshold: float = 0.35
    max_speakers: int = 2
    speaker_embedding_model: str = DEFAULT_SPEAKER_EMBEDDING_MODEL
    speaker_embedding_device: str = DEFAULT_SPEAKER_EMBEDDING_DEVICE
    word_smooth_min_words: int = DEFAULT_WORD_SMOOTH_MIN_WORDS
    word_prob_threshold: float = DEFAULT_WORD_PROB_THRESHOLD
    merge_gap: float = 0.3
    sandwich_max_duration: float = 2.0
    chunk_size: int = 160000
    overlap: int = 16000
    force_whisper: bool = False
    force_diarization: bool = False
    preprocess_audio: bool = True
    word_timestamps: bool = True
    diarization_method: str = DEFAULT_DIARIZATION_METHOD
    transcription_method: str = DEFAULT_TRANSCRIPTION_METHOD  # 'qwen_asr' | 'whisper'
    qwen_asr_model: str = DEFAULT_QWEN_ASR_MODEL
    qwen_asr_language: str = DEFAULT_QWEN_ASR_LANGUAGE
    merge_adjacent_roles: bool = DEFAULT_MERGE_ADJACENT_ROLES


def _serialize_segments(segments: List[SpeakerSegment]) -> list:
    return [{'start': s.start, 'end': s.end, 'speaker': s.speaker} for s in segments]


def _deserialize_segments(data: list) -> List[SpeakerSegment]:
    return [
        SpeakerSegment(start=float(s['start']), end=float(s['end']), speaker=str(s['speaker']))
        for s in data
    ]


def _is_diarization_cache_valid(payload: dict, opts: 'ProcessingOptions') -> bool:
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
    opts: 'ProcessingOptions',
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
    segments = post_process_diarization(
        segments,
        merge_gap=opts.merge_gap,
        sandwich_max_duration=opts.sandwich_max_duration,
    )
    logging.info('Diarization: segments after post-processing %d', len(segments))
    save_json(save_path, {
        'method': 'embedding_clustering_v4',
        'segments': _serialize_segments(segments),
        'params': {
            'diarization_method': opts.diarization_method,
            'num_speakers': opts.max_speakers,
            'merge_gap': opts.merge_gap,
            'sandwich_max_duration': opts.sandwich_max_duration,
            'silence_threshold': opts.silence_threshold,
            'min_segment_duration': opts.min_segment_duration,
            'embedding_model': opts.speaker_embedding_model,
            'embedding_device': opts.speaker_embedding_device,
        },
    })
    logging.info('Diarization: post-processing saved %s', save_path)
    emit('diarization', 92.0, 'Diarization post-processing saved')
    return segments


def _run_diarization(
    opts: ProcessingOptions,
    processed_audio_path: Path,
    transcript_dir: Path,
    emit: Callable,
) -> List[SpeakerSegment]:
    """Run diarization with caching and post-processing. Returns SpeakerSegment list."""
    diarization_post_path = transcript_dir / 'diarization_post.json'
    emit('diarization', 82.0, 'Starting diarization')
    logging.info('Diarization method: %s', opts.diarization_method)

    if diarization_post_path.exists() and not opts.force_diarization:
        payload = load_json(diarization_post_path)
        if _is_diarization_cache_valid(payload, opts):
            logging.info('Diarization: loading from cache %s', diarization_post_path)
            emit('diarization', 92.0, 'Diarization loaded from cache')
            return _deserialize_segments(payload.get('segments', []))

    return _run_mlx_diarization(opts, processed_audio_path, transcript_dir, diarization_post_path, emit)


def process_audio_file(
    audio_path: Union[str, Path],
    options: Optional[ProcessingOptions] = None,
    progress_callback: Optional[Callable[[str, Optional[float], str], None]] = None,
) -> Tuple[Path, Path]:
    def emit(stage: str, percent: Optional[float], message: str) -> None:
        if progress_callback:
            progress_callback(stage, percent, message)

    opts = options or ProcessingOptions()
    audio_path = Path(audio_path).expanduser()
    output_docx = (
        Path(opts.output_docx).expanduser()
        if opts.output_docx
        else audio_path.with_suffix(".docx")
    )

    emit("start", 0.0, "Starting processing")
    logging.info("Processing started")
    logging.info("Audio: %s", audio_path)
    logging.info("Transcription method: %s", opts.transcription_method)
    if opts.transcription_method == 'qwen_asr':
        logging.info("Qwen ASR model: %s", opts.qwen_asr_model)
    else:
        logging.info("Whisper model: %s", opts.whisper_model)
    logging.info('Diarization method: %s', opts.diarization_method)
    if opts.diarization_method == 'mlx_segmentation':
        logging.info('Diarization model: %s', opts.diarization_model)
        logging.info(
            'Diarization params: min_duration=%.2f silence_threshold=%.2f chunk=%d overlap=%d',
            opts.min_segment_duration,
            opts.silence_threshold,
            opts.chunk_size,
            opts.overlap,
        )
        logging.info('Embeddings: model=%s device=%s', opts.speaker_embedding_model, opts.speaker_embedding_device)
    logging.info('Clustering: num_speakers=%d', opts.max_speakers)
    logging.info('Word-smoothing: min_words=%d', opts.word_smooth_min_words)
    logging.info('Word-filter: prob_threshold=%.2f', opts.word_prob_threshold)
    logging.info(
        'Post-processing: merge_gap=%.2f sandwich_max_duration=%.2f',
        opts.merge_gap,
        opts.sandwich_max_duration,
    )

    transcript_root = Path(opts.transcript_dir).expanduser()
    transcript_dir = transcript_root / audio_path.stem
    transcript_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Cache dir: %s", transcript_dir)
    emit("prepare", 5.0, "Prepared cache directory")

    processed_audio_path = audio_path
    if opts.preprocess_audio:
        preprocessed_wav = transcript_dir / 'audio_preprocessed.wav'
        if not preprocessed_wav.exists() or opts.force_whisper:
            emit('prepare', 6.0, 'Preprocessing audio')
            preprocess_audio(str(audio_path), str(preprocessed_wav))
        else:
            logging.info('Audio preprocessing: using cached %s', preprocessed_wav)
        processed_audio_path = preprocessed_wav

    transcript_json_path = transcript_dir / "transcript.json"
    transcript_txt_path = transcript_dir / "transcript.txt"

    is_qwen_transcription = opts.transcription_method == 'qwen_asr'

    # --- QWEN-ASR path: diarize first, then per-segment transcription ---
    if is_qwen_transcription:
        force = opts.force_whisper or opts.force_diarization

        diarization_segments = _run_diarization(
            opts, processed_audio_path, transcript_dir, emit=emit,
        )

        qwen_segments_path = transcript_dir / 'qwen_segments.json'
        if qwen_segments_path.exists() and not force:
            replicas = load_json(qwen_segments_path)
            emit('whisper', 95.0, 'Qwen replicas loaded from cache')
            logging.info('Qwen: replicas loaded from cache %s', qwen_segments_path)
        else:
            emit('whisper', 10.0, 'Qwen per-segment transcription started')
            logging.info('Qwen: per-segment transcription started')
            replicas = transcribe_per_diarization(
                str(processed_audio_path),
                diarization_segments,
                opts.qwen_asr_model,
                tmp_dir=str(transcript_dir),
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
        logging.info('Replicas after postprocessing: %d', len(replicas))

    # --- WHISPER path: transcribe first, diarize, align ---
    else:
        whisper_segments_path = transcript_dir / "whisper_segments.json"
        transcript_meta_path = transcript_dir / "transcript_meta.json"

        use_word_timestamps = opts.word_timestamps

        cache_valid = transcript_json_path.exists() and not opts.force_whisper
        if cache_valid:
            if transcript_meta_path.exists():
                cached_meta = load_json(transcript_meta_path)
                if cached_meta.get('transcription_method') != opts.transcription_method:
                    logging.info('Transcription method changed, invalidating cache')
                    cache_valid = False
            else:
                logging.info('Transcription meta missing, invalidating cache')
                cache_valid = False

        if cache_valid:
            logging.info("Transcription: loading from cache %s", transcript_json_path)
            whisper_result = load_json(transcript_json_path)
            emit("whisper", 80.0, "Transcript loaded from cache")
        else:
            emit("whisper", 5.0, "Transcription started")

            def transcription_progress(percent: float) -> None:
                mapped = 5.0 + (75.0 * (percent / 100.0))
                emit("whisper", mapped, f"Transcription {int(percent)}%")

            logging.info(
                "Whisper: starting transcription (word_timestamps=%s)", use_word_timestamps,
            )
            whisper_result = transcribe_audio(
                str(processed_audio_path),
                opts.whisper_model,
                word_timestamps=use_word_timestamps,
                progress_callback=transcription_progress,
            )
            save_json(transcript_json_path, whisper_result)
            save_json(transcript_meta_path, {'transcription_method': opts.transcription_method})
            logging.info("Transcription: cache saved %s", transcript_json_path)
            emit("whisper", 80.0, "Transcription completed")

        if opts.force_whisper or not transcript_txt_path.exists():
            save_text(transcript_txt_path, whisper_result.get("text", ""))
            logging.info("Whisper: transcript saved %s", transcript_txt_path)

        if opts.force_whisper or not whisper_segments_path.exists():
            whisper_segments = whisper_result.get("segments", [])
            save_json(whisper_segments_path, {"segments": whisper_segments})
            logging.info("Whisper: segments saved %s", whisper_segments_path)
        else:
            logging.info("Whisper: segments from cache %s", whisper_segments_path)
            whisper_segments_payload = load_json(whisper_segments_path)
            whisper_segments = whisper_segments_payload.get("segments", [])

        sentences_txt_path = transcript_dir / "sentences.txt"
        if opts.force_whisper or not sentences_txt_path.exists():
            save_sentences_txt(sentences_txt_path, whisper_segments)
            logging.info("Sentences TXT: %s", sentences_txt_path)

        diarization_segments = _run_diarization(
            opts, processed_audio_path, transcript_dir, emit=emit,
        )

        words = (
            extract_words(whisper_result, prob_threshold=opts.word_prob_threshold)
            if use_word_timestamps
            else []
        )
        if use_word_timestamps and words:
            logging.info("Whisper: word timestamps %d", len(words))
            replicas = build_replicas_from_words(
                words,
                diarization_segments,
                smooth_min_words=opts.word_smooth_min_words,
            )
        else:
            if use_word_timestamps:
                logging.warning("No word timestamps available, falling back to segments")
            replicas = build_replicas(whisper_segments, diarization_segments)

        for replica in replicas:
            replica['text'] = postprocess_replica_text(replica['text'])
        replicas = [r for r in replicas if r['text']]
        logging.info('Replicas after postprocessing: %d', len(replicas))

    emit('replicas', 95.0, 'Building output replicas')
    explicit_map = parse_speaker_map(opts.speaker_map)
    role_map = map_speakers_to_roles(replicas, explicit_map)
    for replica in replicas:
        replica['role'] = role_map.get(replica['speaker'], 'К')

    if is_qwen_transcription:
        if opts.merge_adjacent_roles:
            replicas = merge_adjacent_by_role(replicas)
            logging.info('Role merge: merged adjacent qwen replicas to %d entries', len(replicas))

    metadata = {
        'ФИО': opts.fio,
        'Номер группы': opts.group,
        'Дата': opts.date,
        'Тема протокола': opts.topic,
        'Задание': opts.task,
    }

    logging.info('Generating DOCX: %s', output_docx)
    emit('output', 98.0, 'Generating DOCX and TXT')
    try:
        create_docx(
            output_path=str(output_docx),
            replicas=replicas,
            metadata=metadata,
        )
    except Exception:
        logging.exception('DOCX generation or validation failed: %s', output_docx)
        raise
    if not output_docx.exists():
        raise RuntimeError(f'DOCX output file was not created: {output_docx}')
    logging.info('DOCX ready: %s (%d bytes)', output_docx, output_docx.stat().st_size)
    dialogue_txt_path = output_docx.with_suffix('.txt')
    save_dialogue_txt(dialogue_txt_path, replicas)
    logging.info('Dialogue TXT: %s', dialogue_txt_path)

    timed_txt_path = transcript_dir / 'timed_dialogue.txt'
    save_timed_dialogue_txt(timed_txt_path, replicas)
    logging.info('Timed dialogue TXT: %s', timed_txt_path)

    logging.info('Whisper TXT (raw): %s', transcript_txt_path)
    logging.info('Done')
    emit('done', 100.0, 'Done')
    return output_docx, transcript_txt_path
