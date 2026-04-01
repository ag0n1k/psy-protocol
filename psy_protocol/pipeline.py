import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from .audio_preprocess import preprocess_audio
from .config import (
    DEFAULT_DIARIZATION_METHOD,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_EMBEDDING_MIN_DURATION,
    DEFAULT_LLM_API_BASE,
    DEFAULT_LLM_ENABLED,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TASKS,
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
from .io_utils import save_json
from .stages.alignment_stage import merge_segments_to_replicas, run_alignment, run_qwen_combined
from .stages.asr import run_asr
from .stages.diarization_stage import run_diarization
from .stages.llm_stage import run_llm
from .stages.output_stage import run_output


@dataclass
class ProcessingOptions:
    output_docx: Optional[Union[str, Path]] = None
    transcript_dir: Union[str, Path] = 'transcripts'
    whisper_model: str = DEFAULT_WHISPER_MODEL
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    speaker_map: Optional[str] = None
    fio: str = ''
    group: str = ''
    date: str = ''
    topic: str = ''
    task: str = ''
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
    # LLM options
    llm_enabled: bool = DEFAULT_LLM_ENABLED
    llm_api_base: str = DEFAULT_LLM_API_BASE
    llm_model: str = DEFAULT_LLM_MODEL
    llm_tasks: List[str] = field(default_factory=lambda: list(DEFAULT_LLM_TASKS))


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

    emit('start', 0.0, 'Starting processing')
    logging.info('Processing started')
    logging.info('Audio: %s', audio_path)
    logging.info('Transcription method: %s', opts.transcription_method)
    if opts.transcription_method == 'qwen_asr':
        logging.info('Qwen ASR model: %s', opts.qwen_asr_model)
    else:
        logging.info('Whisper model: %s', opts.whisper_model)
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
        logging.info(
            'Embeddings: model=%s device=%s',
            opts.speaker_embedding_model,
            opts.speaker_embedding_device,
        )
    logging.info('Clustering: num_speakers=%d', opts.max_speakers)
    logging.info('Word-smoothing: min_words=%d', opts.word_smooth_min_words)
    logging.info('Word-filter: prob_threshold=%.2f', opts.word_prob_threshold)
    logging.info(
        'Post-processing: merge_gap=%.2f sandwich_max_duration=%.2f',
        opts.merge_gap,
        opts.sandwich_max_duration,
    )
    logging.info('LLM enabled: %s', opts.llm_enabled)

    # Stage 0: setup cache dir
    transcript_root = Path(opts.transcript_dir).expanduser()
    cache_dir = transcript_root / audio_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    logging.info('Cache dir: %s', cache_dir)
    emit('prepare', 5.0, 'Prepared cache directory')

    # Stage 0: audio preprocessing
    processed_audio_path = audio_path
    if opts.preprocess_audio:
        preprocessed_wav = cache_dir / 'audio_preprocessed.wav'
        if not preprocessed_wav.exists() or opts.force_whisper:
            emit('prepare', 6.0, 'Preprocessing audio')
            preprocess_audio(str(audio_path), str(preprocessed_wav))
        else:
            logging.info('Audio preprocessing: using cached %s', preprocessed_wav)
        processed_audio_path = preprocessed_wav

    # Stages 1–3a: ASR + diarization + alignment → UNMERGED segments with predicted roles
    if opts.transcription_method == 'qwen_asr':
        alignment = run_qwen_combined(processed_audio_path, opts, cache_dir, emit)
    else:
        asr_result = run_asr(processed_audio_path, opts, cache_dir, emit)
        diar_result = run_diarization(processed_audio_path, opts, cache_dir, emit)
        alignment = run_alignment(asr_result, diar_result, opts)
        save_json(cache_dir / 'alignment_result.json', alignment.to_dict())
        logging.info('Alignment result saved')

    emit('replicas', 95.0, 'Building output replicas')

    # Stage 4: LLM corrects roles on UNMERGED segments
    if opts.llm_enabled:
        logging.info('LLM: starting enhancement')
        emit('llm', 95.5, 'LLM enhancement started')
        llm_result = run_llm(alignment, opts)
        save_json(cache_dir / 'llm_result.json', llm_result.to_dict())
        logging.info('LLM result saved')
        segments = llm_result.segments
        emit('llm', 97.0, 'LLM enhancement done')
    else:
        segments = alignment.segments

    # Stage 3b: merge adjacent same-role segments into replicas
    replicas = merge_segments_to_replicas(segments)
    logging.info('Replicas after merge: %d', len(replicas))

    # Stage 5: output
    emit('output', 98.0, 'Generating DOCX and TXT')
    output_docx, txt_path = run_output(replicas, opts, audio_path, cache_dir)

    logging.info('Done')
    emit('done', 100.0, 'Done')
    return output_docx, txt_path
