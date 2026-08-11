import argparse
import logging
from pathlib import Path

from .config import (
    DEFAULT_DIARIZATION_METHOD,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_LLM_API_BASE,
    DEFAULT_LLM_ENABLED,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TASKS,
    DEFAULT_MERGE_ADJACENT_ROLES,
    DEFAULT_QWEN_ASR_MODEL,
    DEFAULT_SPEAKER_EMBEDDING_DEVICE,
    DEFAULT_SPEAKER_EMBEDDING_MODEL,
    DEFAULT_TRANSCRIPTION_METHOD,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WORD_PROB_THRESHOLD,
    DEFAULT_WORD_SMOOTH_MIN_WORDS,
    LOG_FORMAT,
)
from .dialogue_parser import parse_dialogue_text
from .pipeline import ProcessingOptions, process_audio_file


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    parser.add_argument(
        '--transcript-dir',
        default='transcripts',
        help='Каталог для сохранения транскриптов и диаризации',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        help='Уровень логирования (DEBUG, INFO, WARNING, ERROR)',
    )


def _add_diarization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--diarization-model',
        default=DEFAULT_DIARIZATION_MODEL,
        help='Путь или HF-репозиторий модели pyannote MLX',
    )
    parser.add_argument(
        '--min-segment-duration',
        type=float,
        default=0.5,
        help='Минимальная длительность сегмента (сек)',
    )
    parser.add_argument(
        '--silence-threshold',
        type=float,
        default=0.35,
        help='Порог уверенности для тишины',
    )
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=2,
        help='Число спикеров для кластеризации',
    )
    parser.add_argument(
        '--speaker-embedding-model',
        default=DEFAULT_SPEAKER_EMBEDDING_MODEL,
        help='Модель спикер-эмбеддингов (SpeechBrain)',
    )
    parser.add_argument(
        '--speaker-embedding-device',
        default=DEFAULT_SPEAKER_EMBEDDING_DEVICE,
        help='Устройство для спикер-эмбеддингов (cpu/mps/cuda)',
    )
    parser.add_argument(
        '--merge-gap',
        type=float,
        default=0.2,
        help='Сшивать соседние сегменты одного спикера (сек)',
    )
    parser.add_argument(
        '--sandwich-max-duration',
        type=float,
        default=1.0,
        help='Макс. длительность вставки между одинаковыми спикерами (сек)',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=160000,
        help='Размер чанка для диаризации (семплы)',
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=16000,
        help='Перекрытие чанков (семплы)',
    )
    parser.add_argument(
        '--diarization-method',
        default=DEFAULT_DIARIZATION_METHOD,
        choices=['mlx_segmentation'],
        help='Метод диаризации',
    )
    parser.add_argument(
        '--force-diarization',
        action='store_true',
        help='Перезапустить диаризацию и обновить кэш',
    )


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--llm',
        action='store_true',
        dest='llm_enabled',
        default=DEFAULT_LLM_ENABLED,
        help='Включить LLM-улучшение (Stage 4)',
    )
    parser.add_argument(
        '--llm-api-base',
        default=DEFAULT_LLM_API_BASE,
        help='Base URL для LLM API (LMStudio OpenAI-compatible)',
    )
    parser.add_argument(
        '--llm-model',
        default=DEFAULT_LLM_MODEL,
        help='Имя модели в LMStudio',
    )
    parser.add_argument(
        '--llm-tasks',
        default=','.join(DEFAULT_LLM_TASKS),
        help='Задачи LLM через запятую: role_validation,text_correction,analysis',
    )


def _build_full_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser('full', help='Полный pipeline (по умолчанию)')
    p.add_argument('--audio', required=True, help='Путь к аудиофайлу')
    p.add_argument(
        '--output-docx',
        default=None,
        help='Путь к выходному DOCX (по умолчанию: <audio>.docx)',
    )
    _add_common_args(p)
    p.add_argument(
        '--whisper-model',
        default=DEFAULT_WHISPER_MODEL,
        help='Путь или HF-репозиторий модели Whisper-MLX',
    )
    p.add_argument('--speaker-map', default=None, help='SPEAKER_00=К,SPEAKER_01=Т')
    p.add_argument('--fio', default='', help='ФИО')
    p.add_argument('--group', default='', help='Номер группы')
    p.add_argument('--date', default='', help='Дата')
    p.add_argument('--topic', default='', help='Тема протокола')
    p.add_argument('--task', default='', help='Задание')
    p.add_argument(
        '--word-smooth-min-words',
        type=int,
        default=DEFAULT_WORD_SMOOTH_MIN_WORDS,
        help='Сглаживание word-диаризации',
    )
    p.add_argument(
        '--word-prob-threshold',
        type=float,
        default=DEFAULT_WORD_PROB_THRESHOLD,
        help='Порог вероятности слова',
    )
    p.add_argument(
        '--force-whisper',
        action='store_true',
        help='Перезапустить Whisper и обновить кэш',
    )
    p.add_argument(
        '--no-word-timestamps',
        action='store_false',
        dest='word_timestamps',
        default=True,
        help='Отключить word timestamps в Whisper',
    )
    p.add_argument(
        '--swap',
        action='store_true',
        help='Поменять К↔Т (shortcut для --speaker-map SPEAKER_00=Т,SPEAKER_01=К)',
    )
    p.add_argument(
        '--transcription-method',
        default=DEFAULT_TRANSCRIPTION_METHOD,
        choices=['whisper', 'qwen_asr'],
        help='Метод транскрибации',
    )
    p.add_argument(
        '--qwen-asr-model',
        default=DEFAULT_QWEN_ASR_MODEL,
        help='Модель Qwen3-ASR',
    )
    p.add_argument(
        '--merge-adjacent-roles',
        action='store_true',
        dest='merge_adjacent_roles',
        default=DEFAULT_MERGE_ADJACENT_ROLES,
        help='Склеивать соседние реплики с одинаковой ролью',
    )
    p.add_argument(
        '--no-merge-adjacent-roles',
        action='store_false',
        dest='merge_adjacent_roles',
        help='Не склеивать соседние реплики с одинаковой ролью',
    )
    _add_diarization_args(p)
    _add_llm_args(p)
    return p


def _build_asr_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser('asr', help='Только Stage 1: ASR транскрибация')
    p.add_argument('--audio', required=True, help='Путь к аудиофайлу')
    _add_common_args(p)
    p.add_argument(
        '--whisper-model',
        default=DEFAULT_WHISPER_MODEL,
        help='Путь или HF-репозиторий модели Whisper-MLX',
    )
    p.add_argument(
        '--word-prob-threshold',
        type=float,
        default=DEFAULT_WORD_PROB_THRESHOLD,
        help='Порог вероятности слова',
    )
    p.add_argument(
        '--force-whisper',
        action='store_true',
        help='Перезапустить Whisper',
    )
    p.add_argument(
        '--no-word-timestamps',
        action='store_false',
        dest='word_timestamps',
        default=True,
    )
    p.add_argument(
        '--transcription-method',
        default=DEFAULT_TRANSCRIPTION_METHOD,
        choices=['whisper', 'qwen_asr'],
    )
    return p


def _build_diarize_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser('diarize', help='Только Stage 2: диаризация')
    p.add_argument('--audio', required=True, help='Путь к аудиофайлу')
    _add_common_args(p)
    _add_diarization_args(p)
    return p


def _build_align_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser('align', help='Stage 3: выравнивание из кэша')
    p.add_argument('--cache-dir', required=True, help='Путь к каталогу кэша (transcripts/<stem>)')
    _add_common_args(p)
    p.add_argument('--speaker-map', default=None, help='SPEAKER_00=К,SPEAKER_01=Т')
    p.add_argument(
        '--word-smooth-min-words',
        type=int,
        default=DEFAULT_WORD_SMOOTH_MIN_WORDS,
    )
    p.add_argument(
        '--swap',
        action='store_true',
        help='Поменять К↔Т',
    )
    return p


def _build_llm_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser('llm', help='Stage 4: LLM-улучшение из кэша')
    p.add_argument('--cache-dir', required=True, help='Путь к каталогу кэша (transcripts/<stem>)')
    _add_common_args(p)
    _add_llm_args(p)
    return p


def _build_from_text_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        'from-text',
        help='Собрать DOCX из текстового файла с репликами (К:/Т:)',
    )
    p.add_argument('--input', required=True, help='Путь к текстовому файлу с репликами')
    p.add_argument('--output-docx', default=None, help='Путь к выходному DOCX (по умолчанию: <input>.docx)')
    p.add_argument('--fio', default='', help='ФИО')
    p.add_argument('--group', default='', help='Номер группы')
    p.add_argument('--date', default='', help='Дата')
    p.add_argument('--topic', default='', help='Тема протокола')
    p.add_argument('--task', default='', help='Задание')
    p.add_argument('--log-level', default='INFO', help='Уровень логирования')
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Транскрибировать аудио, определить спикеров и собрать DOCX.',
    )

    # Keep backward compatibility: if first arg is not a subcommand, treat as 'full'
    subparsers = parser.add_subparsers(dest='command')
    _build_full_parser(subparsers)
    _build_asr_parser(subparsers)
    _build_diarize_parser(subparsers)
    _build_align_parser(subparsers)
    _build_llm_parser(subparsers)
    _build_from_text_parser(subparsers)

    # Legacy flat args for backward compatibility (no subcommand)
    parser.add_argument('--audio', default=None, help='Путь к аудиофайлу')
    parser.add_argument('--output-docx', default=None)
    parser.add_argument('--transcript-dir', default='transcripts')
    parser.add_argument('--whisper-model', default=DEFAULT_WHISPER_MODEL)
    parser.add_argument('--diarization-model', default=DEFAULT_DIARIZATION_MODEL)
    parser.add_argument('--speaker-map', default=None)
    parser.add_argument('--fio', default='')
    parser.add_argument('--group', default='')
    parser.add_argument('--date', default='')
    parser.add_argument('--topic', default='')
    parser.add_argument('--task', default='')
    parser.add_argument('--min-segment-duration', type=float, default=0.5)
    parser.add_argument('--silence-threshold', type=float, default=0.35)
    parser.add_argument('--max-speakers', type=int, default=2)
    parser.add_argument('--speaker-embedding-model', default=DEFAULT_SPEAKER_EMBEDDING_MODEL)
    parser.add_argument('--speaker-embedding-device', default=DEFAULT_SPEAKER_EMBEDDING_DEVICE)
    parser.add_argument('--word-smooth-min-words', type=int, default=DEFAULT_WORD_SMOOTH_MIN_WORDS)
    parser.add_argument('--word-prob-threshold', type=float, default=DEFAULT_WORD_PROB_THRESHOLD)
    parser.add_argument('--merge-gap', type=float, default=0.2)
    parser.add_argument('--sandwich-max-duration', type=float, default=1.0)
    parser.add_argument('--chunk-size', type=int, default=160000)
    parser.add_argument('--overlap', type=int, default=16000)
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--force-whisper', action='store_true')
    parser.add_argument('--force-diarization', action='store_true')
    parser.add_argument(
        '--no-word-timestamps', action='store_false', dest='word_timestamps', default=True,
    )
    parser.add_argument(
        '--diarization-method', default=DEFAULT_DIARIZATION_METHOD, choices=['mlx_segmentation'],
    )
    parser.add_argument('--swap', action='store_true')
    parser.add_argument(
        '--transcription-method', default=DEFAULT_TRANSCRIPTION_METHOD, choices=['whisper', 'qwen_asr'],
    )
    parser.add_argument('--qwen-asr-model', default=DEFAULT_QWEN_ASR_MODEL)
    parser.add_argument(
        '--merge-adjacent-roles', action='store_true', dest='merge_adjacent_roles',
        default=DEFAULT_MERGE_ADJACENT_ROLES,
    )
    parser.add_argument('--no-merge-adjacent-roles', action='store_false', dest='merge_adjacent_roles')
    parser.add_argument('--llm', action='store_true', dest='llm_enabled', default=DEFAULT_LLM_ENABLED)
    parser.add_argument('--llm-api-base', default=DEFAULT_LLM_API_BASE)
    parser.add_argument('--llm-model', default=DEFAULT_LLM_MODEL)
    parser.add_argument('--llm-tasks', default=','.join(DEFAULT_LLM_TASKS))
    return parser


def _options_from_args(args) -> ProcessingOptions:
    speaker_map = getattr(args, 'speaker_map', None)
    if getattr(args, 'swap', False) and not speaker_map:
        speaker_map = 'SPEAKER_00=Т,SPEAKER_01=К'
    llm_tasks_raw = getattr(args, 'llm_tasks', ','.join(DEFAULT_LLM_TASKS))
    llm_tasks = [t.strip() for t in llm_tasks_raw.split(',') if t.strip()]
    return ProcessingOptions(
        output_docx=getattr(args, 'output_docx', None),
        transcript_dir=getattr(args, 'transcript_dir', 'transcripts'),
        whisper_model=getattr(args, 'whisper_model', DEFAULT_WHISPER_MODEL),
        diarization_model=getattr(args, 'diarization_model', DEFAULT_DIARIZATION_MODEL),
        speaker_map=speaker_map,
        fio=getattr(args, 'fio', ''),
        group=getattr(args, 'group', ''),
        date=getattr(args, 'date', ''),
        topic=getattr(args, 'topic', ''),
        task=getattr(args, 'task', ''),
        min_segment_duration=getattr(args, 'min_segment_duration', 0.5),
        silence_threshold=getattr(args, 'silence_threshold', 0.35),
        max_speakers=getattr(args, 'max_speakers', 2),
        speaker_embedding_model=getattr(args, 'speaker_embedding_model', DEFAULT_SPEAKER_EMBEDDING_MODEL),
        speaker_embedding_device=getattr(args, 'speaker_embedding_device', DEFAULT_SPEAKER_EMBEDDING_DEVICE),
        word_smooth_min_words=getattr(args, 'word_smooth_min_words', DEFAULT_WORD_SMOOTH_MIN_WORDS),
        word_prob_threshold=getattr(args, 'word_prob_threshold', DEFAULT_WORD_PROB_THRESHOLD),
        merge_gap=getattr(args, 'merge_gap', 0.2),
        sandwich_max_duration=getattr(args, 'sandwich_max_duration', 1.0),
        chunk_size=getattr(args, 'chunk_size', 160000),
        overlap=getattr(args, 'overlap', 16000),
        force_whisper=getattr(args, 'force_whisper', False),
        force_diarization=getattr(args, 'force_diarization', False),
        word_timestamps=getattr(args, 'word_timestamps', True),
        diarization_method=getattr(args, 'diarization_method', DEFAULT_DIARIZATION_METHOD),
        transcription_method=getattr(args, 'transcription_method', DEFAULT_TRANSCRIPTION_METHOD),
        qwen_asr_model=getattr(args, 'qwen_asr_model', DEFAULT_QWEN_ASR_MODEL),
        merge_adjacent_roles=getattr(args, 'merge_adjacent_roles', DEFAULT_MERGE_ADJACENT_ROLES),
        llm_enabled=getattr(args, 'llm_enabled', DEFAULT_LLM_ENABLED),
        llm_api_base=getattr(args, 'llm_api_base', DEFAULT_LLM_API_BASE),
        llm_model=getattr(args, 'llm_model', DEFAULT_LLM_MODEL),
        llm_tasks=llm_tasks,
    )


def _run_full(args) -> None:
    options = _options_from_args(args)
    process_audio_file(args.audio, options)


def _run_asr(args) -> None:
    from pathlib import Path
    from .stages.asr import run_asr
    from .audio_preprocess import preprocess_audio

    opts = _options_from_args(args)
    audio_path = Path(args.audio).expanduser()
    transcript_root = Path(args.transcript_dir).expanduser()
    cache_dir = transcript_root / audio_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    processed = audio_path
    if opts.preprocess_audio:
        preprocessed_wav = cache_dir / 'audio_preprocessed.wav'
        if not preprocessed_wav.exists() or opts.force_whisper:
            preprocess_audio(str(audio_path), str(preprocessed_wav))
        processed = preprocessed_wav

    result = run_asr(processed, opts, cache_dir, emit=lambda *_: None)
    logging.info('ASR done: %d segments, %d words', len(result.segments), len(result.words))


def _run_diarize(args) -> None:
    from pathlib import Path
    from .stages.diarization_stage import run_diarization
    from .audio_preprocess import preprocess_audio

    opts = _options_from_args(args)
    audio_path = Path(args.audio).expanduser()
    transcript_root = Path(args.transcript_dir).expanduser()
    cache_dir = transcript_root / audio_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    processed = audio_path
    if opts.preprocess_audio:
        preprocessed_wav = cache_dir / 'audio_preprocessed.wav'
        if not preprocessed_wav.exists() or opts.force_diarization:
            preprocess_audio(str(audio_path), str(preprocessed_wav))
        processed = preprocessed_wav

    result = run_diarization(processed, opts, cache_dir, emit=lambda *_: None)
    logging.info('Diarization done: %d segments', len(result.segments))


def _run_align(args) -> None:
    from pathlib import Path
    from .models import AsrResult, DiarizationResult
    from .stages.alignment_stage import run_alignment, merge_segments_to_replicas
    from .io_utils import load_json, save_json

    opts = _options_from_args(args)
    if getattr(args, 'swap', False) and not opts.speaker_map:
        opts.speaker_map = 'SPEAKER_00=Т,SPEAKER_01=К'

    cache_dir = Path(args.cache_dir).expanduser()
    asr_result = AsrResult.from_dict(load_json(cache_dir / 'asr_result.json'))
    diar_result = DiarizationResult.from_dict(load_json(cache_dir / 'diarization_post.json'))

    alignment = run_alignment(asr_result, diar_result, opts)
    save_json(cache_dir / 'alignment_result.json', alignment.to_dict())
    logging.info('Alignment done: %d segments saved to %s', len(alignment.segments), cache_dir / 'alignment_result.json')


def _run_from_text(args) -> None:
    from .docx_writer import create_docx

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        logging.error('Input file not found: %s', input_path)
        return

    text = input_path.read_text(encoding='utf-8')
    replicas = parse_dialogue_text(text)
    if not replicas:
        logging.error('No replicas found in %s (expected lines like "К: ..." or "Т: ...")', input_path)
        return

    logging.info('Parsed %d replicas from %s', len(replicas), input_path)

    output_docx = (
        Path(args.output_docx).expanduser()
        if args.output_docx
        else input_path.with_suffix('.docx')
    )

    metadata = {
        'ФИО': args.fio,
        'Номер группы': args.group,
        'Дата': args.date,
        'Тема протокола': args.topic,
        'Задание': args.task,
    }

    create_docx(
        output_path=str(output_docx),
        replicas=replicas,
        metadata=metadata,
    )
    logging.info('DOCX ready: %s', output_docx)


def _run_llm(args) -> None:
    from pathlib import Path
    from .models import AlignmentResult
    from .stages.llm_stage import run_llm
    from .io_utils import load_json, save_json

    opts = _options_from_args(args)
    opts.llm_enabled = True

    cache_dir = Path(args.cache_dir).expanduser()
    alignment = AlignmentResult.from_dict(load_json(cache_dir / 'alignment_result.json'))

    llm_result = run_llm(alignment, opts)
    save_json(cache_dir / 'llm_result.json', llm_result.to_dict())
    logging.info('LLM done: %d segments', len(llm_result.segments))
    if llm_result.analysis:
        analysis_path = cache_dir / 'analysis.txt'
        analysis_path.write_text(llm_result.analysis, encoding='utf-8')
        logging.info('Analysis saved: %s', analysis_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(args, 'log_level', 'INFO').upper(), format=LOG_FORMAT)

    command = getattr(args, 'command', None)

    # No subcommand: legacy flat args or default to 'full'
    if command is None:
        if not args.audio:
            parser.print_help()
            return
        _run_full(args)
        return

    dispatch = {
        'full': _run_full,
        'asr': _run_asr,
        'diarize': _run_diarize,
        'align': _run_align,
        'llm': _run_llm,
        'from-text': _run_from_text,
    }
    handler = dispatch.get(command)
    if handler:
        handler(args)
    else:
        parser.print_help()
