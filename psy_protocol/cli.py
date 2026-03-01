import argparse
import logging

from .config import (
    DEFAULT_DIARIZATION_METHOD,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_SPEAKER_EMBEDDING_DEVICE,
    DEFAULT_SPEAKER_EMBEDDING_MODEL,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WORD_PROB_THRESHOLD,
    DEFAULT_WORD_SMOOTH_MIN_WORDS,
    LOG_FORMAT,
)
from .pipeline import ProcessingOptions, process_audio_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Транскрибировать аудио, определить спикеров и собрать DOCX."
    )
    parser.add_argument("--audio", required=True, help="Путь к аудиофайлу")
    parser.add_argument(
        "--output-docx",
        default=None,
        help="Путь к выходному DOCX (по умолчанию: <audio>.docx)",
    )
    parser.add_argument(
        "--transcript-dir",
        default="transcripts",
        help="Каталог для сохранения транскриптов и диаризации",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Путь или HF-репозиторий модели Whisper-MLX",
    )
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help="Путь или HF-репозиторий модели pyannote MLX",
    )
    parser.add_argument(
        "--speaker-map",
        default=None,
        help="Связать спикеров и роли: SPEAKER_00=К,SPEAKER_01=Т",
    )
    parser.add_argument("--fio", default="", help="ФИО")
    parser.add_argument("--group", default="", help="Номер группы")
    parser.add_argument("--date", default="", help="Дата")
    parser.add_argument("--topic", default="", help="Тема протокола")
    parser.add_argument("--task", default="", help="Задание")
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=0.5,
        help="Минимальная длительность сегмента (сек)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.35,
        help="Порог уверенности для тишины (0 = отключить)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=2,
        help="Число спикеров для кластеризации",
    )
    parser.add_argument(
        "--speaker-embedding-model",
        default=DEFAULT_SPEAKER_EMBEDDING_MODEL,
        help="Модель спикер-эмбеддингов (SpeechBrain)",
    )
    parser.add_argument(
        "--speaker-embedding-device",
        default=DEFAULT_SPEAKER_EMBEDDING_DEVICE,
        help="Устройство для спикер-эмбеддингов (cpu/mps/cuda)",
    )
    parser.add_argument(
        "--word-smooth-min-words",
        type=int,
        default=DEFAULT_WORD_SMOOTH_MIN_WORDS,
        help="Сглаживание word-диаризации: минимум слов в вставке",
    )
    parser.add_argument(
        "--word-prob-threshold",
        type=float,
        default=DEFAULT_WORD_PROB_THRESHOLD,
        help="Порог вероятности слова (0 = не фильтровать)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.2,
        help="Сшивать соседние сегменты одного спикера (сек)",
    )
    parser.add_argument(
        "--sandwich-max-duration",
        type=float,
        default=1.0,
        help="Макс. длительность «вставки» между одинаковыми спикерами (сек)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=160000,
        help="Размер чанка для диаризации (семплы, 10с при 16кГц)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=16000,
        help="Перекрытие чанков (семплы, 1с при 16кГц)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Уровень логирования (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        '--force-whisper',
        action='store_true',
        help='Перезапустить Whisper и обновить кэш',
    )
    parser.add_argument(
        '--force-diarization',
        action='store_true',
        help='Перезапустить диаризацию и обновить кэш',
    )
    parser.add_argument(
        '--no-word-timestamps',
        action='store_false',
        dest='word_timestamps',
        default=True,
        help='Отключить word timestamps в Whisper',
    )
    parser.add_argument(
        '--diarization-method',
        default=DEFAULT_DIARIZATION_METHOD,
        choices=['custom_mlx', 'pyannote_pipeline'],
        help='Метод диаризации (по умолчанию: custom_mlx)',
    )
    parser.add_argument(
        '--clustering-method',
        default='kmeans',
        choices=['kmeans', 'spectral', 'agglomerative'],
        help='Метод кластеризации спикеров',
    )
    parser.add_argument(
        '--swap',
        action='store_true',
        help='Поменять К↔Т (shortcut для --speaker-map SPEAKER_00=Т,SPEAKER_01=К)',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format=LOG_FORMAT)
    speaker_map = args.speaker_map
    if args.swap and not speaker_map:
        speaker_map = 'SPEAKER_00=Т,SPEAKER_01=К'
    options = ProcessingOptions(
        output_docx=args.output_docx,
        transcript_dir=args.transcript_dir,
        whisper_model=args.whisper_model,
        diarization_model=args.diarization_model,
        speaker_map=speaker_map,
        fio=args.fio,
        group=args.group,
        date=args.date,
        topic=args.topic,
        task=args.task,
        min_segment_duration=args.min_segment_duration,
        silence_threshold=args.silence_threshold,
        max_speakers=args.max_speakers,
        speaker_embedding_model=args.speaker_embedding_model,
        speaker_embedding_device=args.speaker_embedding_device,
        word_smooth_min_words=args.word_smooth_min_words,
        word_prob_threshold=args.word_prob_threshold,
        merge_gap=args.merge_gap,
        sandwich_max_duration=args.sandwich_max_duration,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        force_whisper=args.force_whisper,
        force_diarization=args.force_diarization,
        word_timestamps=args.word_timestamps,
        diarization_method=args.diarization_method,
        clustering_method=args.clustering_method,
    )
    process_audio_file(args.audio, options)
