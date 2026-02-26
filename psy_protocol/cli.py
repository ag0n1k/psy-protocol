import argparse
import logging
from pathlib import Path

from .alignment import build_replicas, build_replicas_from_words
from .config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_EMBEDDING_MIN_DURATION,
    DEFAULT_SPEAKER_EMBEDDING_DEVICE,
    DEFAULT_SPEAKER_EMBEDDING_MODEL,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WORD_PROB_THRESHOLD,
    DEFAULT_WORD_SMOOTH_MIN_WORDS,
    LOG_FORMAT,
)
from .diarization import (
    SpeakerSegment,
    cluster_segments_by_embeddings,
    diarize_audio_raw,
    load_audio,
    post_process_diarization,
)
from .docx_writer import create_docx
from .io_utils import load_json, save_json, save_text
from .roles import map_speakers_to_roles, parse_speaker_map
from .text_outputs import save_dialogue_txt
from .whisper_transcribe import extract_words, transcribe_audio


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
        "--force-whisper",
        action="store_true",
        help="Перезапустить Whisper и обновить кэш",
    )
    parser.add_argument(
        "--no-word-timestamps",
        action="store_false",
        dest="word_timestamps",
        default=True,
        help="Отключить word timestamps в Whisper",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format=LOG_FORMAT)

    logging.info("Старт обработки")
    logging.info("Аудио: %s", args.audio)
    logging.info("Модель Whisper: %s", args.whisper_model)
    logging.info("Модель диаризации: %s", args.diarization_model)
    logging.info(
        "Параметры диаризации: min_duration=%.2f, silence_threshold=%.2f, chunk=%d, overlap=%d",
        args.min_segment_duration,
        args.silence_threshold,
        args.chunk_size,
        args.overlap,
    )
    logging.info("Кластеризация: num_speakers=%d", args.max_speakers)
    logging.info(
        "Эмбеддинги: model=%s device=%s",
        args.speaker_embedding_model,
        args.speaker_embedding_device,
    )
    logging.info("Word-smoothing: min_words=%d", args.word_smooth_min_words)
    logging.info("Word-filter: prob_threshold=%.2f", args.word_prob_threshold)
    logging.info(
        "Постобработка: merge_gap=%.2f, sandwich_max_duration=%.2f",
        args.merge_gap,
        args.sandwich_max_duration,
    )

    audio_path = Path(args.audio)
    output_docx = (
        Path(args.output_docx) if args.output_docx else audio_path.with_suffix(".docx")
    )

    transcript_root = Path(args.transcript_dir).expanduser()
    transcript_dir = transcript_root / audio_path.stem
    transcript_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Кэш: %s", transcript_dir)

    transcript_json_path = transcript_dir / "transcript.json"
    transcript_txt_path = transcript_dir / "transcript.txt"
    whisper_segments_path = transcript_dir / "whisper_segments.json"

    if transcript_json_path.exists() and not args.force_whisper:
        logging.info("Whisper: загрузка из кэша %s", transcript_json_path)
        whisper_result = load_json(transcript_json_path)
    else:
        logging.info("Whisper: запуск распознавания (word_timestamps=%s)", args.word_timestamps)
        whisper_result = transcribe_audio(
            str(audio_path),
            args.whisper_model,
            word_timestamps=args.word_timestamps,
        )
        save_json(transcript_json_path, whisper_result)
        logging.info("Whisper: сохранен кэш %s", transcript_json_path)

    if args.force_whisper or not transcript_txt_path.exists():
        save_text(transcript_txt_path, whisper_result.get("text", ""))
        logging.info("Whisper: сохранен текст %s", transcript_txt_path)

    if args.force_whisper or not whisper_segments_path.exists():
        whisper_segments = whisper_result.get("segments", [])
        save_json(whisper_segments_path, {"segments": whisper_segments})
        logging.info("Whisper: сегменты сохранены %s", whisper_segments_path)
    else:
        logging.info("Whisper: сегменты из кэша %s", whisper_segments_path)
        whisper_segments_payload = load_json(whisper_segments_path)
        whisper_segments = whisper_segments_payload.get("segments", [])

    diarization_post_path = transcript_dir / "diarization_post.json"
    diarization_payload = None
    if diarization_post_path.exists():
        diarization_payload = load_json(diarization_post_path)
        params = diarization_payload.get("params", {})
        if (
            diarization_payload.get("method") != "embedding_kmeans_v2"
            or params.get("num_speakers") != args.max_speakers
            or params.get("merge_gap") != args.merge_gap
            or params.get("sandwich_max_duration") != args.sandwich_max_duration
            or params.get("silence_threshold") != args.silence_threshold
            or params.get("min_segment_duration") != args.min_segment_duration
            or params.get("embedding_min_duration") != DEFAULT_EMBEDDING_MIN_DURATION
            or params.get("embedding_model") != args.speaker_embedding_model
            or params.get("embedding_device") != args.speaker_embedding_device
        ):
            diarization_payload = None

    if diarization_payload:
        logging.info("Диаризация: загрузка из кэша %s", diarization_post_path)
        diarization_segments = [
            SpeakerSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=str(seg["speaker"]),
            )
            for seg in diarization_payload.get("segments", [])
        ]
    else:
        diarization_raw_path = transcript_dir / "diarization.json"
        if diarization_raw_path.exists():
            logging.info("Диаризация: сырые сегменты из кэша %s", diarization_raw_path)
            diarization_raw_payload = load_json(diarization_raw_path)
            raw_segments = [
                SpeakerSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    speaker=str(seg.get("speaker", "RAW_00")),
                )
                for seg in diarization_raw_payload.get("segments", [])
            ]
        else:
            logging.info("Диаризация: запуск сегментации")
            raw_segments = diarize_audio_raw(
                str(audio_path),
                args.diarization_model,
                min_duration=args.min_segment_duration,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                silence_threshold=args.silence_threshold,
            )
            save_json(
                diarization_raw_path,
                {
                    "segments": [
                        {"start": s.start, "end": s.end, "speaker": s.speaker}
                        for s in raw_segments
                    ]
                },
            )
            logging.info("Диаризация: сырые сегменты сохранены %s", diarization_raw_path)

        waveform, sample_rate = load_audio(str(audio_path))
        logging.info("Диаризация: сырых сегментов %d", len(raw_segments))
        diarization_segments = cluster_segments_by_embeddings(
            waveform,
            sample_rate,
            raw_segments,
            num_speakers=args.max_speakers,
            embedding_model_id=args.speaker_embedding_model,
            embedding_device=args.speaker_embedding_device,
        )
        logging.info("Диаризация: сегментов после кластеризации %d", len(diarization_segments))
        diarization_segments = post_process_diarization(
            diarization_segments,
            merge_gap=args.merge_gap,
            sandwich_max_duration=args.sandwich_max_duration,
        )
        logging.info("Диаризация: сегментов после постобработки %d", len(diarization_segments))
        save_json(
            diarization_post_path,
            {
                "method": "embedding_kmeans_v2",
                "segments": [
                    {"start": s.start, "end": s.end, "speaker": s.speaker}
                    for s in diarization_segments
                ],
                "params": {
                    "num_speakers": args.max_speakers,
                    "merge_gap": args.merge_gap,
                    "sandwich_max_duration": args.sandwich_max_duration,
                    "silence_threshold": args.silence_threshold,
                    "min_segment_duration": args.min_segment_duration,
                    "embedding_min_duration": DEFAULT_EMBEDDING_MIN_DURATION,
                    "embedding_model": args.speaker_embedding_model,
                    "embedding_device": args.speaker_embedding_device,
                },
            },
        )
        logging.info("Диаризация: сохранена постобработка %s", diarization_post_path)

    words = (
        extract_words(whisper_result, prob_threshold=args.word_prob_threshold)
        if args.word_timestamps
        else []
    )
    if args.word_timestamps and words:
        logging.info("Whisper: word timestamps %d", len(words))
        replicas = build_replicas_from_words(
            words,
            diarization_segments,
            smooth_min_words=args.word_smooth_min_words,
        )
    else:
        if args.word_timestamps:
            logging.warning("Whisper: word timestamps отсутствуют, использую сегменты")
        replicas = build_replicas(whisper_segments, diarization_segments)

    logging.info("Реплики: %d", len(replicas))
    explicit_map = parse_speaker_map(args.speaker_map)
    role_map = map_speakers_to_roles(replicas, explicit_map)
    for replica in replicas:
        replica["role"] = role_map.get(replica["speaker"], "К")

    metadata = {
        "ФИО": args.fio,
        "Номер группы": args.group,
        "Дата": args.date,
        "Тема протокола": args.topic,
        "Задание": args.task,
    }

    logging.info("Генерация DOCX: %s", output_docx)
    create_docx(
        output_path=str(output_docx),
        replicas=replicas,
        metadata=metadata,
    )
    dialogue_txt_path = output_docx.with_suffix(".txt")
    save_dialogue_txt(dialogue_txt_path, replicas)
    logging.info("Диалог TXT: %s", dialogue_txt_path)
    logging.info("Готово")
