#!/usr/bin/env python3
import asyncio
import copy
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import SimpleFilesPathWrapper, TelegramAPIServer
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from psy_protocol.config import LOG_FORMAT
from psy_protocol.pipeline import ProcessingOptions, process_audio_file


TEMP_ROOT = Path("transcripts/telegram_temp")
SUPPORTED_AUDIO_MIME_PREFIX = "audio/"
SESSION_TTL_SECONDS = 3600  # 1 hour
WHISPER_LARGE_V3 = "~/.cache/mlx/large-v3"

PRESETS: Dict[str, Dict[str, Any]] = {
    "large_model": {
        "label": "🎙 Точнее (large-v3)",
        "whisper_model": WHISPER_LARGE_V3,
        "force_whisper": True,
    },
    "noisy": {
        "label": "🔇 Плохой звук",
        "silence_threshold": 0.25,
        "merge_gap": 0.5,
        "sandwich_max_duration": 2.0,
        "word_prob_threshold": 0.15,
    },
    "interrupts": {
        "label": "🗣 Много перебиваний",
        "merge_gap": 1.0,
        "sandwich_max_duration": 3.0,
        "word_prob_threshold": 0.1,
        "word_smooth_min_words": 3,
    },
    "sentences": {
        "label": "📝 По фразам",
    },
    "swap": {
        "label": "🔄 Поменять К↔Т",
        "speaker_map": "SPEAKER_00=Т,SPEAKER_01=К",
        "force_diarization": False,
    },
    "llm_diarize": {
        "label": "🧠 LLM-диаризация",
        "diarization_method": "llm",
        "force_diarization": True,
    },
    "raw_text": {
        "label": "📄 Сырой текст",
    },
    "timed": {
        "label": "⏱ С таймкодами",
    },
}

CONSENT_TEXT = r"""📋 *Пользовательское соглашение*

Перед использованием бота ознакомьтесь с условиями обработки данных\.

*Что делает бот:*
Принимает аудиозаписи, распознаёт речь и формирует текстовый протокол\. Вся обработка выполняется локально, без передачи аудио сторонним облачным сервисам\.

*Ваши данные:*
• Аудиофайл временно сохраняется для обработки и удаляется по истечении сессии \(1 час\)\.
• Результаты \(TXT, DOCX\) хранятся в течение сессии и удаляются вместе с аудио\.
• Данные не передаются и не продаются третьим лицам\.

*Ответственность:*
Автор бота принимает разумные технические меры для защиты данных, однако не несёт ответственности за ущерб вследствие обстоятельств вне его контроля \(взлом серверов, утечки на стороне инфраструктуры и т\.п\.\)\.

*Ваши обязательства:*
Отправляя аудио, вы подтверждаете, что имеете законное право на передачу данной записи и несёте ответственность за правомерность её использования\.

Нажмите «✅ Принять», чтобы продолжить\."""


@dataclass
class JobSession:
    work_dir: Path
    audio_path: Path
    base_options: Any  # ProcessingOptions — imported below
    expires_at: float  # time.monotonic() + TTL


# Keyed by chat_id (int)
job_sessions: Dict[int, "JobSession"] = {}

CONSENTS_FILE = Path("consents/accepted.txt")
consented_users: set[int] = set()


def load_consents() -> None:
    """Load persisted chat_ids into consented_users set."""
    if CONSENTS_FILE.exists():
        for line in CONSENTS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
                consented_users.add(int(line))
    logging.info("Consents loaded: %d users", len(consented_users))


def save_consent(chat_id: int) -> None:
    """Persist a new consent by appending chat_id to file."""
    CONSENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CONSENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{chat_id}\n")


@dataclass
class TelegramSettings:
    token: str
    api_base_url: Optional[str] = None
    local_server_file_root: str = "/var/lib/telegram-bot-api"
    local_host_file_root: str = "./telegram-bot-api-data"
    api_is_local: bool = True
    whisper_model: Optional[str] = None
    diarization_model: Optional[str] = None
    max_speakers: Optional[int] = None


def parse_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_settings() -> TelegramSettings:
    env_file_values = parse_env_file(Path(".env"))
    env = {**env_file_values, **os.environ}
    token = env.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment or .env file")
    max_speakers = env.get("PSY_MAX_SPEAKERS")
    api_is_local_raw = env.get("TELEGRAM_API_IS_LOCAL", "true").lower()
    api_is_local = api_is_local_raw not in ("false", "0", "no")
    return TelegramSettings(
        token=token,
        api_base_url=env.get("TELEGRAM_BOT_API_BASE_URL"),
        local_server_file_root=env.get(
            "TELEGRAM_LOCAL_SERVER_FILE_ROOT", "/var/lib/telegram-bot-api"
        ),
        local_host_file_root=env.get(
            "TELEGRAM_LOCAL_HOST_FILE_ROOT", "./telegram-bot-api-data"
        ),
        api_is_local=api_is_local,
        whisper_model=env.get("PSY_WHISPER_MODEL"),
        diarization_model=env.get("PSY_DIARIZATION_MODEL"),
        max_speakers=int(max_speakers) if max_speakers else None,
    )


def create_bot(settings: TelegramSettings) -> Bot:
    if not settings.api_base_url:
        return Bot(token=settings.token)
    if settings.api_is_local:
        local_host_root = Path(settings.local_host_file_root).expanduser().resolve()
        api_server = TelegramAPIServer.from_base(
            settings.api_base_url,
            is_local=True,
            wrap_local_file=SimpleFilesPathWrapper(
                server_path=Path(settings.local_server_file_root),
                local_path=local_host_root,
            ),
        )
        logging.info(
            "Using local Telegram Bot API at %s (server_root=%s, host_root=%s)",
            settings.api_base_url,
            settings.local_server_file_root,
            local_host_root,
        )
    else:
        api_server = TelegramAPIServer.from_base(
            settings.api_base_url,
            is_local=False,
        )
        logging.info(
            "Using remote Telegram Bot API at %s (files downloaded via HTTP)",
            settings.api_base_url,
        )
    session = AiohttpSession(api=api_server)
    return Bot(token=settings.token, session=session)


def build_processing_options(
    settings: TelegramSettings, output_docx: Path, cache_dir: Path,
) -> ProcessingOptions:
    options = ProcessingOptions(
        output_docx=output_docx,
        transcript_dir=cache_dir,
        diarization_method='custom_mlx',
    )
    if settings.whisper_model:
        options.whisper_model = settings.whisper_model
    if settings.diarization_model:
        options.diarization_model = settings.diarization_model
    if settings.max_speakers is not None:
        options.max_speakers = settings.max_speakers
    return options


def apply_preset(base_opts: Any, preset_key: str) -> Any:
    opts = copy.copy(base_opts)
    overrides = {k: v for k, v in PRESETS[preset_key].items() if k != "label"}
    for key, val in overrides.items():
        setattr(opts, key, val)
    return opts


def build_consent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="✅ Принять", callback_data="consent:accept"),
        ]]
    )


def build_retry_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=PRESETS['large_model']['label'],
                    callback_data='retry:large_model',
                ),
                InlineKeyboardButton(
                    text=PRESETS['noisy']['label'],
                    callback_data='retry:noisy',
                ),
            ],
            [
                InlineKeyboardButton(
                    text=PRESETS['interrupts']['label'],
                    callback_data='retry:interrupts',
                ),
                InlineKeyboardButton(
                    text=PRESETS['swap']['label'],
                    callback_data='retry:swap',
                ),
            ],
            [
                InlineKeyboardButton(
                    text=PRESETS['sentences']['label'],
                    callback_data='retry:sentences',
                ),
                InlineKeyboardButton(
                    text=PRESETS['timed']['label'],
                    callback_data='retry:timed',
                ),
            ],
            [
                InlineKeyboardButton(
                    text=PRESETS['llm_diarize']['label'],
                    callback_data='retry:llm_diarize',
                ),
                InlineKeyboardButton(
                    text=PRESETS['raw_text']['label'],
                    callback_data='retry:raw_text',
                ),
            ],
        ]
    )


def ensure_temp_root() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def build_work_paths(message: Message, suffix: str) -> Tuple[Path, Path, Path]:
    chat_id = message.chat.id if message.chat else 0
    message_id = message.message_id
    work_dir = TEMP_ROOT / f"{chat_id}_{message_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / f"input{suffix}"
    output_docx = work_dir / "result.docx"
    cache_dir = work_dir / "cache"
    return work_dir, audio_path, output_docx


async def download_audio(
    message: Message, bot: Bot, settings: TelegramSettings,
) -> Optional[Tuple[Path, Path, Path]]:
    file_id: Optional[str] = None
    suffix = ".audio"

    if message.voice:
        file_id = message.voice.file_id
        suffix = ".ogg"
    elif message.audio:
        file_id = message.audio.file_id
        if message.audio.file_name:
            suffix = Path(message.audio.file_name).suffix or ".audio"
    elif message.document:
        mime_type = message.document.mime_type or ""
        if not mime_type.startswith(SUPPORTED_AUDIO_MIME_PREFIX):
            return None
        file_id = message.document.file_id
        if message.document.file_name:
            suffix = Path(message.document.file_name).suffix or ".audio"

    if not file_id:
        return None

    work_dir, audio_path, output_docx = build_work_paths(message, suffix)
    tg_file = await bot.get_file(file_id)
    file_path = tg_file.file_path or ""
    await bot.download_file(file_path, destination=audio_path)
    return work_dir, audio_path, output_docx


def cleanup_work_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def build_bar(percent: float) -> str:
    clamped = max(0.0, min(100.0, percent))
    filled = int(clamped // 10)
    return f"{'█' * filled}{'░' * (10 - filled)}"


def stage_label(stage: str) -> str:
    labels = {
        "start": "Старт",
        "prepare": "Подготовка",
        "whisper": "Распознавание (Whisper)",
        "diarization": "Диаризация",
        "replicas": "Форматирование",
        "output": "Формирование файлов",
        "done": "Готово",
    }
    return labels.get(stage, stage.title())


def stage_hint(stage: str) -> str:
    hints = {
        "start": "Запускаю обработку.",
        "prepare": "Подготавливаю файлы и кэш.",
        "whisper": "Распознаю речь (Whisper).",
        "diarization": "Определяю спикеров.",
        "replicas": "Собираю реплики диалога.",
        "output": "Формирую итоговые документы.",
        "done": "Обработка завершена.",
    }
    return hints.get(stage, "Обрабатываю аудио.")


def render_progress_text(progress: Dict[str, Any]) -> str:
    if progress.get("done"):
        if progress.get("success"):
            return "✅ Готово! Отправляю файлы."
        return "❌ Не удалось обработать аудио."

    stage = progress.get("stage", "start")
    percent = progress.get("percent")
    value = float(percent) if percent is not None else 0.0
    bar = build_bar(value)
    percent_text = f"{int(value)}%"

    return (
        "⏳ Идёт обработка аудио\n"
        f"Этап: {stage_label(stage)}\n"
        f"[{bar}] {percent_text}\n"
        f"{stage_hint(stage)}"
    )


async def progress_updater(
    status_message: Message, progress: Dict[str, Any], interval_seconds: int = 5,
) -> None:
    last_text = ""
    while not progress.get("done"):
        text = render_progress_text(progress)
        if text != last_text:
            try:
                await status_message.edit_text(text)
                last_text = text
            except Exception:
                logging.debug("Status message edit skipped", exc_info=True)
        await asyncio.sleep(interval_seconds)

    final_text = render_progress_text(progress)
    if final_text != last_text:
        try:
            await status_message.edit_text(final_text)
        except Exception:
            logging.debug("Final status message edit skipped", exc_info=True)


async def run_pipeline_and_send(
    chat_id: int,
    audio_path: Path,
    opts: Any,
    status_message: Message,
    reply_target: Message,
    progress: Dict[str, Any],
) -> bool:
    """Run the pipeline and send result files. Returns True on success."""
    updater_task = asyncio.create_task(
        progress_updater(status_message, progress, interval_seconds=5)
    )
    try:
        docx_path, txt_path = await asyncio.to_thread(
            process_audio_file, audio_path, opts, lambda s, p, m: _update_progress(progress, s, p, m, chat_id)
        )
        progress["done"] = True
        progress["success"] = True
        await updater_task
        await reply_target.answer_document(FSInputFile(path=str(txt_path)))
        await reply_target.answer_document(FSInputFile(path=str(docx_path)))
        await reply_target.answer(
            "Если результат неточный — попробуйте один из вариантов:"
            "- Точнее -- использовать другую модель в 3-5 раз медленее"
            "- Плохой звук -- изменить параметры ",
            reply_markup=build_retry_keyboard(),
        )
        return True
    except Exception:
        progress["done"] = True
        progress["success"] = False
        logging.exception("Pipeline failed for chat_id=%s", chat_id)
        try:
            await updater_task
        except Exception:
            logging.debug("Updater task finished with error", exc_info=True)
        return False
    finally:
        if not updater_task.done():
            updater_task.cancel()


def _update_progress(
    progress: Dict[str, Any],
    stage: str,
    percent: Optional[float],
    status_text: str,
    chat_id: int,
) -> None:
    if progress.get("stage") != stage:
        progress["stage_started_at"] = time.monotonic()
    progress["stage"] = stage
    progress["percent"] = percent
    progress["message"] = status_text
    if stage == "whisper" and percent is not None:
        rounded = int(float(percent))
        last_logged = int(progress.get("last_logged_whisper_percent", -1))
        if rounded >= last_logged + 5:
            logging.info(
                "Telegram chat_id=%s whisper_progress=%d%%",
                chat_id,
                rounded,
            )
            progress["last_logged_whisper_percent"] = rounded


def _make_progress() -> Dict[str, Any]:
    now = time.monotonic()
    return {
        "done": False,
        "success": False,
        "stage": "start",
        "percent": 0.0,
        "message": "Queued",
        "started_at": now,
        "stage_started_at": now,
        "last_logged_whisper_percent": -1,
    }


async def process_and_reply(message: Message, bot: Bot, settings: TelegramSettings) -> None:
    chat_id = message.chat.id if message.chat else 0
    if chat_id not in consented_users:
        await message.answer(
            CONSENT_TEXT,
            parse_mode="MarkdownV2",
            reply_markup=build_consent_keyboard(),
        )
        return

    download_result = await download_audio(message, bot, settings)
    if not download_result:
        await message.answer(
            "Пожалуйста, отправьте голосовое сообщение, аудио или аудиофайл документом 🙏"
        )
        return

    work_dir, audio_path, output_docx = download_result
    status_message = await message.answer(
        "Спасибо! 😊 Аудио получено, начинаю обработку. "
        "Пожалуйста, немного подождите ⏳"
    )
    chat_id = message.chat.id if message.chat else 0
    options = build_processing_options(settings, output_docx=output_docx, cache_dir=work_dir / "cache")
    progress = _make_progress()

    success = await run_pipeline_and_send(
        chat_id=chat_id,
        audio_path=audio_path,
        opts=options,
        status_message=status_message,
        reply_target=message,
        progress=progress,
    )

    if success:
        job_sessions[chat_id] = JobSession(
            work_dir=work_dir,
            audio_path=audio_path,
            base_options=options,
            expires_at=time.monotonic() + SESSION_TTL_SECONDS,
        )
        logging.info("Session stored for chat_id=%s", chat_id)
    else:
        logging.error("Failed to process audio from Telegram for chat_id=%s", chat_id)
        await message.answer(
            "Извините, не получилось обработать это аудио 😔 "
            "Пожалуйста, попробуйте другой файл."
        )
        cleanup_work_dir(work_dir)


async def handle_retry_callback(
    callback: CallbackQuery, bot: Bot, settings: TelegramSettings
) -> None:
    await callback.answer()
    preset_key = (callback.data or "").split(":", 1)[1]
    chat_id = callback.message.chat.id if callback.message and callback.message.chat else 0

    session = job_sessions.get(chat_id)
    if not session or time.monotonic() > session.expires_at:
        if callback.message:
            await callback.message.answer("Сессия истекла, отправьте аудио заново.")
        return

    # Refresh TTL
    session.expires_at = time.monotonic() + SESSION_TTL_SECONDS

    if preset_key in ('sentences', 'raw_text', 'timed'):
        transcript_dir = Path(session.base_options.transcript_dir) / session.audio_path.stem
        file_map = {
            'sentences': transcript_dir / 'sentences.txt',
            'raw_text': transcript_dir / 'transcript.txt',
            'timed': transcript_dir / 'timed_dialogue.txt',
        }
        file_path = file_map[preset_key]
        if file_path.exists():
            await callback.message.answer_document(FSInputFile(path=str(file_path)))
        else:
            await callback.message.answer('Файл не найден, отправьте аудио заново.')
        return

    opts = apply_preset(session.base_options, preset_key)
    # Reuse same output_docx path (overwrites previous result)
    opts.output_docx = session.base_options.output_docx

    status_message = await callback.message.answer(
        f"⏳ Повторная обработка ({PRESETS[preset_key]['label']})…"
    )
    progress = _make_progress()

    success = await run_pipeline_and_send(
        chat_id=chat_id,
        audio_path=session.audio_path,
        opts=opts,
        status_message=status_message,
        reply_target=callback.message,
        progress=progress,
    )

    if not success:
        logging.error("Failed to retry audio processing for chat_id=%s preset=%s", chat_id, preset_key)
        await callback.message.answer(
            "Извините, не получилось обработать аудио при повторной попытке 😔"
        )


async def handle_consent_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    chat_id = callback.message.chat.id if callback.message and callback.message.chat else 0
    if chat_id not in consented_users:
        consented_users.add(chat_id)
        save_consent(chat_id)
        logging.info("Consent accepted for chat_id=%s", chat_id)
    if callback.message:
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer(
            "✅ Соглашение принято. Отправьте голосовое сообщение или аудиофайл 📄"
        )


async def session_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(600)  # 10 min
        now = time.monotonic()
        expired = [cid for cid, s in list(job_sessions.items()) if now > s.expires_at]
        for cid in expired:
            session = job_sessions.pop(cid, None)
            if session:
                logging.info("Cleaning up expired session for chat_id=%s", cid)
                cleanup_work_dir(session.work_dir)


def create_dispatcher(settings: TelegramSettings) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def handle_start(message: Message) -> None:
        chat_id = message.chat.id if message.chat else 0
        if chat_id in consented_users:
            await message.answer(
                "Здравствуйте! 👋 Вы уже приняли соглашение.\n"
                "Отправьте голосовое сообщение или аудиофайл 📄"
            )
        else:
            await message.answer(
                CONSENT_TEXT,
                parse_mode="MarkdownV2",
                reply_markup=build_consent_keyboard(),
            )

    @dp.callback_query(F.data == "consent:accept")
    async def handle_consent(callback: CallbackQuery) -> None:
        await handle_consent_callback(callback)

    @dp.message(F.voice)
    async def handle_voice(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    @dp.message(F.audio)
    async def handle_audio(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    @dp.message(F.document)
    async def handle_document(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    @dp.callback_query(F.data.startswith("retry:"))
    async def handle_retry(callback: CallbackQuery, bot: Bot) -> None:
        await handle_retry_callback(callback, bot, settings)

    return dp


async def run_bot() -> None:
    settings = load_settings()
    ensure_temp_root()
    load_consents()
    logging.info("Starting Telegram bot polling")
    bot = create_bot(settings)
    dp = create_dispatcher(settings)
    cleanup_task = asyncio.create_task(session_cleanup_loop())
    try:
        await dp.start_polling(bot)
    finally:
        cleanup_task.cancel()
        await bot.session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
