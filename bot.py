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

PRESETS: Dict[str, Dict[str, Any]] = {
    "other_approach": {
        "label": "🎙 Попробовать иначе",
        "transcription_method": "whisper",
    },
    "swap": {
        "label": "🔄 Поменять К↔Т",
        "speaker_map": "SPEAKER_00=Т,SPEAKER_01=К",
        "force_diarization": False,
    },
    "raw_text": {
        "label": "📄 Сырой текст",
    },
    "timed": {
        "label": "⏱ С таймкодами",
    },
}

CONSENT_TEXT = """📋 <b>Пользовательское соглашение</b>

Перед использованием бота ознакомьтесь с условиями обработки данных.

<b>Что делает бот:</b>
Принимает аудиозаписи, распознаёт речь и формирует текстовый протокол. Вся обработка выполняется локально, без передачи аудио сторонним облачным сервисам.
Этот бот не отменяет прослушивания, валидации, дополнений, исправлений, разбора сессии.
Цель бота - убрать большую часть рутинных операций с текстом.
Этот бот не гарантирует хорошего качества, он может быть совсем неточен на плохих записях (~5-20% точности),
однако на хороших качество на выборке доходило до 75%.

<b>Ваши данные:</b>
• Аудиофайл временно сохраняется для обработки и удаляется по истечении сессии (1 час).
• Результаты (TXT, DOCX) хранятся в течение сессии и удаляются вместе с аудио.
• Данные не передаются и не продаются третьим лицам.

<b>Ответственность:</b>
Автор бота принимает разумные технические меры для защиты данных, однако не несёт ответственности за ущерб вследствие обстоятельств вне его контроля (взлом серверов, утечки на стороне инфраструктуры и т.п.).

<b>Ваши обязательства:</b>
Отправляя аудио, вы подтверждаете, что имеете законное право на передачу данной записи и несёте ответственность за правомерность её использования.
Также вы подтверждаете, что полностью прочитали данное соглашение и согласны с ним.

Нажмите «✅ Принять», чтобы продолжить."""


@dataclass
class JobSession:
    work_dir: Path
    audio_path: Path
    base_options: Any  # ProcessingOptions — imported below


# Keyed by chat_id (int)
job_sessions: Dict[int, "JobSession"] = {}
processing_chats: set[int] = set()
PIPELINE_SEMAPHORE = asyncio.Semaphore(1)

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
                    text=PRESETS['other_approach']['label'],
                    callback_data='retry:other_approach',
                ),
                InlineKeyboardButton(
                    text=PRESETS['swap']['label'],
                    callback_data='retry:swap',
                ),
            ],
            [
                InlineKeyboardButton(
                    text=PRESETS['timed']['label'],
                    callback_data='retry:timed',
                ),
                InlineKeyboardButton(
                    text=PRESETS['raw_text']['label'],
                    callback_data='retry:raw_text',
                ),
            ],
            [
                InlineKeyboardButton(
                    text='✅ Завершить обработку',
                    callback_data='session:finish',
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
        "queue": "Ожидание в очереди",
        "start": "Идёт обработка аудио",
        "prepare": "Идёт обработка аудио",
        "whisper": "Идёт обработка аудио",
        "diarization": "Идёт обработка аудио",
        "replicas": "Идёт обработка аудио",
        "output": "Идёт обработка аудио",
        "done": "Готово",
    }
    return labels.get(stage, stage.title())


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
        f"⏳ {stage_label(stage)}\n"
        f"[{bar}] {percent_text}"
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
        if PIPELINE_SEMAPHORE.locked():
            _update_progress(progress, "queue", 0.0, "Waiting in processing queue", chat_id)

        async with PIPELINE_SEMAPHORE:
            docx_path, txt_path = await asyncio.to_thread(
                process_audio_file, audio_path, opts, lambda s, p, m: _update_progress(progress, s, p, m, chat_id)
            )
        progress["done"] = True
        progress["success"] = True
        await updater_task
        await reply_target.answer_document(FSInputFile(path=str(txt_path)))
        await reply_target.answer_document(FSInputFile(path=str(docx_path)))
        await reply_target.answer(
            "Если результат неточный — выберите один из вариантов на кнопках ниже.",
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
            parse_mode="HTML",
            reply_markup=build_consent_keyboard(),
        )
        return

    if chat_id in processing_chats:
        await message.answer(
            "Файл уже обрабатывается. Дождитесь завершения или нажмите «✅ Завершить обработку».",
        )
        return

    active_session = job_sessions.get(chat_id)
    if active_session:
        if not active_session.audio_path.exists():
            cleanup_work_dir(active_session.work_dir)
            job_sessions.pop(chat_id, None)
        else:
            await message.answer(
                "Сейчас уже есть активная обработка этого файла. "
                "Нажмите «✅ Завершить обработку», чтобы загрузить новый.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[[
                        InlineKeyboardButton(
                            text='✅ Завершить обработку',
                            callback_data='session:finish',
                        ),
                    ]]
                ),
            )
            return

    processing_chats.add(chat_id)
    try:
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
            )
            logging.info("Active file session stored for chat_id=%s", chat_id)
        else:
            logging.error("Failed to process audio from Telegram for chat_id=%s", chat_id)
            await message.answer(
                "Извините, не получилось обработать это аудио 😔 "
                "Пожалуйста, попробуйте другой файл."
            )
            cleanup_work_dir(work_dir)
    finally:
        processing_chats.discard(chat_id)


async def handle_retry_callback(
    callback: CallbackQuery, bot: Bot, settings: TelegramSettings
) -> None:
    await callback.answer()
    preset_key = (callback.data or "").split(":", 1)[1]
    chat_id = callback.message.chat.id if callback.message and callback.message.chat else 0

    if chat_id in processing_chats:
        if callback.message:
            await callback.message.answer('Обработка уже выполняется. Дождитесь завершения.')
        return

    session = job_sessions.get(chat_id)
    if not session:
        if callback.message:
            await callback.message.answer("Активный файл не найден, отправьте аудио заново.")
        return
    if not session.audio_path.exists():
        if callback.message:
            await callback.message.answer('Исходный аудиофайл недоступен, отправьте аудио заново.')
        cleanup_work_dir(session.work_dir)
        job_sessions.pop(chat_id, None)
        return

    if preset_key in ('raw_text', 'timed'):
        transcript_dir = Path(session.base_options.transcript_dir) / session.audio_path.stem
        file_map = {
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
    processing_chats.add(chat_id)
    try:
        success = await run_pipeline_and_send(
            chat_id=chat_id,
            audio_path=session.audio_path,
            opts=opts,
            status_message=status_message,
            reply_target=callback.message,
            progress=progress,
        )
    finally:
        processing_chats.discard(chat_id)

    if not success:
        logging.error("Failed to retry audio processing for chat_id=%s preset=%s", chat_id, preset_key)
        await callback.message.answer(
            "Извините, не получилось обработать аудио при повторной попытке 😔"
        )


async def handle_finish_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    chat_id = callback.message.chat.id if callback.message and callback.message.chat else 0
    session = job_sessions.pop(chat_id, None)
    if not session:
        if callback.message:
            await callback.message.answer('Нет активной обработки. Можете отправить новый файл.')
        return

    cleanup_work_dir(session.work_dir)
    logging.info('Session finished and cleaned for chat_id=%s', chat_id)
    if callback.message:
        await callback.message.answer('Обработка завершена, кэш очищен. Отправьте новый файл.')


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
                parse_mode="HTML",
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

    @dp.callback_query(F.data == "session:finish")
    async def handle_finish(callback: CallbackQuery) -> None:
        await handle_finish_callback(callback)

    return dp


async def run_bot() -> None:
    settings = load_settings()
    ensure_temp_root()
    load_consents()
    logging.info("Starting Telegram bot polling")
    bot = create_bot(settings)
    dp = create_dispatcher(settings)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
