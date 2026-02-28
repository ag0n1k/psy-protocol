#!/usr/bin/env python3
import asyncio
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
from aiogram.types import FSInputFile, Message

from psy_protocol.config import LOG_FORMAT
from psy_protocol.pipeline import ProcessingOptions, process_audio_file


TEMP_ROOT = Path("transcripts/telegram_temp")
SUPPORTED_AUDIO_MIME_PREFIX = "audio/"


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


def build_processing_options(settings: TelegramSettings, output_docx: Path, cache_dir: Path) -> ProcessingOptions:
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


async def download_audio(message: Message, bot: Bot, settings: TelegramSettings) -> Optional[Tuple[Path, Path, Path]]:
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
    # Remote Bot API server running with --local returns absolute filesystem paths.
    # Strip the server root so aiogram builds a valid relative HTTP download URL.
    # if not settings.api_is_local and file_path.startswith("/"):
        # server_root = settings.local_server_file_root.rstrip("/") + "/"
        # if file_path.startswith(server_root):
        #     file_path = file_path[len(server_root):]
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


async def progress_updater(status_message: Message, progress: Dict[str, Any], interval_seconds: int = 5) -> None:
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


async def process_and_reply(message: Message, bot: Bot, settings: TelegramSettings) -> None:
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
    started_at = time.monotonic()
    progress: Dict[str, Any] = {
        "done": False,
        "success": False,
        "stage": "start",
        "percent": 0.0,
        "message": "Queued",
        "started_at": started_at,
        "stage_started_at": started_at,
        "last_logged_whisper_percent": -1,
    }

    def on_progress(stage: str, percent: Optional[float], status_text: str) -> None:
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
                    "Telegram chat_id=%s message_id=%s whisper_progress=%d%%",
                    message.chat.id if message.chat else 0,
                    message.message_id,
                    rounded,
                )
                progress["last_logged_whisper_percent"] = rounded

    updater_task = asyncio.create_task(progress_updater(status_message, progress, interval_seconds=5))
    try:
        options = build_processing_options(settings, output_docx=output_docx, cache_dir=work_dir / "cache")
        docx_path, txt_path = await asyncio.to_thread(process_audio_file, audio_path, options, on_progress)
        progress["done"] = True
        progress["success"] = True
        await updater_task
        await message.answer_document(FSInputFile(path=str(txt_path)))
        await message.answer_document(FSInputFile(path=str(docx_path)))
    except Exception:
        progress["done"] = True
        progress["success"] = False
        try:
            await updater_task
        except Exception:
            logging.debug("Updater task finished with error", exc_info=True)
        logging.exception("Failed to process audio from Telegram")
        await message.answer(
            "Извините, не получилось обработать это аудио 😔 "
            "Пожалуйста, попробуйте другой файл."
        )
    finally:
        if not updater_task.done():
            updater_task.cancel()
        cleanup_work_dir(work_dir)


def create_dispatcher(settings: TelegramSettings) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def handle_start(message: Message) -> None:
        await message.answer(
            "Здравствуйте! 👋\n"
            "Я с радостью помогу обработать аудио.\n"
            "Отправьте голосовое сообщение или аудиофайл, "
            "и я верну TXT и DOCX 📄"
        )

    @dp.message(F.voice)
    async def handle_voice(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    @dp.message(F.audio)
    async def handle_audio(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    @dp.message(F.document)
    async def handle_document(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    return dp


async def run_bot() -> None:
    settings = load_settings()
    ensure_temp_root()
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
