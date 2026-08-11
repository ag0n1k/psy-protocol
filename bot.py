#!/usr/bin/env python3
import asyncio
import copy
import logging
import logging.handlers
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import SimpleFilesPathWrapper, TelegramAPIServer
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from psy_protocol.dialogue_parser import parse_dialogue_text
from psy_protocol.config import LOG_FORMAT
from psy_protocol.docx_writer import create_docx
from psy_protocol.pipeline import ProcessingOptions, process_audio_file
from psy_protocol.usage_stats import (
    EVENT_CONSENT,
    EVENT_PROCESSED,
    EVENT_REJECTED,
    EVENT_TEXT_DOCX,
    format_report,
    load_events,
    record_event,
)


TEMP_ROOT = Path("transcripts/telegram_temp")
SUPPORTED_AUDIO_MIME_PREFIX = "audio/"

PHOTO_CAPTION_LIMIT = 1024
DEFAULT_MAX_AUDIO_MINUTES = 120
DEFAULT_MAX_AUDIO_MB = 300
DEFAULT_MAX_QUEUE_LENGTH = 5

LOG_FILE = Path('logs/bot.log')
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

DONATE_BANNER = Path('assets/donate_banner.png')
DONATE_LINK_BUTTON_TEXT = '☕️ Купить кофе'
DONATE_TON_BUTTON_TEXT = '💎 USDT (TON)'
# USD₮ jetton master on TON: https://tonscan.com/EQCxE6mUtQJKFnGfaROTKOt1lZbDiiX1kCixRv7Nw2Id_sDs
TON_USDT_JETTON_MASTER = 'EQCxE6mUtQJKFnGfaROTKOt1lZbDiiX1kCixRv7Nw2Id_sDs'
DONATE_INTRO = (
    '☕️ <b>Кофе для бота</b>\n\n'
    'Бот работает на домашней машине: транскрипции считаются локально, '
    'без облаков и подписок. Если он сэкономил вам час рутины — '
    'можно поддержать проект на кофе и электричество.\n\n'
    'Любая сумма по желанию. Это не влияет на работу бота: '
    'всё бесплатно и остаётся бесплатным.'
)

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
class QueueTicket:
    """Место в очереди на пайплайн; живёт с постановки до захвата семафора."""

    chat_id: int
    enqueued_at: float


@dataclass
class JobSession:
    work_dir: Path
    audio_path: Path
    base_options: ProcessingOptions
    created_at: float


# Keyed by chat_id (int)
job_sessions: Dict[int, "JobSession"] = {}
processing_chats: set[int] = set()
# Каталоги задач, которые прямо сейчас в очереди или в обработке: уборщик их не трогает.
active_work_dirs: set[Path] = set()
PIPELINE_SEMAPHORE = asyncio.Semaphore(1)

pipeline_queue: "list[QueueTicket]" = []

# Соглашение обещает удаление аудио и результатов через час — этот срок и выдерживаем.
SESSION_TTL_SECONDS = 3600
CLEANUP_INTERVAL_SECONDS = 300

CONSENTS_FILE = Path("consents/accepted.txt")
consented_users: set[int] = set()
# Кому доступна /stats; заполняется в run_bot() из PSY_ADMIN_CHAT_IDS.
admin_chat_ids: set[int] = set()
T = TypeVar('T')


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


def revoke_consent(chat_id: int) -> None:
    """Убрать chat_id из файла согласий, переписав его целиком."""
    consented_users.discard(chat_id)
    if not CONSENTS_FILE.exists():
        return
    kept = [
        line for line in CONSENTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() != str(chat_id)
    ]
    CONSENTS_FILE.write_text('\n'.join(kept) + ('\n' if kept else ''), encoding="utf-8")


async def _run_with_retries(
    call: Callable[[], Awaitable[T]],
    operation: str,
    attempts: int = 3,
) -> T:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except TelegramNetworkError as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = float(attempt)
            logging.warning(
                'Telegram timeout during %s (attempt %d/%d), retrying in %.1fs',
                operation,
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)
    raise last_error or RuntimeError(f'Failed operation: {operation}')


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


@dataclass
class Limits:
    """Операционные пределы: зависят от машины, поэтому живут в окружении."""

    max_audio_seconds: int = DEFAULT_MAX_AUDIO_MINUTES * 60
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_MB * 1024 * 1024
    # Сколько задач может ждать очереди, не считая обрабатываемой сейчас.
    max_queue_length: int = DEFAULT_MAX_QUEUE_LENGTH


limits = Limits()


@dataclass
class DonateSettings:
    """Реквизиты для донатов, задаются только через окружение."""

    link_url: Optional[str] = None
    ton_address: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.link_url or self.ton_address)


# Заполняется в run_bot(): клавиатуры собираются там, куда settings не прокидываются.
donate_settings = DonateSettings()


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


def read_env() -> Dict[str, str]:
    """Значения из .env, перекрытые переменными окружения."""
    return {**parse_env_file(Path(".env")), **os.environ}


def load_settings(env: Optional[Dict[str, str]] = None) -> TelegramSettings:
    env = env if env is not None else read_env()
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


def _env_int(env: Dict[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.warning('Invalid %s=%r, falling back to %d', key, raw, default)
        return default
    if value <= 0:
        logging.warning('Non-positive %s=%d, falling back to %d', key, value, default)
        return default
    return value


def load_limits(env: Optional[Dict[str, str]] = None) -> Limits:
    env = env if env is not None else read_env()
    loaded = Limits(
        max_audio_seconds=_env_int(env, 'PSY_MAX_AUDIO_MINUTES', DEFAULT_MAX_AUDIO_MINUTES) * 60,
        max_audio_bytes=_env_int(env, 'PSY_MAX_AUDIO_MB', DEFAULT_MAX_AUDIO_MB) * 1024 * 1024,
        max_queue_length=_env_int(env, 'PSY_MAX_QUEUE_LENGTH', DEFAULT_MAX_QUEUE_LENGTH),
    )
    logging.info(
        'Limits: audio <= %d min / %d MB, queue <= %d waiting',
        loaded.max_audio_seconds // 60,
        loaded.max_audio_bytes // (1024 * 1024),
        loaded.max_queue_length,
    )
    return loaded


def load_admin_chat_ids(env: Optional[Dict[str, str]] = None) -> set[int]:
    env = env if env is not None else read_env()
    raw = env.get("PSY_ADMIN_CHAT_IDS", "")
    ids = set()
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            logging.warning("Ignoring non-numeric admin chat id: %r", chunk)
    if not ids:
        logging.info("No admins configured, /stats is disabled (set PSY_ADMIN_CHAT_IDS)")
    return ids


def load_donate_settings(env: Optional[Dict[str, str]] = None) -> DonateSettings:
    env = env if env is not None else read_env()
    settings = DonateSettings(
        link_url=env.get("PSY_DONATE_URL") or None,
        ton_address=env.get("PSY_DONATE_TON_ADDRESS") or None,
    )
    if not settings.is_configured:
        logging.warning(
            "Donations are not configured: set PSY_DONATE_URL and/or PSY_DONATE_TON_ADDRESS"
        )
    return settings


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


def ton_transfer_url(address: str) -> str:
    """Universal link Tonkeeper на перевод USD₮ в сети TON."""
    return f'https://app.tonkeeper.com/transfer/{address}?jetton={TON_USDT_JETTON_MASTER}'


def build_donate_rows() -> list[list[InlineKeyboardButton]]:
    """Ряды кнопок донатов; пустой список, если реквизиты не заданы."""
    buttons = []
    if donate_settings.link_url:
        buttons.append(
            InlineKeyboardButton(text=DONATE_LINK_BUTTON_TEXT, url=donate_settings.link_url)
        )
    if donate_settings.ton_address:
        buttons.append(
            InlineKeyboardButton(
                text=DONATE_TON_BUTTON_TEXT,
                url=ton_transfer_url(donate_settings.ton_address),
            )
        )
    return [buttons] if buttons else []


def build_donate_keyboard() -> Optional[InlineKeyboardMarkup]:
    rows = build_donate_rows()
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def build_donate_text() -> str:
    text = DONATE_INTRO
    if donate_settings.link_url:
        text = f'{text}\n\nКартой или через СБП — по кнопке ниже.'
    if donate_settings.ton_address:
        text = (
            f'{text}\n\nВ крипте — USD₮ или TON в сети TON:\n'
            f'<code>{donate_settings.ton_address}</code>\n'
            'Адрес копируется нажатием. Кнопка ниже открывает Tonkeeper с уже '
            'выбранным USD₮; в других кошельках отправляйте на адрес вручную.'
        )
    return text


async def send_donate_message(message: Message) -> None:
    """Отправить баннер донатов с кнопками; без картинки — только текст."""
    keyboard = build_donate_keyboard()
    if not keyboard:
        await message.answer(
            'Донаты пока не настроены — спасибо за желание поддержать! 🙏'
        )
        return

    donate_text = build_donate_text()
    # Подпись к фото ограничена 1024 символами, длинный текст уходит отдельным сообщением.
    fits_in_caption = len(donate_text) <= PHOTO_CAPTION_LIMIT
    if DONATE_BANNER.exists():
        try:
            await _run_with_retries(
                lambda: message.answer_photo(
                    photo=FSInputFile(path=str(DONATE_BANNER)),
                    caption=donate_text if fits_in_caption else None,
                    parse_mode='HTML',
                    reply_markup=keyboard if fits_in_caption else None,
                ),
                operation='send donate banner',
            )
            if fits_in_caption:
                return
        except Exception:
            # Не только сеть: битый файл или превышенный лимит дают TelegramBadRequest,
            # и без этого фолбэка пользователь не получил бы вообще ничего.
            logging.warning('Failed to send donate banner, falling back to text', exc_info=True)
    else:
        logging.warning('Donate banner not found at %s', DONATE_BANNER)

    await message.answer(donate_text, parse_mode='HTML', reply_markup=keyboard)


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
            *build_donate_rows(),
        ]
    )


def ensure_temp_root() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def build_work_paths(message: Message, suffix: str) -> Tuple[Path, Path, Path, Path]:
    work_dir = TEMP_ROOT / f"{message.chat.id}_{message.message_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / f"input{suffix}"
    output_docx = work_dir / "result.docx"
    cache_dir = work_dir / "cache"
    return work_dir, audio_path, output_docx, cache_dir


def check_media_limits(media: Any) -> Optional[str]:
    """Причина отказа, если файл слишком велик или длинен; None — можно брать."""
    duration = getattr(media, 'duration', None)
    if isinstance(duration, int) and duration > limits.max_audio_seconds:
        return (
            f'Запись длиннее {limits.max_audio_seconds // 60} минут '
            f'({duration // 60} мин) — такую бот не потянет. '
            'Разрежьте её на части и пришлите по очереди 🙏'
        )
    size = getattr(media, 'file_size', None)
    if isinstance(size, int) and size > limits.max_audio_bytes:
        return (
            f'Файл больше {limits.max_audio_bytes // (1024 * 1024)} МБ '
            f'({size // (1024 * 1024)} МБ). Если это WAV с диктофона — '
            'пересохраните в mp3 или m4a, звук для распознавания не пострадает 🙏'
        )
    return None


async def download_audio(
    message: Message, bot: Bot, settings: TelegramSettings,
) -> Optional[Tuple[Path, Path, Path, Path]]:
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

    media = message.voice or message.audio or message.document
    media_size = getattr(media, "file_size", None) if media else None
    media_kind = type(media).__name__ if media else "unknown"
    logging.info(
        "download_audio: kind=%s file_size=%s file_id=%s",
        media_kind, media_size, file_id,
    )

    work_dir, audio_path, output_docx, cache_dir = build_work_paths(message, suffix)
    tg_file = await bot.get_file(file_id)
    file_path = tg_file.file_path or ""
    await bot.download_file(file_path, destination=audio_path)
    return work_dir, audio_path, output_docx, cache_dir


def cleanup_work_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def tree_mtime(path: Path) -> float:
    """Самое свежее время изменения в поддереве — каталог кэша обновляется, а сам work_dir нет."""
    latest = path.stat().st_mtime
    for child in path.rglob('*'):
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            continue
    return latest


def purge_temp_root() -> None:
    """Снести всё временное при старте: после рестарта активных задач заведомо нет."""
    if not TEMP_ROOT.exists():
        return
    removed = 0
    for child in TEMP_ROOT.iterdir():
        if child.is_dir():
            cleanup_work_dir(child)
            removed += 1
    if removed:
        logging.info("Purged %d stale work dirs at startup", removed)


def cleanup_expired_data() -> int:
    """Удалить аудио и результаты старше SESSION_TTL_SECONDS. Возвращает число каталогов."""
    now = time.time()
    removed = 0

    for chat_id, session in list(job_sessions.items()):
        if session.work_dir in active_work_dirs:
            continue
        if now - session.created_at < SESSION_TTL_SECONDS:
            continue
        cleanup_work_dir(session.work_dir)
        job_sessions.pop(chat_id, None)
        removed += 1
        logging.info("Session data expired and removed for chat_id=%s", chat_id)

    # Каталоги без сессии: остались от упавших задач или от прошлых запусков.
    known_dirs = {session.work_dir for session in job_sessions.values()} | active_work_dirs
    if TEMP_ROOT.exists():
        for child in TEMP_ROOT.iterdir():
            if not child.is_dir() or child in known_dirs:
                continue
            try:
                if now - tree_mtime(child) < SESSION_TTL_SECONDS:
                    continue
            except OSError:
                continue
            cleanup_work_dir(child)
            removed += 1
            logging.info("Orphaned work dir expired and removed: %s", child)

    return removed


async def cleanup_worker() -> None:
    """Фоновая уборка временных данных по TTL."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            removed = await asyncio.to_thread(cleanup_expired_data)
            if removed:
                logging.info("Cleanup removed %d expired work dirs", removed)
        except Exception:
            logging.exception("Cleanup iteration failed")


def queue_is_full() -> bool:
    return len(pipeline_queue) >= limits.max_queue_length


def enqueue_ticket(chat_id: int) -> QueueTicket:
    ticket = QueueTicket(chat_id=chat_id, enqueued_at=time.monotonic())
    pipeline_queue.append(ticket)
    logging.info(
        "Queued chat_id=%s, waiting=%d, busy=%s",
        chat_id, len(pipeline_queue), PIPELINE_SEMAPHORE.locked(),
    )
    return ticket


def release_ticket(ticket: QueueTicket) -> None:
    if ticket in pipeline_queue:
        pipeline_queue.remove(ticket)
        logging.info(
            "Dequeued chat_id=%s after %.1fs, waiting=%d",
            ticket.chat_id, time.monotonic() - ticket.enqueued_at, len(pipeline_queue),
        )


def queue_position(ticket: QueueTicket) -> int:
    """1 — следующий на обработку; 0 — тикет уже покинул очередь."""
    try:
        return pipeline_queue.index(ticket) + 1
    except ValueError:
        return 0


def build_bar(percent: float) -> str:
    clamped = max(0.0, min(100.0, percent))
    filled = int(clamped // 10)
    return f"{'█' * filled}{'░' * (10 - filled)}"


STAGE_LABELS = {
    "queue": "Ожидание в очереди",
    "start": "Идёт обработка аудио",
    "prepare": "Идёт обработка аудио",
    "whisper": "Идёт обработка аудио",
    "diarization": "Идёт обработка аудио",
    "replicas": "Идёт обработка аудио",
    "output": "Идёт обработка аудио",
    "done": "Готово",
}


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.title())


def render_queue_text(ticket: Optional[QueueTicket]) -> str:
    """Текст ожидания: бот считает по одному файлу за раз."""
    position = queue_position(ticket) if ticket else 0
    if position <= 1:
        place = "Ваш файл — следующий."
    else:
        place = f"Перед вами в очереди: {position - 1}."
    return (
        "⏳ Сейчас обрабатывается другой файл.\n"
        f"{place}\n"
        "Начну автоматически, как освободится — ждать в чате не нужно."
    )


def render_progress_text(progress: Dict[str, Any]) -> str:
    if progress.get("done"):
        if progress.get("success"):
            return "✅ Готово! Отправляю файлы."
        return "❌ Не удалось обработать аудио."

    stage = progress.get("stage", "start")
    if stage == "queue":
        return render_queue_text(progress.get("ticket"))

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
    finished: Optional[asyncio.Event] = progress.get("finished")
    while not progress.get("done"):
        text = render_progress_text(progress)
        if text != last_text:
            try:
                await status_message.edit_text(text)
                last_text = text
            except Exception:
                logging.debug("Status message edit skipped", exc_info=True)
        if finished is None:
            await asyncio.sleep(interval_seconds)
            continue
        try:
            # Просыпаемся сразу по завершении, иначе результат ждёт конца интервала.
            await asyncio.wait_for(finished.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass

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
    source: str = 'audio',
    ticket: Optional[QueueTicket] = None,
) -> bool:
    """Run the pipeline and send result files. Returns True on success."""
    updater_task = asyncio.create_task(
        progress_updater(status_message, progress, interval_seconds=5)
    )
    if ticket is None:
        ticket = enqueue_ticket(chat_id)
    started_at = time.monotonic()
    try:
        progress["ticket"] = ticket
        # Проверка стоит вплотную к acquire: между ними нет ни одного await,
        # поэтому увиденное состояние семафора не успевает устареть.
        if PIPELINE_SEMAPHORE.locked() or queue_position(ticket) > 1:
            _update_progress(progress, "queue", 0.0, "Waiting in processing queue", chat_id)

        async with PIPELINE_SEMAPHORE:
            release_ticket(ticket)
            _update_progress(progress, "start", 0.0, "Processing started", chat_id)
            docx_path, txt_path = await asyncio.to_thread(
                process_audio_file, audio_path, opts, lambda s, p, m: _update_progress(progress, s, p, m, chat_id)
            )
        _finish_progress(progress, success=True)
        await updater_task
        await _run_with_retries(
            lambda: reply_target.answer_document(FSInputFile(path=str(txt_path))),
            operation='send txt result',
        )
        await _run_with_retries(
            lambda: reply_target.answer_document(FSInputFile(path=str(docx_path))),
            operation='send docx result',
        )
        await _run_with_retries(
            lambda: reply_target.answer(
                "Если результат неточный — выберите один из вариантов на кнопках ниже.",
                reply_markup=build_retry_keyboard(),
            ),
            operation='send retry keyboard',
        )
        record_event(
            EVENT_PROCESSED, chat_id, ok=True, source=source,
            duration_sec=round(time.monotonic() - started_at, 1),
        )
        return True
    except Exception:
        _finish_progress(progress, success=False)
        record_event(
            EVENT_PROCESSED, chat_id, ok=False, source=source,
            duration_sec=round(time.monotonic() - started_at, 1),
        )
        logging.exception("Pipeline failed for chat_id=%s", chat_id)
        try:
            await updater_task
        except Exception:
            logging.debug("Updater task finished with error", exc_info=True)
        return False
    finally:
        release_ticket(ticket)
        if not updater_task.done():
            updater_task.cancel()


def _finish_progress(progress: Dict[str, Any], success: bool) -> None:
    progress["done"] = True
    progress["success"] = success
    finished = progress.get("finished")
    if finished is not None:
        finished.set()


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
        "finished": asyncio.Event(),
        "done": False,
        "success": False,
        "stage": "start",
        "percent": 0.0,
        "message": "Queued",
        "started_at": now,
        "stage_started_at": now,
        "last_logged_whisper_percent": -1,
    }


def read_text_file(path: Path) -> Optional[str]:
    """Прочитать .txt: из Word на Windows такие файлы приходят в cp1251, а не в UTF-8."""
    for encoding in ('utf-8', 'utf-8-sig', 'cp1251'):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            logging.exception('Failed to read text file %s', path)
            return None
    logging.warning('Unsupported encoding for text file %s', path)
    return None


async def process_text_file_and_reply(message: Message, bot: Bot) -> None:
    """Download a text document, parse К:/Т: replicas, generate and send DOCX."""
    chat_id = message.chat.id
    if chat_id not in consented_users:
        await message.answer(
            CONSENT_TEXT,
            parse_mode='HTML',
            reply_markup=build_consent_keyboard(),
        )
        return

    file_id = message.document.file_id
    work_dir = TEMP_ROOT / f'{chat_id}_{message.message_id}'
    work_dir.mkdir(parents=True, exist_ok=True)

    original_name = message.document.file_name or 'input.txt'
    input_path = work_dir / original_name
    tg_file = await bot.get_file(file_id)
    await bot.download_file(tg_file.file_path or '', destination=input_path)

    text = read_text_file(input_path)
    if text is None:
        await message.answer(
            'Не удалось прочитать файл: ожидается текст в кодировке UTF-8 или Windows-1251. '
            'Пересохраните файл в UTF-8 и пришлите снова 🙏'
        )
        cleanup_work_dir(work_dir)
        return

    replicas = parse_dialogue_text(text)
    if not replicas:
        await message.answer(
            'Не удалось найти реплики в файле. '
            'Ожидается формат: строки вида «К: текст» и «Т: текст».',
        )
        cleanup_work_dir(work_dir)
        return

    logging.info(
        'Text file from chat_id=%s: %d replicas parsed', chat_id, len(replicas),
    )

    output_docx = work_dir / Path(original_name).with_suffix('.docx').name
    metadata = {'ФИО': '', 'Номер группы': '', 'Дата': '', 'Тема протокола': '', 'Задание': ''}

    try:
        await asyncio.to_thread(
            create_docx,
            output_path=str(output_docx),
            replicas=replicas,
            metadata=metadata,
        )
    except Exception:
        logging.exception('DOCX generation failed for text file, chat_id=%s', chat_id)
        await message.answer('Не удалось сгенерировать DOCX 😔')
        cleanup_work_dir(work_dir)
        return

    await _run_with_retries(
        lambda: message.answer_document(FSInputFile(path=str(output_docx))),
        operation='send text-to-docx result',
    )
    record_event(EVENT_TEXT_DOCX, chat_id, replicas=len(replicas))
    cleanup_work_dir(work_dir)


async def process_and_reply(message: Message, bot: Bot, settings: TelegramSettings) -> None:
    chat_id = message.chat.id
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

    if queue_is_full():
        record_event(EVENT_REJECTED, chat_id, reason='queue_full')
        await message.answer(
            "Сейчас очередь заполнена: бот обрабатывает файлы по одному, "
            f"и в ожидании уже {len(pipeline_queue)}. Попробуйте отправить запись чуть позже 🙏"
        )
        return

    media = message.voice or message.audio or message.document
    limit_error = check_media_limits(media) if media else None
    if limit_error:
        record_event(EVENT_REJECTED, chat_id, reason='too_large')
        await message.answer(limit_error)
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
    # Тикет берём до скачивания: иначе между проверкой очереди и постановкой в неё
    # проходят минуты закачки, и одновременные отправки пробивают лимит.
    ticket = enqueue_ticket(chat_id)
    download_result: Optional[Tuple[Path, Path, Path, Path]] = None
    try:
        download_result = await download_audio(message, bot, settings)
        if not download_result:
            await message.answer(
                "Пожалуйста, отправьте голосовое сообщение, аудио или аудиофайл документом 🙏"
            )
            return

        work_dir, audio_path, output_docx, cache_dir = download_result
        active_work_dirs.add(work_dir)
        if PIPELINE_SEMAPHORE.locked():
            initial_status = (
                "Спасибо! 😊 Аудио получено и поставлено в очередь: "
                "бот обрабатывает файлы по одному ⏳"
            )
        else:
            initial_status = (
                "Спасибо! 😊 Аудио получено, начинаю обработку. "
                "Пожалуйста, немного подождите ⏳"
            )
        status_message = await _run_with_retries(
            lambda: message.answer(initial_status),
            operation='send initial status',
        )
        options = build_processing_options(settings, output_docx=output_docx, cache_dir=cache_dir)
        progress = _make_progress()

        success = await run_pipeline_and_send(
            chat_id=chat_id,
            audio_path=audio_path,
            opts=options,
            status_message=status_message,
            reply_target=message,
            progress=progress,
            source='voice' if message.voice else 'audio',
            ticket=ticket,
        )

        if success:
            previous = job_sessions.get(chat_id)
            if previous and previous.work_dir != work_dir:
                # Иначе каталог прошлой сессии остаётся без ссылок и переживёт TTL-уборку.
                cleanup_work_dir(previous.work_dir)
            job_sessions[chat_id] = JobSession(
                work_dir=work_dir,
                audio_path=audio_path,
                base_options=options,
                created_at=time.time(),
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
        release_ticket(ticket)
        processing_chats.discard(chat_id)
        if download_result:
            active_work_dirs.discard(download_result[0])


async def handle_retry_callback(
    callback: CallbackQuery, bot: Bot, settings: TelegramSettings
) -> None:
    try:
        await _run_with_retries(
            lambda: callback.answer(),
            operation='ack retry callback',
        )
    except TelegramNetworkError:
        logging.warning('Telegram timeout while acknowledging retry callback')

    # Telegram не присылает message для кнопок старше 48 часов, а они живут в чате вечно.
    if not callback.message:
        logging.info('Retry callback without message (too old), ignoring')
        return

    data = callback.data or ''
    preset_key = data.split(':', 1)[1] if ':' in data else ''
    if preset_key not in PRESETS:
        logging.warning('Unknown retry preset in callback data: %r', data)
        await callback.message.answer('Неизвестное действие, отправьте аудио заново.')
        return

    chat_id = callback.message.chat.id

    if chat_id in processing_chats:
        await callback.message.answer('Обработка уже выполняется. Дождитесь завершения.')
        return

    if queue_is_full():
        record_event(EVENT_REJECTED, chat_id, reason='queue_full')
        await callback.message.answer(
            'Сейчас очередь заполнена, попробуйте повторить обработку чуть позже 🙏'
        )
        return

    session = job_sessions.get(chat_id)
    if not session:
        await callback.message.answer("Активный файл не найден, отправьте аудио заново.")
        return
    if not session.audio_path.exists():
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

    status_message = await _run_with_retries(
        lambda: callback.message.answer(
            f"⏳ Повторная обработка ({PRESETS[preset_key]['label']})…"
        ),
        operation='send retry status',
    )
    progress = _make_progress()
    processing_chats.add(chat_id)
    active_work_dirs.add(session.work_dir)
    try:
        success = await run_pipeline_and_send(
            chat_id=chat_id,
            audio_path=session.audio_path,
            opts=opts,
            status_message=status_message,
            reply_target=callback.message,
            progress=progress,
            source=f'retry:{preset_key}',
        )
    finally:
        processing_chats.discard(chat_id)
        active_work_dirs.discard(session.work_dir)

    if not success:
        logging.error("Failed to retry audio processing for chat_id=%s preset=%s", chat_id, preset_key)
        await callback.message.answer(
            "Извините, не получилось обработать аудио при повторной попытке 😔"
        )


async def handle_finish_callback(callback: CallbackQuery) -> None:
    try:
        await _run_with_retries(
            lambda: callback.answer(),
            operation='ack finish callback',
        )
    except TelegramNetworkError:
        logging.warning('Telegram timeout while acknowledging finish callback')

    if not callback.message:
        logging.info('Finish callback without message (too old), ignoring')
        return
    chat_id = callback.message.chat.id

    # Кнопка живёт в истории чата вечно: без этой проверки нажатие под старым
    # результатом снесёт каталог задачи, которая сейчас в очереди или в работе.
    if chat_id in processing_chats:
        await callback.message.answer(
            'Сейчас идёт обработка файла — дождитесь её завершения.'
        )
        return

    session = job_sessions.pop(chat_id, None)
    if not session:
        await callback.message.answer('Нет активной обработки. Можете отправить новый файл.')
        return

    cleanup_work_dir(session.work_dir)
    logging.info('Session finished and cleaned for chat_id=%s', chat_id)
    await callback.message.answer('Обработка завершена, кэш очищен. Отправьте новый файл.')


def build_forget_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text='🗑 Да, удалить', callback_data='forget:confirm'),
        ]]
    )


async def handle_forget_callback(callback: CallbackQuery) -> None:
    try:
        await _run_with_retries(
            lambda: callback.answer(),
            operation='ack forget callback',
        )
    except TelegramNetworkError:
        logging.warning('Telegram timeout while acknowledging forget callback')

    if not callback.message:
        logging.info('Forget callback without message (too old), ignoring')
        return
    chat_id = callback.message.chat.id

    if chat_id in processing_chats:
        await callback.message.answer(
            'Сейчас идёт обработка вашего файла — дождитесь её завершения и повторите /forget.'
        )
        return

    session = job_sessions.pop(chat_id, None)
    if session:
        cleanup_work_dir(session.work_dir)
    revoke_consent(chat_id)
    logging.info('Data removed and consent revoked for chat_id=%s', chat_id)

    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(
        '🗑 Готово. Аудио и результаты удалены, согласие отозвано.\n\n'
        'Чтобы снова пользоваться ботом, отправьте /start и примите соглашение.'
    )


async def handle_consent_callback(callback: CallbackQuery) -> None:
    try:
        await _run_with_retries(
            lambda: callback.answer(),
            operation='ack consent callback',
        )
    except TelegramNetworkError:
        logging.warning('Telegram timeout while acknowledging consent callback')

    if not callback.message:
        logging.info('Consent callback without message (too old), ignoring')
        return
    chat_id = callback.message.chat.id

    if chat_id not in consented_users:
        consented_users.add(chat_id)
        save_consent(chat_id)
        record_event(EVENT_CONSENT, chat_id)
        logging.info("Consent accepted for chat_id=%s", chat_id)
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(
        "✅ Соглашение принято. Отправьте голосовое сообщение или аудиофайл 📄"
    )


def create_dispatcher(settings: TelegramSettings) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def handle_start(message: Message) -> None:
        chat_id = message.chat.id
        if chat_id in consented_users:
            await message.answer(
                "Здравствуйте! 👋 Вы уже приняли соглашение.\n"
                "Отправьте голосовое сообщение или аудиофайл 📄\n\n"
                "Поддержать проект: /donate ☕️"
            )
        else:
            await message.answer(
                CONSENT_TEXT,
                parse_mode="HTML",
                reply_markup=build_consent_keyboard(),
            )

    @dp.message(Command('donate'))
    async def handle_donate(message: Message) -> None:
        await send_donate_message(message)

    @dp.message(Command('forget'))
    async def handle_forget(message: Message) -> None:
        await message.answer(
            '🗑 <b>Удаление данных</b>\n\n'
            'Будут удалены загруженное аудио и результаты обработки, '
            'а согласие с пользовательским соглашением отозвано.\n\n'
            'Обезличенная статистика использования (без вашего id) сохраняется.',
            parse_mode='HTML',
            reply_markup=build_forget_keyboard(),
        )

    @dp.message(Command('stats'))
    async def handle_stats(message: Message) -> None:
        chat_id = message.chat.id
        if chat_id not in admin_chat_ids:
            # Для остальных команда как будто не существует — обычное приветствие.
            await handle_start(message)
            return
        events = await asyncio.to_thread(load_events)
        await message.answer(format_report(events), parse_mode='HTML')

    @dp.callback_query(F.data == "consent:accept")
    async def handle_consent(callback: CallbackQuery) -> None:
        await handle_consent_callback(callback)

    @dp.callback_query(F.data == "forget:confirm")
    async def handle_forget_confirm(callback: CallbackQuery) -> None:
        await handle_forget_callback(callback)

    @dp.message(F.voice)
    async def handle_voice(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    @dp.message(F.audio)
    async def handle_audio(message: Message, bot: Bot) -> None:
        await process_and_reply(message, bot, settings)

    @dp.message(F.document)
    async def handle_document(message: Message, bot: Bot) -> None:
        mime = message.document.mime_type or '' if message.document else ''
        fname = (message.document.file_name or '') if message.document else ''
        is_text = mime.startswith('text/') or fname.endswith('.txt')
        if is_text:
            await process_text_file_and_reply(message, bot)
            return
        await process_and_reply(message, bot, settings)

    @dp.callback_query(F.data.startswith("retry:"))
    async def handle_retry(callback: CallbackQuery, bot: Bot) -> None:
        await handle_retry_callback(callback, bot, settings)

    @dp.callback_query(F.data == "session:finish")
    async def handle_finish(callback: CallbackQuery) -> None:
        await handle_finish_callback(callback)

    @dp.message()
    async def handle_any(message: Message) -> None:
        await handle_start(message)

    return dp


async def run_bot() -> None:
    global donate_settings, admin_chat_ids, limits
    env = read_env()
    settings = load_settings(env)
    donate_settings = load_donate_settings(env)
    admin_chat_ids = load_admin_chat_ids(env)
    limits = load_limits(env)
    ensure_temp_root()
    purge_temp_root()
    load_consents()
    logging.info("Starting Telegram bot polling")
    bot = create_bot(settings)
    dp = create_dispatcher(settings)
    cleanup_task = asyncio.create_task(cleanup_worker())
    try:
        try:
            await bot.set_my_commands([
                BotCommand(command='start', description='Начать работу'),
                BotCommand(command='donate', description='Поддержать проект ☕️'),
                BotCommand(command='forget', description='Удалить мои данные'),
            ])
        except Exception:
            # Меню команд не критично: без него бот работает, а падение здесь
            # раньше роняло процесс на каждом старте при недоступном Bot API.
            logging.warning('Failed to publish bot commands, continuing', exc_info=True)
        await dp.start_polling(bot)
    finally:
        cleanup_task.cancel()
        await bot.session.close()


def configure_logging() -> None:
    """Логи бота — в ротируемый файл; launchd-овский stderr остаётся для падений."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def main() -> None:
    configure_logging()
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
