import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EVENTS_FILE = Path('stats/events.jsonl')

EVENT_PROCESSED = 'processed'
EVENT_TEXT_DOCX = 'text_docx'
EVENT_CONSENT = 'consent'
EVENT_REJECTED = 'rejected'


def user_key(chat_id: int) -> str:
    """Стабильный псевдоним чата: считаем уникальных, не храня сам chat_id."""
    return hashlib.sha256(str(chat_id).encode('utf-8')).hexdigest()[:12]


def record_event(
    event: str,
    chat_id: int,
    path: Path = EVENTS_FILE,
    **fields: Any,
) -> None:
    """Дописать событие в JSONL. Сбои учёта не должны ломать обработку."""
    payload = {
        'ts': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'event': event,
        'user': user_key(chat_id),
    }
    payload.update(fields)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(f'{json.dumps(payload, ensure_ascii=False)}\n')
    except OSError:
        logging.warning('Failed to record usage event %s', event, exc_info=True)


def load_events(path: Path = EVENTS_FILE) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logging.warning('Skipping malformed usage event: %.80s', line)
    return events


def _parse_ts(event: Dict[str, Any]) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(event['ts'])
    except (KeyError, ValueError):
        return None


def summarize(
    events: Iterable[Dict[str, Any]],
    since: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Свод по окну: обработки, уникальные пользователи, ошибки, время работы."""
    processed = failed = text_docx = rejected = 0
    users = set()
    durations: List[float] = []

    for event in events:
        moment = _parse_ts(event)
        if since is not None and (moment is None or moment < since):
            continue
        users.add(event.get('user'))
        kind = event.get('event')
        if kind == EVENT_PROCESSED:
            if event.get('ok'):
                processed += 1
                duration = event.get('duration_sec')
                if isinstance(duration, (int, float)):
                    durations.append(float(duration))
            else:
                failed += 1
        elif kind == EVENT_TEXT_DOCX:
            text_docx += 1
        elif kind == EVENT_REJECTED:
            rejected += 1

    users.discard(None)
    return {
        'processed': processed,
        'failed': failed,
        'text_docx': text_docx,
        'rejected': rejected,
        'users': len(users),
        'avg_duration': sum(durations) / len(durations) if durations else None,
        'max_duration': max(durations) if durations else None,
    }


def _format_minutes(seconds: Optional[float]) -> str:
    if seconds is None:
        return '—'
    if seconds < 60:
        return f'{int(seconds)} с'
    return f'{seconds / 60:.1f} мин'


def _format_window(title: str, summary: Dict[str, Any]) -> str:
    total = summary['processed'] + summary['failed']
    if not total and not summary['text_docx']:
        return f'<b>{title}</b>\nпусто'
    lines = [
        f'<b>{title}</b>',
        f'обработок: {summary["processed"]}'
        + (f' (ошибок: {summary["failed"]})' if summary['failed'] else ''),
        f'пользователей: {summary["users"]}',
    ]
    if summary['text_docx']:
        lines.append(f'текстовых файлов: {summary["text_docx"]}')
    if summary['rejected']:
        lines.append(f'отказов из-за очереди: {summary["rejected"]}')
    if summary['avg_duration'] is not None:
        lines.append(
            f'время обработки: среднее {_format_minutes(summary["avg_duration"])}, '
            f'макс {_format_minutes(summary["max_duration"])}'
        )
    return '\n'.join(lines)


def format_report(
    events: List[Dict[str, Any]],
    now: Optional[datetime.datetime] = None,
) -> str:
    """Готовый HTML-текст для /stats."""
    if not events:
        return '📊 Статистика пока пуста — событий не записано.'

    now = now or datetime.datetime.now().astimezone()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    windows = [
        ('Сегодня', today),
        ('7 дней', now - datetime.timedelta(days=7)),
        ('30 дней', now - datetime.timedelta(days=30)),
        ('Всё время', None),
    ]
    blocks = [_format_window(title, summarize(events, since)) for title, since in windows]

    first = min((ts for ts in (_parse_ts(e) for e in events) if ts), default=None)
    footer = f'\n\nучёт с {first.strftime("%d.%m.%Y")}' if first else ''
    return '📊 <b>Использование бота</b>\n\n' + '\n\n'.join(blocks) + footer
