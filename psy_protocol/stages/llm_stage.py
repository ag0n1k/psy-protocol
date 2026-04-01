"""Stage 4: LLM enhancement via OpenAI SDK → LMStudio."""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from ..models import AlignmentResult, LlmResult, Replica

THERAPIST_MARKERS = [
    'я замечаю',
    'я чувствую',
    'я слышу',
    'я вижу',
    'я удивил',
    'что для тебя',
    'можешь рассказать',
    'расскажи мне',
    'предлагаю',
    'предлагаю остановиться',
    'как ты',
    'что ты чувствуешь',
    'у тебя голос',
    'замечаю, что',
    'мне важно',
    'я хочу спросить',
    'ты говорил',
    'ты говорила',
    'ты сказал',
    'ты сказала',
]

ROLE_VALIDATION_SYSTEM = """\
Ты — эксперт по анализу транскриптов гештальт-терапевтических сессий.
В сессии двое участников: К (клиент) и Т (терапевт).

Маркеры терапевта (Т):
- Контрперенос: "я замечаю...", "я чувствую...", "я слышу...", "я удивился..."
- Вопросы к клиенту: "что для тебя...", "можешь рассказать...", "как ты..."
- Наблюдения: "у тебя голос изменился...", "ты говорил что..."
- Управление сессией: "предлагаю остановиться...", "давай сосредоточимся..."

Маркеры клиента (К):
- Личные истории и переживания
- Ответы на вопросы терапевта
- Описание своих ощущений и эмоций
- Как правило, больше говорит по объёму

Правила:
- Соседние фрагменты чаще принадлежат одному спикеру
- Короткие фрагменты ("Ну.", "Да.", "Ой.") — реакции или продолжения
- После вопроса Т следует ответ К
- predicted-роль может быть ошибочной — определяй роль по содержанию

Формат ввода: N [MM:SS] predicted=РОЛЬ: "текст"
Формат вывода: JSON-массив строк ["К", "Т", ...] той же длины, что входных фрагментов.
Никакого другого текста — только JSON-массив.\
"""

FEW_SHOT_USER = """\
12 фрагментов транскрипта:
0 [00:13] predicted=К: "я испытываю раздражение на свою какую-то неспособность"
1 [00:59] predicted=Т: "ну"
2 [01:00] predicted=К: "знаешь я сначала удивился потому что ты говорила про это"
3 [01:20] predicted=Т: "да"
4 [01:22] predicted=К: "какая-то дисфункция как будто я должна это признать"
5 [02:15] predicted=К: "остановиться здесь на секунду"
6 [02:45] predicted=Т: "что для тебя означает это раздражение?"
7 [03:10] predicted=К: "это такое ощущение что я не справляюсь"
8 [03:45] predicted=Т: "я замечаю как твой голос изменился когда ты сказал это"
9 [04:00] predicted=К: "да ты права"
10 [04:02] predicted=Т: "да"
11 [04:05] predicted=К: "это страшно признавать"\
"""

FEW_SHOT_ASSISTANT = '["К", "К", "Т", "К", "К", "Т", "Т", "К", "Т", "К", "К", "К"]'

TEXT_CORRECTION_SYSTEM = """\
Ты — редактор транскриптов психологических сессий.
Исправь ошибки ASR: пунктуацию, явные опечатки, границы предложений.
НЕ меняй смысл. НЕ добавляй слова. НЕ удаляй слова.

Формат ввода: JSON-массив строк с текстами реплик.
Формат вывода: JSON-массив строк с исправленными текстами той же длины.
Никакого другого текста — только JSON-массив.\
"""

ANALYSIS_SYSTEM = """\
Ты — эксперт по гештальт-терапии. Проанализируй сессию и дай краткое резюме:
- Ключевые темы и запрос клиента
- Основные феномены и динамика
- Терапевтические интервенции и их эффект
- Общие наблюдения по процессу

Отвечай на русском языке, структурированно, 2-4 абзаца.\
"""

CHUNK_SIZE = 30
CHUNK_OVERLAP = 3
SINGLE_REQUEST_THRESHOLD = 40


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    return f'{m:02d}:{s:02d}'


def _parse_json_array(text: str) -> Optional[list]:
    """Robustly parse a JSON array from LLM output."""
    text = text.strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Try truncating incomplete array
    idx = text.rfind('"')
    if idx > 0:
        truncated = text[:idx + 1] + ']'
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    logging.warning('LLM: failed to parse JSON array from: %s', text[:200])
    return None


def _get_client(opts) -> 'openai.OpenAI':
    import openai
    return openai.OpenAI(
        base_url=opts.llm_api_base,
        api_key='lm-studio',
        timeout=120.0,
        max_retries=0,  # handled manually in _call_llm
    )


def _call_llm(client, model: str, messages: list, attempt: int = 0) -> Optional[str]:
    """Call LLM with one retry on network error."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content or ''
    except Exception as exc:
        if attempt == 0:
            logging.warning('LLM request failed (%s), retrying...', exc)
            return _call_llm(client, model, messages, attempt=1)
        logging.error('LLM request failed after retry: %s', exc)
        return None


def _validate_roles_chunk(
    segments: List[Replica],
    client,
    model: str,
) -> Optional[List[str]]:
    """Call LLM to validate roles for a chunk of segments. Returns list of role labels."""
    n = len(segments)
    lines = [f'{i} [{_format_timestamp(s.start)}] predicted={s.role}: "{s.text}"'
             for i, s in enumerate(segments)]
    user_msg = f'{n} фрагментов транскрипта:\n' + '\n'.join(lines)

    messages = [
        {'role': 'system', 'content': ROLE_VALIDATION_SYSTEM},
        {'role': 'user', 'content': FEW_SHOT_USER},
        {'role': 'assistant', 'content': FEW_SHOT_ASSISTANT},
        {'role': 'user', 'content': user_msg},
    ]

    raw = _call_llm(client, model, messages)
    if raw is None:
        return None

    parsed = _parse_json_array(raw)
    if parsed is None:
        return None

    labels = [str(item).upper() for item in parsed]
    # Normalize labels
    labels = ['Т' if l in ('Т', 'T') else 'К' for l in labels]

    if len(labels) != n:
        logging.warning(
            'LLM role validation: expected %d labels, got %d — using fallback', n, len(labels),
        )
        return None

    return labels


def _postprocess_labels(segments: List[Replica], labels: List[str]) -> List[str]:
    """Keyword-based correction: flip К→Т if therapist marker found."""
    result = list(labels)
    for i, (seg, label) in enumerate(zip(segments, labels)):
        if label == 'К':
            text_lower = seg.text.lower()
            for marker in THERAPIST_MARKERS:
                if marker in text_lower:
                    result[i] = 'Т'
                    break
    return result


def _validate_roles(segments: List[Replica], opts) -> List[Replica]:
    """Task 1: validate and correct role assignments."""
    client = _get_client(opts)
    model = opts.llm_model
    n = len(segments)

    if n <= SINGLE_REQUEST_THRESHOLD:
        chunks = [(0, n)]
    else:
        chunks = []
        i = 0
        while i < n:
            end = min(i + CHUNK_SIZE, n)
            chunks.append((i, end))
            if end >= n:
                break
            i += CHUNK_SIZE - CHUNK_OVERLAP

    all_labels: List[str] = []
    for chunk_idx, (start, end) in enumerate(chunks):
        chunk = segments[start:end]
        labels = _validate_roles_chunk(chunk, client, model)

        if labels is None:
            logging.warning('LLM: chunk %d role validation failed, using predicted roles', chunk_idx)
            labels = [s.role for s in chunk]

        labels = _postprocess_labels(chunk, labels)

        skip = CHUNK_OVERLAP if chunk_idx > 0 else 0
        all_labels.extend(labels[skip:])

    # Adjust for any length mismatch due to chunking
    if len(all_labels) != n:
        logging.warning(
            'LLM: total labels %d != segments %d, padding with predicted roles', len(all_labels), n,
        )
        while len(all_labels) < n:
            all_labels.append(segments[len(all_labels)].role)
        all_labels = all_labels[:n]

    result = []
    for seg, label in zip(segments, all_labels):
        result.append(Replica(
            speaker=seg.speaker,
            role=label,
            text=seg.text,
            start=seg.start,
            end=seg.end,
        ))
    return result


def _correct_text(segments: List[Replica], opts) -> List[Replica]:
    """Task 2: ASR error correction for text."""
    client = _get_client(opts)
    model = opts.llm_model
    CHUNK = 15

    corrected_texts: List[str] = []
    for i in range(0, len(segments), CHUNK):
        chunk = segments[i:i + CHUNK]
        texts = [s.text for s in chunk]
        user_msg = json.dumps(texts, ensure_ascii=False)
        messages = [
            {'role': 'system', 'content': TEXT_CORRECTION_SYSTEM},
            {'role': 'user', 'content': user_msg},
        ]
        raw = _call_llm(client, model, messages)
        if raw is None:
            corrected_texts.extend(texts)
            continue

        parsed = _parse_json_array(raw)
        if parsed is None or len(parsed) != len(chunk):
            logging.warning('LLM text correction: parse failed or length mismatch, using original')
            corrected_texts.extend(texts)
        else:
            corrected_texts.extend(str(t) for t in parsed)

    result = []
    for seg, text in zip(segments, corrected_texts):
        result.append(Replica(
            speaker=seg.speaker,
            role=seg.role,
            text=text,
            start=seg.start,
            end=seg.end,
        ))
    return result


def _analyze_session(segments: List[Replica], opts) -> Optional[str]:
    """Task 3: session analysis."""
    client = _get_client(opts)
    model = opts.llm_model

    dialogue_lines = [f'{s.role}: {s.text}' for s in segments]
    dialogue_text = '\n'.join(dialogue_lines)

    messages = [
        {'role': 'system', 'content': ANALYSIS_SYSTEM},
        {'role': 'user', 'content': dialogue_text},
    ]
    return _call_llm(client, model, messages)


def run_llm(alignment_result: AlignmentResult, opts) -> LlmResult:
    """Stage 4: LLM enhancement — role correction, text correction, analysis."""
    tasks = set(opts.llm_tasks)
    segments = alignment_result.segments

    if 'role_validation' in tasks:
        logging.info('LLM: validating roles for %d segments', len(segments))
        try:
            segments = _validate_roles(segments, opts)
            logging.info('LLM: role validation done')
        except Exception:
            logging.exception('LLM: role validation failed, using predicted roles')

    if 'text_correction' in tasks:
        logging.info('LLM: correcting text for %d segments', len(segments))
        try:
            segments = _correct_text(segments, opts)
            logging.info('LLM: text correction done')
        except Exception:
            logging.exception('LLM: text correction failed, using original text')

    analysis: Optional[str] = None
    if 'analysis' in tasks:
        logging.info('LLM: analyzing session')
        try:
            analysis = _analyze_session(segments, opts)
            logging.info('LLM: analysis done (%d chars)', len(analysis) if analysis else 0)
        except Exception:
            logging.exception('LLM: analysis failed')

    return LlmResult(segments=segments, analysis=analysis)
