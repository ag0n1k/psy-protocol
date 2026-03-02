import json
import logging
import re
from typing import Dict, List, Tuple

from .config import DEFAULT_LLM_DIARIZATION_MODEL
from .diarization import SpeakerSegment

_cached_model = None
_cached_tokenizer = None
_cached_model_path = None

SYSTEM_PROMPT = (
    'Ты — эксперт по диаризации транскриптов психотерапевтических сессий '
    '(гештальт-терапия). В сессии участвуют двое: К (клиент) и Т (терапевт).\n'
    '\n'
    'Тебе дан список пронумерованных фрагментов с таймкодами. '
    'Определи для каждого фрагмента говорящего: К или Т.\n'
    '\n'
    '## Характерные маркеры терапевта (Т)\n'
    '\n'
    'Терапевт делится своими внутренними ощущениями от контакта с клиентом '
    '(контрперенос):\n'
    '- «Знаешь, я сейчас замечаю...», «Я внутри себя ловлю...»\n'
    '- «Я себя почувствовал...», «Я чувствую...», «Я вижу, что ты...»\n'
    '- «У меня такое ощущение что...», «Меня трогает/удивляет...»\n'
    '\n'
    'Терапевт задаёт вопросы и предлагает эксперименты:\n'
    '- «Что для тебя здесь означает?», «Как ты себя сейчас чувствуешь?»\n'
    '- «Можешь рассказать поподробнее?», «Можешь описать?»\n'
    '- «Давай попробуем...», «Я предлагаю...», «Можешь побыть в этом?»\n'
    '- «Может быть в теле что-то?», «Где это находится?»\n'
    '\n'
    'Терапевт наблюдает за поведением клиента:\n'
    '- «У тебя голос повысился», «Ты сейчас чуть задрожала»\n'
    '- «Я замечаю что ты зеваешь/сжимаешься/улыбаешься»\n'
    '- «Вот сейчас у тебя слёзы наворачиваются»\n'
    '\n'
    'Терапевт управляет структурой сессии:\n'
    '- «Я предлагаю здесь остановиться», «Давай вернёмся к...»\n'
    '- «Что для тебя было новым?», «Что-то изменилось?»\n'
    '\n'
    'Терапевт часто начинает реплику со слова «Знаешь».\n'
    '\n'
    '## Характерные маркеры клиента (К)\n'
    '\n'
    'Клиент рассказывает личные истории и переживания:\n'
    '- Длинные нарративы о ситуациях из жизни, отношениях, работе\n'
    '- «Я испытываю...», «У меня появляется...», «Мне страшно/больно...»\n'
    '- «Какая-то злость/грусть/напряжение поднимается»\n'
    '- «Как будто бы я...», «Я не знаю...», «Я боюсь если...»\n'
    '\n'
    'Клиент отвечает на вопросы терапевта:\n'
    '- Развёрнутые ответы после вопросов Т\n'
    '- Короткие подтверждения «Да», «Угу», «Нет» в ответ на уточнения Т\n'
    '\n'
    'Клиент говорит значительно больше по объёму (60-80% текста).\n'
    '\n'
    '## Правила работы с фрагментами\n'
    '\n'
    '1. ВАЖНО: соседние фрагменты чаще всего принадлежат одному говорящему. '
    'Смена спикера происходит редко — примерно 20-40 раз за сессию, '
    'а фрагментов может быть 100-200. Не меняй спикера без причины.\n'
    '2. Очень короткие фрагменты (1-3 слова) почти всегда продолжают '
    'предыдущий фрагмент того же спикера.\n'
    '3. После вопроса Т обычно следует ответ К, и наоборот.\n'
    '4. Сессия обычно начинается с К (клиент озвучивает тему).\n'
    '5. В конце сессии Т обычно подводит итоги, затем оба говорят «Спасибо».\n'
    '\n'
    'Верни ТОЛЬКО JSON-массив из строк "К" или "Т". Без пояснений.'
)

VALID_LABELS = {'К', 'Т'}

# фразы-маркеры терапевта: если блок содержит одну из них, скорее всего это Т
THERAPIST_MARKERS = [
    'что для тебя',
    'как ты себя',
    'как ты сейчас',
    'можешь рассказать',
    'можешь описать',
    'можешь побыть',
    'можешь сфокусироваться',
    'давай попробуем',
    'давай попробуй',
    'я предлагаю',
    'предлагаю остановиться',
    'предлагаю здесь',
    'я замечаю',
    'я заметил',
    'я сейчас заметил',
    'я чувствую',
    'я себя почувствовал',
    'я внутри себя ловлю',
    'я вижу что ты',
    'у тебя голос',
    'ты сейчас чуть',
    'меня трогает',
    'меня удивляет',
    'может быть в теле',
    'что-то изменилось',
    'что было новым',
]


def _load_llm(model_path: str) -> Tuple:
    global _cached_model, _cached_tokenizer, _cached_model_path
    if _cached_model_path != model_path:
        from mlx_lm import load
        logging.info('LLM diarization: loading model %s', model_path)
        _cached_model, _cached_tokenizer = load(model_path)
        _cached_model_path = model_path
    return _cached_model, _cached_tokenizer


FEW_SHOT_USER = (
    'Фрагменты транскрипта (12 шт.):\n'
    '\n'
    '0 [00:01]: "привет"\n'
    '1 [00:03]: "привет я хотела тебе рассказать как я плакала навзрыд '
    'такое довольно редко бывает может быть раз в пятнадцатилетие"\n'
    '2 [00:18]: "и последний раз я плакала навзрыд когда я вышла замуж '
    'а тут мне лечили зуб и его сломали и его надо было вырывать"\n'
    '3 [00:35]: "и он начал тянуть зуб и я его остановила и спросила '
    'почему он так себя ведет и он сказал надо вырвать зуб это силовая процедура"\n'
    '4 [00:52]: "и он мне сделал замечание он сказал что у вас сильная анестезия '
    'и надо проявить терпение я сказала окей и у меня просто навзрыд полились слёзы"\n'
    '5 [01:10]: "знаешь два момента я отметил в твоей речи когда у тебя '
    'прям порывались слёзы это когда доктор сказал что нужно проявить терпение"\n'
    '6 [01:25]: "и второе это когда директивно без ответа тебя оставили '
    'и сказали идите на снимок что тебе больше здесь внутри"\n'
    '7 [01:38]: "что идите что без ответа"\n'
    '8 [01:42]: "вот когда я тебе говорю идите на снимок '
    'что-то в теле ты чувствуешь где-то"\n'
    '9 [01:50]: "да у меня вообще всё от живота наверх парализуется"\n'
    '10 [01:56]: "можешь сейчас сфокусироваться на этом парализовании '
    'я вижу что ты так схватилась в кресло"\n'
    '11 [02:08]: "ну да я челюсть сжала сильно как будто бы я вся деревянная"\n'
    '\n'
    'JSON:'
)

FEW_SHOT_ASSISTANT = '["Т", "К", "К", "К", "К", "Т", "Т", "К", "Т", "К", "Т", "К"]'


def _format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f'{m:02d}:{s:02d}'


def build_prompt(segments: List[Dict]) -> List[Dict[str, str]]:
    """Формирует chat messages для LLM из Whisper-сегментов."""
    lines = []
    for idx, seg in enumerate(segments):
        text = seg.get('text', '').strip()
        ts = _format_timestamp(seg.get('start', 0.0))
        lines.append(f'{idx} [{ts}]: "{text}"')

    user_content = (
        f'Фрагменты транскрипта ({len(segments)} шт.):\n'
        '\n'
        + '\n'.join(lines)
        + '\n\nJSON:'
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': FEW_SHOT_USER},
        {'role': 'assistant', 'content': FEW_SHOT_ASSISTANT},
        {'role': 'user', 'content': user_content},
    ]


def parse_response(response: str, num_segments: int) -> List[str]:
    """Извлекает JSON-массив ['К', 'Т', ...] из ответа LLM."""
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        raw = match.group()
    else:
        # LLM может не закрыть массив при обрезке по max_tokens
        bracket_pos = response.find('[')
        if bracket_pos == -1:
            raise ValueError(
                f'No JSON array found in LLM response (len={len(response)})'
            )
        raw = response[bracket_pos:].rstrip().rstrip(',') + ']'

    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        # обрезанный последний элемент — убираем и закрываем
        cleaned = re.sub(r',\s*"[^"]*$', '', raw.rstrip(']')).rstrip(',') + ']'
        try:
            labels = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f'Invalid JSON in LLM response: {exc}'
            ) from exc

    if not isinstance(labels, list):
        raise ValueError(f'Expected JSON array, got {type(labels).__name__}')

    # LLM может выдать больше элементов, чем сегментов — обрезаем
    if len(labels) > num_segments:
        labels = labels[:num_segments]

    if len(labels) < num_segments:
        deficit = num_segments - len(labels)
        if deficit <= max(3, num_segments // 10) and labels:
            pad = labels[-1]
            logging.warning(
                'LLM returned %d labels, expected %d, padding with %r',
                len(labels), num_segments, pad,
            )
            labels.extend([pad] * deficit)
        else:
            raise ValueError(
                f'Not enough labels: got {len(labels)}, expected {num_segments}'
            )

    for idx, label in enumerate(labels):
        if label not in VALID_LABELS:
            raise ValueError(
                f'Invalid label at index {idx}: {label!r} (expected "К" or "Т")'
            )

    return labels


def _postprocess_labels(blocks: List[Dict], labels: List[str]) -> List[str]:
    """Корректирует метки на основе keyword-маркеров терапевта."""
    result = list(labels)
    flipped = 0
    for i, (block, label) in enumerate(zip(blocks, labels)):
        if label == 'К':
            text_lower = block['text'].lower()
            for marker in THERAPIST_MARKERS:
                if marker in text_lower:
                    result[i] = 'Т'
                    flipped += 1
                    break
    if flipped:
        logging.info('LLM postprocess: flipped %d blocks К→Т by markers', flipped)
    return result


def _generate_llm(model, tokenizer, prompt_text: str, max_tokens: int) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    return generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),
    )


def diarize_with_llm(
    whisper_segments: List[Dict],
    model_path: str = DEFAULT_LLM_DIARIZATION_MODEL,
) -> List[SpeakerSegment]:
    """Основная функция: отправляет raw Whisper-сегменты в LLM для диаризации."""
    logging.info('LLM diarization: %d segments', len(whisper_segments))

    model, tokenizer = _load_llm(model_path)
    messages = build_prompt(whisper_segments)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    max_tokens = max(len(whisper_segments) * 12, 256)
    logging.info('LLM diarization: max_tokens=%d', max_tokens)

    response = _generate_llm(model, tokenizer, prompt_text, max_tokens)
    logging.info(
        'LLM diarization response (len=%d, last 50 chars): ...%s',
        len(response),
        response[-50:],
    )

    labels = parse_response(response, len(whisper_segments))
    labels = _postprocess_labels(whisper_segments, labels)

    result = []
    for seg, label in zip(whisper_segments, labels):
        result.append(SpeakerSegment(
            start=float(seg.get('start', 0.0)),
            end=float(seg.get('end', 0.0)),
            speaker=label,
        ))

    return result
