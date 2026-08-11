# Рефакторинг pipeline в стадийную архитектуру + LLM стадия (Qwen3.5)

## Context

Текущий `pipeline.py` (447 строк) — монолит, где все стадии обработки смешаны в одной функции `process_audio_file()`. Невозможно запустить отдельную стадию, протестировать её результат и перейти к следующей. Нужно чёткое разделение на независимые стадии. Также нужно добавить стадию 4 — LLM-улучшение через Qwen3.5-9B (LMStudio OpenAI API).

## Архитектура стадий

```
Stage 0: Preprocessing (audio → 16kHz WAV)
Stage 1: ASR (audio → текст + сегменты)
Stage 2: Diarization (audio → спикер-сегменты)
Stage 3a: Alignment (ASR + diarization → НЕСКЛЕЕННЫЕ сегменты с predicted-ролями)
Stage 4: LLM Enhancement (сегменты → исправленные роли + merge-инструкции + анализ)
Stage 3b: Merge + Post-processing (склейка соседних + text cleanup)
Stage 5: Output (реплики → DOCX + TXT)
```

**Критично:** слияние (merge) соседних сегментов одного спикера происходит ПОСЛЕ LLM.
Иначе если диаризация ошибочно дала один speaker двум разным людям, их тексты
склеятся в одну реплику и LLM не сможет их разделить.

**Без LLM:** стадия 3a сразу переходит в 3b (merge), как сейчас. Поведение не меняется.

**Qwen-ASR** остаётся combined path (стадии 1+2+3a объединены): diarize → transcribe
per segment → несклеенные реплики. Merge происходит после LLM (или сразу, если LLM отключена).

## Новая файловая структура

```
psy_protocol/
    stages/
        __init__.py          # экспорты: run_asr, run_diarization, run_alignment, run_llm, run_output
        asr.py               # Stage 1: ASR orchestration + caching
        diarization_stage.py # Stage 2: diarization orchestration + caching
        alignment_stage.py   # Stage 3: alignment + text postprocess + role mapping
        llm_stage.py         # Stage 4: LLM enhancement (openai SDK → LMStudio)
        output_stage.py      # Stage 5: DOCX + TXT generation
    models.py                # dataclasses: AsrResult, DiarizationResult, AlignmentResult, LlmResult
    pipeline.py              # SLIM: тонкий оркестратор, вызывает stages
    cli.py                   # REFACTORED: subcommands для отдельных стадий
    config.py                # + LLM defaults
    # остальные модули без изменений
```

## Промежуточные форматы (models.py)

```python
@dataclass
class AsrSegment:
    start: float
    end: float
    text: str

@dataclass
class AsrWord:
    start: float
    end: float
    word: str
    probability: float = 1.0

@dataclass
class AsrResult:
    text: str
    segments: list[AsrSegment]
    words: list[AsrWord]       # может быть пустым (qwen_asr не даёт слов)
    method: str                # 'whisper' | 'qwen_asr'
    model: str

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> 'AsrResult': ...

@dataclass
class DiarizationResult:
    segments: list[SpeakerSegment]
    method: str
    params: dict[str, Any]

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> 'DiarizationResult': ...

@dataclass
class Replica:
    speaker: str
    role: str          # 'К' | 'Т'
    text: str
    start: float
    end: float

@dataclass
class AlignmentResult:
    """Результат Stage 3a — НЕСКЛЕЕННЫЕ сегменты с predicted-ролями."""
    segments: list[Replica]    # каждый whisper-сегмент / qwen-сегмент отдельно

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> 'AlignmentResult': ...

@dataclass
class LlmResult:
    segments: list[Replica]    # сегменты с исправленными ролями (до merge)
    analysis: str | None = None

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> 'LlmResult': ...
```

**Ключевое:** `AlignmentResult.segments` — это НЕСКЛЕЕННЫЕ сегменты (каждый whisper-сегмент
или qwen per-diarization сегмент отдельно). Слияние соседних сегментов одной роли в реплики
происходит ПОЗЖЕ — в Stage 3b (после LLM или сразу, если LLM отключена).

## Детали реализации по стадиям

### Stage 1: `stages/asr.py`

Извлекаем из `pipeline.py` строки 316-400 (whisper path transcription logic).

```python
def run_asr(audio_path: Path, opts: ProcessingOptions, cache_dir: Path, emit: Callable) -> AsrResult:
```
- Проверяет кэш (`transcript.json`, `transcript_meta.json`)
- Dispatches to `whisper_transcribe.transcribe_audio()` or `qwen_transcribe.transcribe_audio_qwen()`
- Сохраняет кэш
- Возвращает `AsrResult`

### Stage 2: `stages/diarization_stage.py`

Извлекаем из `pipeline.py` функции `_run_diarization()` (строки 189-207) и `_run_mlx_diarization()` (строки 103-186).

```python
def run_diarization(audio_path: Path, opts: ProcessingOptions, cache_dir: Path, emit: Callable) -> DiarizationResult:
```
- Код переносится as-is, только возвращает `DiarizationResult`
- Вспомогательные `_serialize_segments`, `_deserialize_segments`, `_is_diarization_cache_valid` переезжают сюда

### Stage 3a: `stages/alignment_stage.py` — Alignment (БЕЗ слияния)

Извлекаем из `pipeline.py` + модифицируем `alignment.py`.

```python
def run_alignment(
    asr_result: AsrResult,
    diarization_result: DiarizationResult,
    opts: ProcessingOptions,
) -> AlignmentResult:
```

**Whisper path:**
- Вызывает `assign_speakers_to_segments()` (или `assign_speakers_to_spans()` для слов)
- Применяет `smooth_word_speakers()` если есть word timestamps
- НО НЕ склеивает (не вызывает `build_replicas()`)
- Каждый whisper-сегмент получает speaker и predicted role
- Применяет `postprocess_replica_text()` к каждому сегменту
- Применяет `parse_speaker_map()` + `map_speakers_to_roles()`
- Возвращает `AlignmentResult` со списком несклеенных сегментов

**Нужна новая функция в `alignment.py`:**
```python
def assign_segments_without_merge(
    whisper_segments: list[dict],
    diarization_segments: list[SpeakerSegment],
) -> list[dict]:
    """Как build_replicas(), но без слияния соседних."""
    speakers = assign_speakers_to_segments(whisper_segments, diarization_segments)
    result = []
    for seg, speaker in zip(whisper_segments, speakers):
        text = seg.get('text', '').strip()
        if not text:
            continue
        result.append({
            'speaker': speaker,
            'text': text,
            'start': float(seg.get('start', 0.0)),
            'end': float(seg.get('end', 0.0)),
        })
    return result
```

Аналогично для word timestamps:
```python
def assign_words_without_merge(
    words: list[dict],
    diarization_segments: list[SpeakerSegment],
    smooth_min_words: int,
) -> list[dict]:
    """Как build_replicas_from_words(), но группирует слова по whisper-сегментам, не по спикерам."""
    # Группировка: слова → whisper-сегменты (по timestamps)
    # Каждый сегмент получает majority-speaker из его слов
```

**Qwen-ASR combined path:**
```python
def run_qwen_combined(
    audio_path: Path,
    opts: ProcessingOptions,
    cache_dir: Path,
    emit: Callable,
) -> AlignmentResult:
```
- Вызывает `run_diarization()` + `transcribe_per_diarization()`
- Каждый per-diarization сегмент уже отдельный → НЕ склеиваем
- Применяет `postprocess_replica_text()` + role mapping
- Возвращает `AlignmentResult` с несклеенными сегментами

### Stage 3b: Merge (`stages/alignment_stage.py` — вторая функция)

```python
def merge_segments_to_replicas(segments: list[Replica]) -> list[Replica]:
    """Склеивает соседние сегменты с одинаковой ролью в реплики."""
```
- Логика из текущего `build_replicas()` строки 62-71
- И из `merge_adjacent_by_role()` в `replica_postprocess.py`
- Вызывается ПОСЛЕ LLM (или сразу после 3a, если LLM отключена)

### Stage 4: `stages/llm_stage.py` (NEW)

```python
def run_llm(alignment_result: AlignmentResult, opts: ProcessingOptions) -> LlmResult:
```

**Подключение:** OpenAI SDK → LMStudio `http://localhost:1234/v1`

**Зависимость:** `openai` Python package (добавить в requirements.txt)

---

#### Формат ввода для LLM (одинаковый для Whisper и Qwen-ASR путей)

На вход LLM приходит `AlignmentResult.segments` — список НЕСКЛЕЕННЫХ сегментов
с predicted-ролями от диаризации. Каждый сегмент = 1 whisper-сегмент (~5-30 сек)
или 1 qwen per-diarization сегмент.

Это критично: сегменты НЕ склеены, поэтому LLM видит границы между фрагментами
и может правильно присвоить роли даже когда диаризация перепутала спикеров.

Формат сегментов в промпте:
```
Фрагменты транскрипта (35 шт.):
0 [00:13] predicted=К: "я испытываю раздражение на свою какую-то неспособность..."
1 [00:59] predicted=Т: "ну"
2 [01:00] predicted=К: "знаешь я сначала удивился потому что ты говорила..."
3 [01:20] predicted=Т: "да"
4 [01:22] predicted=К: "какая-то дисфункция как будто я должна это..."
...
```

Формат ответа LLM — JSON-массив строк (как в удалённом llm_diarization.py):
```json
["К", "К", "Т", "Т", "К", ...]
```

Длина массива = количество входных фрагментов. Слияние соседних с одинаковой ролью
происходит автоматически в Stage 3b после LLM — не нужен `merge_with_next`.

#### Проблема: почему 20-30% точности

Сравнение реального output с goal показывает два основных типа ошибок:

1. **Перепутаны спикеры** — k-means кластеризация присвоила SPEAKER_00 не тому.
   Фразы терапевта ("Знаешь, я сначала удивился...") попадают в К.
   Это **глобальная** ошибка — весь маппинг К↔Т инвертирован.

2. **Фрагментация** — диаризация разбивает длинные высказывания на куски,
   между которыми вставляет мусорные короткие реплики ("Ну.", "Да.", "Ой.")
   от другого спикера. Это нарушает структуру диалога.

LLM должна справиться с обоими типами ошибок по смыслу фраз.

#### Три задачи LLM (последовательно)

**Задача 1: Валидация ролей К/Т** (`_validate_roles`)

System prompt (на основе удалённого `llm_diarization.py`, адаптирован):
- Описание контекста: гештальт-терапия, двое участников
- Маркеры терапевта: контрперенос ("я замечаю...", "я чувствую..."),
  вопросы ("что для тебя...", "можешь рассказать..."),
  наблюдения за клиентом ("у тебя голос повысился..."),
  управление сессией ("предлагаю остановиться...")
- Маркеры клиента: личные истории, переживания, ответы на вопросы, больше по объёму
- Правила: соседние фрагменты чаще одного спикера, короткие фрагменты — продолжение,
  после вопроса Т следует ответ К
- Вход: несклеенные сегменты с predicted-ролями (формат выше)
- Выход: JSON-массив строк ["К", "Т", ...] той же длины

Чанкование: по ~30 сегментов с перекрытием в 3 сегмента для контекста на стыках.
Если сегментов < 40, отправляем одним запросом.

Few-shot пример (из удалённого кода, адаптирован):
```
User: 12 фрагментов с predicted ролями...
Assistant: ["К", "К", "Т", "Т", "К", "Т", "Т", "К", "Т", "К", "Т", "К"]
```

Keyword постпроцессинг (из удалённого `_postprocess_labels`):
После LLM дополнительно проверяем THERAPIST_MARKERS — если реплика с ролью К
содержит маркер терапевта, флипаем в Т.

**Задача 2: Исправление текста** (`_correct_text`)

System prompt:
- Исправь ошибки ASR: пунктуация, опечатки, границы предложений
- НЕ меняй смысл, НЕ добавляй слова, НЕ удаляй слова
- Вход: реплики с текстом
- Выход: JSON-массив строк с исправленным текстом

Чанкование: по ~15 реплик (текст длиннее, чем метки).

**Задача 3: Анализ сессии** (`_analyze_session`)

System prompt:
- Краткое резюме сессии: ключевые темы, динамика, наблюдения
- Вход: полный диалог (К: текст, Т: текст)
- Выход: текст анализа (не JSON)

#### Робастность

- Парсинг JSON: regex `\[.*\]`, fallback на обрезку неполных массивов (из удалённого `parse_response`)
- При ошибке парсинга или несовпадении длины — fallback на исходные данные с warning
- Timeout на API-запрос: 120 секунд
- Retry: 1 повторная попытка при сетевой ошибке

#### Переиспользуемый код из удалённого `llm_diarization.py`

- `SYSTEM_PROMPT` — адаптировать маркеры терапевта/клиента
- `THERAPIST_MARKERS` — список keyword-маркеров для постпроцессинга
- `parse_response()` — робастный парсинг JSON из LLM ответа
- `_postprocess_labels()` — keyword-коррекция после LLM
- `FEW_SHOT_USER` / `FEW_SHOT_ASSISTANT` — адаптировать few-shot примеры
- `_format_timestamp()` — форматирование таймкодов

### Stage 5: `stages/output_stage.py`

Извлекаем из `pipeline.py` строки 413-446 (docx + txt generation).

```python
def run_output(
    replicas: list[Replica],
    opts: ProcessingOptions,
    audio_path: Path,
    cache_dir: Path,
) -> tuple[Path, Path]:
```

### Рефакторинг `pipeline.py`

Становится тонким оркестратором (~80 строк):

```python
def process_audio_file(audio_path, options, progress_callback):
    # Stage 0: preprocessing
    processed = maybe_preprocess(audio_path, ...)

    # Stages 1-3a: получаем НЕСКЛЕЕННЫЕ сегменты с predicted-ролями
    if options.transcription_method == 'qwen_asr':
        alignment = run_qwen_combined(processed, options, cache_dir, emit)
    else:
        asr = run_asr(processed, options, cache_dir, emit)
        diar = run_diarization(processed, options, cache_dir, emit)
        alignment = run_alignment(asr, diar, options)

    # Stage 4: LLM исправляет роли НА УРОВНЕ СЕГМЕНТОВ (до склейки)
    if options.llm_enabled:
        llm_result = run_llm(alignment, options)
        segments = llm_result.segments
    else:
        segments = alignment.segments

    # Stage 3b: ТЕПЕРЬ склеиваем соседние сегменты одной роли
    replicas = merge_segments_to_replicas(segments)

    # Stage 5: output
    return run_output(replicas, options, ...)
```

**Обратная совместимость:** `ProcessingOptions` сохраняет flat-поля + добавляем LLM-поля. `bot.py` и `test.py` продолжают работать без изменений.

### Рефакторинг `cli.py`

Добавляем subcommands через `argparse` subparsers:

```
psy-protocol full --audio file.ogg [все опции]     # полный pipeline (default)
psy-protocol asr --audio file.ogg [asr опции]      # только Stage 1
psy-protocol diarize --audio file.ogg [diar опции]  # только Stage 2
psy-protocol align --cache-dir path [align опции]    # Stage 3 (из кэша)
psy-protocol llm --cache-dir path [llm опции]        # Stage 4 (из кэша)
```

Без subcommand → `full` (backward compatible).

### Изменения в `config.py`

```python
DEFAULT_LLM_ENABLED = False
DEFAULT_LLM_API_BASE = 'http://localhost:1234/v1'
DEFAULT_LLM_MODEL = 'qwen3.5-9b'
DEFAULT_LLM_TASKS = ['role_validation', 'text_correction', 'analysis']
```

### Изменения в `ProcessingOptions`

Добавляем поля:
```python
llm_enabled: bool = DEFAULT_LLM_ENABLED
llm_api_base: str = DEFAULT_LLM_API_BASE
llm_model: str = DEFAULT_LLM_MODEL
llm_tasks: list[str] = field(default_factory=lambda: list(DEFAULT_LLM_TASKS))
```

## Порядок реализации

1. **models.py** — dataclasses для промежуточных результатов
2. **stages/__init__.py** — пустой пакет
3. **stages/diarization_stage.py** — перенос из pipeline.py (самый автономный)
4. **stages/asr.py** — перенос ASR логики
5. **stages/alignment_stage.py** — перенос alignment + qwen combined
6. **stages/output_stage.py** — перенос output логики
7. **pipeline.py** — рефакторинг в тонкий оркестратор
8. **config.py** — добавление LLM defaults
9. **stages/llm_stage.py** — новый LLM модуль
10. **cli.py** — добавление subcommands
11. **requirements.txt** — добавить `openai`
12. Проверить `bot.py` и `test.py` на совместимость

## Промежуточные файлы на диске

Каждая стадия сохраняет результат в `transcripts/<audio_stem>/`:

| Файл | Стадия | Содержимое |
|------|--------|------------|
| `asr_result.json` | 1 | AsrResult (текст, сегменты, слова) |
| `diarization.json` | 2 | Raw segments (существующий) |
| `diarization_post.json` | 2 | Post-processed segments (существующий) |
| `alignment_result.json` | 3a | AlignmentResult (НЕСКЛЕЕННЫЕ сегменты с predicted-ролями) |
| `llm_result.json` | 4 | LlmResult (сегменты с исправленными ролями + анализ) |
| Существующие файлы | — | transcript.json, whisper_segments.json и т.д. сохраняются для совместимости |

## Верификация

1. `python main.py --audio 1.ogg` — полный pipeline работает как раньше
2. `python main.py asr --audio 1.ogg` — только ASR, результат в `transcripts/1/asr_result.json`
3. `python main.py diarize --audio 1.ogg` — только диаризация
4. `python main.py llm --cache-dir transcripts/1` — LLM на существующих результатах
5. `python test.py --config default --test 1` — тест проходит
6. `python bot.py` — бот работает без изменений
