import re
from typing import List


def remove_repetitions(text: str, max_repeats: int = 3) -> str:
    """'да да да да да да' -> 'да да да'."""
    words = text.split()
    if not words:
        return text
    result: List[str] = [words[0]]
    repeat_count = 1
    for word in words[1:]:
        if word.lower() == result[-1].lower():
            repeat_count += 1
            if repeat_count <= max_repeats:
                result.append(word)
        else:
            repeat_count = 1
            result.append(word)
    return ' '.join(result)


_GHOST_WORD_PATTERN = re.compile(
    r'\b(?:Straight|Click|Subscribe|Hello|Yeah|Yes|Okay|OK|Please'
    r'|Thank you|Thanks|Welcome|Bye|Sorry|Right|Well|So|Like'
    r'|Actually|Basically|Literally|Exactly|Absolutely)\b',
    re.IGNORECASE,
)


def remove_ghost_words(text: str, lang: str = 'ru') -> str:
    """Remove English ghost words from Russian text."""
    if lang != 'ru':
        return text
    has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', text))
    if not has_cyrillic:
        return text
    cleaned = _GHOST_WORD_PATTERN.sub('', text)
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    return cleaned


_FILLER_PATTERN = re.compile(
    r'((?:^|\.\s*)([А-Яа-яёЁA-Za-z]+)\.)'
    r'(?:\s+\2\.){2,}',
    re.IGNORECASE,
)


def clean_filler_runs(text: str) -> str:
    """'Угу. Угу. Угу. Угу.' -> 'Угу.'."""
    words = re.split(r'(?<=\.)\s+', text)
    if len(words) < 2:
        return text

    result: List[str] = [words[0]]
    repeat_count = 1
    for word in words[1:]:
        if word.rstrip('.').lower() == result[-1].rstrip('.').lower():
            repeat_count += 1
            if repeat_count <= 1:
                result.append(word)
        else:
            repeat_count = 1
            result.append(word)
    return ' '.join(result)


def postprocess_replica_text(text: str) -> str:
    """Apply all text filters sequentially."""
    text = remove_repetitions(text)
    text = remove_ghost_words(text)
    text = clean_filler_runs(text)
    text = re.sub(r' {2,}', ' ', text).strip()
    return text
