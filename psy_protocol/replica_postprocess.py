import copy
import json
import logging
import re
from typing import Dict, List

VALID_ROLES = {'К', 'Т'}

_cached_model = None
_cached_tokenizer = None
_cached_model_path = None


def _load_llm(model_path: str):
    global _cached_model, _cached_tokenizer, _cached_model_path
    if _cached_model_path != model_path:
        from mlx_lm import load
        logging.info('Role validation: loading model %s', model_path)
        _cached_model, _cached_tokenizer = load(model_path)
        _cached_model_path = model_path
    return _cached_model, _cached_tokenizer


def _format_timestamp(seconds: float) -> str:
    total = int(float(seconds))
    minutes, sec = divmod(total, 60)
    return f'{minutes:02d}:{sec:02d}'


def _build_prompt(replicas: List[Dict]) -> str:
    lines: List[str] = []
    for idx, replica in enumerate(replicas):
        role = replica.get('role', 'К')
        ts = _format_timestamp(replica.get('start', 0.0))
        text = replica.get('text', '').strip()
        lines.append(f'{idx} [{ts}] predicted={role}: "{text}"')

    return (
        'Ты валидируешь роли диалога психотерапевтической сессии.\n'
        'В сессии только два участника: К (клиент) и Т (терапевт).\n'
        'Ниже список реплик с текущей ролью predicted.\n'
        'Исправь только ошибочные роли и верни JSON-массив той же длины: ["К","Т",...].\n'
        'Без пояснений, без markdown, только JSON.\n\n'
        f'Реплики ({len(replicas)} шт.):\n'
        + '\n'.join(lines)
        + '\n\nJSON:'
    )


def _parse_labels(response: str, expected_size: int) -> List[str]:
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if not match:
        raise ValueError('Role validation: no JSON array in response')

    labels = json.loads(match.group())
    if not isinstance(labels, list):
        raise ValueError('Role validation: response is not a JSON array')
    if len(labels) != expected_size:
        raise ValueError(
            f'Role validation: label count mismatch ({len(labels)} != {expected_size})'
        )
    for idx, label in enumerate(labels):
        if label not in VALID_ROLES:
            raise ValueError(f'Role validation: invalid label {label!r} at {idx}')
    return labels


def validate_roles_with_llm(
    replicas: List[Dict],
    llm_model_path: str,
    enabled: bool = True,
) -> List[Dict]:
    """Validate per-replica roles with LLM. Returns updated replicas or original on failure."""
    if not enabled or not replicas:
        return replicas

    try:
        model, tokenizer = _load_llm(llm_model_path)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = _build_prompt(replicas)
        max_tokens = max(256, len(replicas) * 4)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        )
        labels = _parse_labels(response, len(replicas))
    except Exception:
        logging.exception('Role validation: LLM validation failed, fallback to baseline roles')
        return replicas

    validated = [copy.copy(replica) for replica in replicas]
    changed = 0
    for replica, label in zip(validated, labels):
        if replica.get('role') != label:
            changed += 1
        replica['role'] = label
    logging.info('Role validation: %d/%d labels adjusted', changed, len(validated))
    return validated


def merge_adjacent_by_role(replicas: List[Dict]) -> List[Dict]:
    """Merge neighboring replicas when they share the same role."""
    if not replicas:
        return replicas

    merged: List[Dict] = [copy.copy(replicas[0])]
    merged[0]['text'] = merged[0].get('text', '').strip()

    for replica in replicas[1:]:
        current = copy.copy(replica)
        current['text'] = current.get('text', '').strip()
        last = merged[-1]
        if current.get('role') == last.get('role'):
            left = last.get('text', '').strip()
            right = current.get('text', '').strip()
            last['text'] = ' '.join(part for part in [left, right] if part).strip()
            last['end'] = current.get('end', last.get('end'))
            continue
        merged.append(current)

    return merged
