import re
from typing import Dict, List


REPLICA_RE = re.compile(r'^([КТ]):\s*(.*)')


def parse_dialogue_text(text: str) -> List[Dict[str, str]]:
    """Parse text with lines like 'К: ...' / 'Т: ...' into replica dicts."""
    replicas: List[Dict[str, str]] = []
    current_role = None
    current_lines: List[str] = []

    for line in text.splitlines():
        match = REPLICA_RE.match(line)
        if match:
            if current_role is not None and current_lines:
                replicas.append({
                    'role': current_role,
                    'text': ' '.join(current_lines),
                })
            current_role = match.group(1)
            first_text = match.group(2).strip()
            current_lines = [first_text] if first_text else []
        elif current_role is not None:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    if current_role is not None and current_lines:
        replicas.append({
            'role': current_role,
            'text': ' '.join(current_lines),
        })

    return replicas
