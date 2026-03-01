from pathlib import Path
from typing import Dict, List


def save_dialogue_txt(path: Path, replicas: List[Dict[str, str]]) -> None:
    lines = [f"{replica['role']}: {replica['text']}" for replica in replicas]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_timed_dialogue_txt(path: Path, replicas: List[Dict]) -> None:
    """Save dialogue with timestamps: К [00:05]: text."""
    lines = []
    for replica in replicas:
        start = float(replica.get('start', 0.0))
        total_sec = int(start)
        m, s = divmod(total_sec, 60)
        ts = f'[{m:02d}:{s:02d}]'
        role = replica.get('role', 'К')
        text = replica.get('text', '')
        lines.append(f'{role} {ts}: {text}')
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def save_sentences_txt(path: Path, whisper_segments: List[Dict]) -> None:
    lines = []
    for seg in whisper_segments:
        start = float(seg.get("start", 0.0))
        text = seg.get("text", "").strip()
        if not text:
            continue
        total_sec = int(start)
        m, s = divmod(total_sec, 60)
        ts = f"[{m:02d}:{s:02d}]"
        lines.append(f"{ts} {text}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
