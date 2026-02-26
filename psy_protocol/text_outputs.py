from pathlib import Path
from typing import Dict, List


def save_dialogue_txt(path: Path, replicas: List[Dict[str, str]]) -> None:
    lines = [f"{replica['role']}: {replica['text']}" for replica in replicas]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
