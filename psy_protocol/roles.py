from typing import Dict, List, Optional


def parse_speaker_map(value: Optional[str]) -> Dict[str, str]:
    if not value:
        return {}
    mapping: Dict[str, str] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid speaker map entry: {chunk}")
        raw_key, raw_value = chunk.split("=", 1)
        key = raw_key.strip().upper()
        role = raw_value.strip().upper()
        if role in ("C", "К"):
            role = "К"
        elif role in ("T", "Т"):
            role = "Т"
        else:
            raise ValueError(f"Invalid role for {raw_key}: {raw_value}")
        if key.isdigit():
            key = f"SPEAKER_{int(key):02d}"
        if key.startswith("SPEAKER_"):
            key = key
        mapping[key] = role
    return mapping


def map_speakers_to_roles(
    replicas: List[Dict[str, str]],
    explicit_map: Dict[str, str],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    order: List[str] = []
    for replica in replicas:
        speaker = replica["speaker"]
        if speaker in mapping:
            continue
        if speaker in explicit_map:
            mapping[speaker] = explicit_map[speaker]
        else:
            order.append(speaker)

    roles = ["К", "Т"]
    for idx, speaker in enumerate(order):
        mapping[speaker] = roles[idx % len(roles)]

    return mapping
