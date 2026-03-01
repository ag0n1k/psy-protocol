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
        mapping[key] = role
    return mapping


def map_speakers_to_roles(
    replicas: List[Dict[str, str]],
    explicit_map: Dict[str, str],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    order: List[str] = []
    for replica in replicas:
        speaker = replica['speaker']
        if speaker in mapping:
            continue
        if speaker in explicit_map:
            mapping[speaker] = explicit_map[speaker]
        else:
            order.append(speaker)

    if len(order) >= 2:
        volume: Dict[str, int] = {}
        for replica in replicas:
            speaker = replica['speaker']
            if speaker in order:
                volume[speaker] = volume.get(speaker, 0) + len(replica.get('text', ''))
        sorted_by_volume = sorted(order, key=lambda s: volume.get(s, 0), reverse=True)
        top = volume.get(sorted_by_volume[0], 0)
        second = volume.get(sorted_by_volume[1], 0)
        total = top + second
        if total > 0 and (top - second) / total > 0.2:
            mapping[sorted_by_volume[0]] = 'К'
            mapping[sorted_by_volume[1]] = 'Т'
            for idx, speaker in enumerate(sorted_by_volume[2:], start=2):
                mapping[speaker] = ['К', 'Т'][idx % 2]
            return mapping

    roles = ['К', 'Т']
    for idx, speaker in enumerate(order):
        mapping[speaker] = roles[idx % len(roles)]

    return mapping
