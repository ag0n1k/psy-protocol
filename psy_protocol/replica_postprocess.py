import copy
from typing import Dict, List

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
