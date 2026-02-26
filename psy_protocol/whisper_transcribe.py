from pathlib import Path
from typing import Dict, List

import mlx_whisper


def transcribe_audio(audio_path: str, model_path: str, word_timestamps: bool) -> Dict:
    model_path = str(Path(model_path).expanduser())
    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_path,
        word_timestamps=word_timestamps,
    )


def extract_words(whisper_result: Dict, prob_threshold: float) -> List[Dict[str, float]]:
    words: List[Dict[str, float]] = []
    for segment in whisper_result.get("segments", []):
        for word in segment.get("words") or []:
            text = word.get("word")
            start = word.get("start")
            end = word.get("end")
            probability = word.get("probability")
            if text is None or start is None or end is None:
                continue
            if prob_threshold > 0 and probability is not None:
                if float(probability) < prob_threshold:
                    continue
            words.append({"word": str(text), "start": float(start), "end": float(end)})
    return words
