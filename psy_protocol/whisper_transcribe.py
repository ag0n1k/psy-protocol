from pathlib import Path
from typing import Callable, Dict, List, Optional

import importlib

import mlx_whisper


def transcribe_audio(
    audio_path: str,
    model_path: str,
    word_timestamps: bool,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict:
    model_path = str(Path(model_path).expanduser())
    if progress_callback is None:
        return mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            word_timestamps=word_timestamps,
            verbose=False,
        )

    transcribe_module = importlib.import_module(mlx_whisper.transcribe.__module__)
    original_tqdm = transcribe_module.tqdm.tqdm

    def patched_tqdm(*args, **kwargs):
        bar = original_tqdm(*args, **kwargs)
        total = kwargs.get("total", getattr(bar, "total", None))
        if total:
            progress_callback(0.0)
            original_update = bar.update

            def update(n=1):
                result = original_update(n)
                current = float(getattr(bar, "n", 0.0))
                pct = max(0.0, min(100.0, (current / float(total)) * 100.0))
                progress_callback(pct)
                return result

            bar.update = update
        return bar

    transcribe_module.tqdm.tqdm = patched_tqdm
    try:
        return mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            word_timestamps=word_timestamps,
            verbose=False,
        )
    finally:
        transcribe_module.tqdm.tqdm = original_tqdm


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
