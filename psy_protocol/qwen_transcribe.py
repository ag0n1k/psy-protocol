import logging
from typing import Callable, Dict, List, Optional

_cached_model = None
_cached_model_path = None


def _load_model(model_path: str):
    global _cached_model, _cached_model_path
    if _cached_model_path == model_path and _cached_model is not None:
        return _cached_model

    from mlx_audio.stt.utils import load_model

    logging.info('Loading Qwen3-ASR model: %s', model_path)
    model = load_model(model_path)
    _cached_model = model
    _cached_model_path = model_path
    return model


def transcribe_audio_qwen(
    audio_path: str,
    model_path: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict:
    model = _load_model(model_path)
    if progress_callback:
        progress_callback(10.0)

    from mlx_audio.stt.generate import generate_transcription

    logging.info('Qwen3-ASR: transcribing %s', audio_path)
    result = generate_transcription(model=model, audio=audio_path)
    if progress_callback:
        progress_callback(100.0)

    return _to_whisper_format(result)


def _to_whisper_format(result) -> Dict:
    text = getattr(result, 'text', '') or ''
    raw_segments = getattr(result, 'segments', None) or []

    segments: List[Dict] = []
    for i, seg in enumerate(raw_segments):
        if isinstance(seg, dict):
            start = float(seg.get('start', 0))
            end = float(seg.get('end', 0))
            seg_text = seg.get('text', '').strip()
        else:
            start = float(getattr(seg, 'start_time', getattr(seg, 'start', 0)))
            end = float(getattr(seg, 'end_time', getattr(seg, 'end', 0)))
            seg_text = getattr(seg, 'text', '').strip()
        if seg_text:
            segments.append({'id': i, 'start': start, 'end': end, 'text': seg_text, 'words': []})

    if not segments and text.strip():
        segments = [{'id': 0, 'start': 0.0, 'end': 0.0, 'text': text.strip(), 'words': []}]

    return {'text': text, 'segments': segments}
