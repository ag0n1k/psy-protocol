import logging
from pathlib import Path
from typing import Callable

from ..io_utils import load_json, save_json, save_text
from ..models import AsrResult, AsrSegment, AsrWord
from ..text_outputs import save_sentences_txt
from ..whisper_transcribe import extract_words, transcribe_audio


def run_asr(
    audio_path: Path,
    opts,
    cache_dir: Path,
    emit: Callable,
) -> AsrResult:
    """Stage 1: ASR transcription with caching. Returns AsrResult."""
    transcript_json_path = cache_dir / 'transcript.json'
    transcript_txt_path = cache_dir / 'transcript.txt'
    transcript_meta_path = cache_dir / 'transcript_meta.json'
    whisper_segments_path = cache_dir / 'whisper_segments.json'
    sentences_txt_path = cache_dir / 'sentences.txt'
    asr_result_path = cache_dir / 'asr_result.json'

    use_word_timestamps = opts.word_timestamps

    cache_valid = transcript_json_path.exists() and not opts.force_whisper
    if cache_valid:
        if transcript_meta_path.exists():
            cached_meta = load_json(transcript_meta_path)
            if cached_meta.get('transcription_method') != opts.transcription_method:
                logging.info('Transcription method changed, invalidating cache')
                cache_valid = False
        else:
            logging.info('Transcription meta missing, invalidating cache')
            cache_valid = False

    if cache_valid:
        logging.info('Transcription: loading from cache %s', transcript_json_path)
        whisper_result = load_json(transcript_json_path)
        emit('whisper', 80.0, 'Transcript loaded from cache')
    else:
        emit('whisper', 5.0, 'Transcription started')

        def transcription_progress(percent: float) -> None:
            mapped = 5.0 + (75.0 * (percent / 100.0))
            emit('whisper', mapped, f'Transcription {int(percent)}%')

        logging.info('Whisper: starting transcription (word_timestamps=%s)', use_word_timestamps)
        whisper_result = transcribe_audio(
            str(audio_path),
            opts.whisper_model,
            word_timestamps=use_word_timestamps,
            progress_callback=transcription_progress,
        )
        save_json(transcript_json_path, whisper_result)
        save_json(transcript_meta_path, {'transcription_method': opts.transcription_method})
        logging.info('Transcription: cache saved %s', transcript_json_path)
        emit('whisper', 80.0, 'Transcription completed')

    if opts.force_whisper or not transcript_txt_path.exists():
        save_text(transcript_txt_path, whisper_result.get('text', ''))
        logging.info('Whisper: transcript saved %s', transcript_txt_path)

    whisper_segments = whisper_result.get('segments', [])

    if opts.force_whisper or not whisper_segments_path.exists():
        save_json(whisper_segments_path, {'segments': whisper_segments})
        logging.info('Whisper: segments saved %s', whisper_segments_path)
    else:
        logging.info('Whisper: segments from cache %s', whisper_segments_path)
        whisper_segments_payload = load_json(whisper_segments_path)
        whisper_segments = whisper_segments_payload.get('segments', [])

    if opts.force_whisper or not sentences_txt_path.exists():
        save_sentences_txt(sentences_txt_path, whisper_segments)
        logging.info('Sentences TXT: %s', sentences_txt_path)

    words_raw = (
        extract_words(whisper_result, prob_threshold=opts.word_prob_threshold)
        if use_word_timestamps
        else []
    )
    if use_word_timestamps and words_raw:
        logging.info('Whisper: word timestamps %d', len(words_raw))

    segments = [
        AsrSegment(
            start=float(seg.get('start', 0.0)),
            end=float(seg.get('end', 0.0)),
            text=seg.get('text', '').strip(),
        )
        for seg in whisper_segments
    ]
    words = [
        AsrWord(start=float(w['start']), end=float(w['end']), word=w['word'])
        for w in words_raw
    ]
    result = AsrResult(
        text=whisper_result.get('text', ''),
        segments=segments,
        words=words,
        method='whisper',
        model=opts.whisper_model,
    )
    save_json(asr_result_path, result.to_dict())
    logging.info('ASR result saved %s', asr_result_path)
    return result
