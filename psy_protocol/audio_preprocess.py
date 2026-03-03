import logging
import os
import shutil
import subprocess


def _ffmpeg_bin() -> str:
    path = shutil.which('ffmpeg')
    if path and os.path.exists(path):
        return path

    candidates = [
        # common Homebrew location on macOS Apple Silicon
        '/opt/homebrew/bin/ffmpeg',

        # common Homebrew location on macOS Intel
        '/usr/local/bin/ffmpeg',
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        'ffmpeg binary not found. Install ffmpeg and ensure it is available in PATH '
        '(e.g. `brew install ffmpeg`).'
    )


def preprocess_audio(input_path: str, output_path: str) -> None:
    """Convert audio to 16kHz mono WAV with normalization via ffmpeg."""
    cmd = [
        _ffmpeg_bin(), '-y', '-i', input_path,
        '-af', 'highpass=f=80,dynaudnorm=f=150:g=15',
        '-ac', '1',
        '-ar', '16000',
        '-sample_fmt', 's16',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg preprocessing failed: {result.stderr[-500:]}')
    logging.info('Audio preprocessed: %s -> %s', input_path, output_path)


def extract_audio_segment(input_path: str, start: float, end: float, output_path: str) -> None:
    """Extract a time segment from audio using ffmpeg."""
    cmd = [
        _ffmpeg_bin(), '-y',
        '-ss', str(start), '-to', str(end),
        '-i', input_path,
        '-ac', '1', '-ar', '16000', '-sample_fmt', 's16',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg segment extraction failed: {result.stderr[-300:]}')
