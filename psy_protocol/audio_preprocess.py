import logging
import shutil
import subprocess


def _ffmpeg_bin() -> str:
    path = shutil.which('ffmpeg')
    if path is None:
        # common Homebrew location on macOS Apple Silicon
        path = '/opt/homebrew/bin/ffmpeg'
    return path


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
