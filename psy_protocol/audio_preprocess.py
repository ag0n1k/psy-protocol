import logging
import os
import shutil
import subprocess
from typing import Tuple

import numpy as np


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


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode audio into a mono float32 waveform of shape (1, samples) via ffmpeg CLI.

    Not torchaudio.load: it goes through torchcodec, which links against a fixed range
    of FFmpeg library versions and breaks on every major `brew upgrade ffmpeg`.
    """
    cmd = [
        _ffmpeg_bin(), '-v', 'error', '-i', path,
        '-f', 's16le', '-ac', '1', '-ar', str(target_sr),
        '-',
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors='replace')
        raise RuntimeError(f'ffmpeg decoding failed: {stderr[-500:]}')
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples[np.newaxis, :], target_sr


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
