import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: []
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

from .config import (
    DEFAULT_EMBEDDING_MIN_DURATION,
    DEFAULT_EMBEDDING_N_MELS,
    DEFAULT_EMBEDDING_N_MFCC,
    DEFAULT_KMEANS_ITERS,
)


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


class SincConv1d(nn.Module):
    def __init__(
        self,
        n_filters: int = 80,
        kernel_size: int = 251,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
        stride: int = 10,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.stride = stride
        self.half_kernel = kernel_size // 2
        # Pre-computed static filters in MLX format [C_out, K, C_in]; set by aufklarer loader.
        self._static_filters: Optional[mx.array] = None

        self._initialize_filters()
        window = np.hamming(kernel_size)[: self.half_kernel]
        self.window_ = mx.array(window, dtype=mx.float32)

        n = 2 * np.pi * np.arange(-self.half_kernel, 0.0) / sample_rate
        self.n_ = mx.array(n.reshape(1, -1), dtype=mx.float32)

    def _initialize_filters(self) -> None:
        def to_mel(hz: float) -> float:
            return 2595 * np.log10(1 + hz / 700)

        def to_hz(mel: float) -> float:
            return 700 * (10 ** (mel / 2595) - 1)

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(
            to_mel(low_hz),
            to_mel(high_hz),
            self.n_filters // 2 + 1,
            dtype=np.float32,
        )
        hz = to_hz(mel)

        self.low_hz_ = mx.array(hz[:-1].reshape(-1, 1), dtype=mx.float32)
        self.band_hz_ = mx.array(np.diff(hz).reshape(-1, 1), dtype=mx.float32)

    def make_filters(self, low: mx.array, high: mx.array, filt_type: str = "cos") -> mx.array:
        band = high[:, 0] - low[:, 0]
        ft_low = low @ self.n_
        ft_high = high @ self.n_

        if filt_type == "cos":
            bp_left = ((mx.sin(ft_high) - mx.sin(ft_low)) / (self.n_ / 2)) * self.window_
            bp_center = 2 * band.reshape(-1, 1)
            bp_right = bp_left[:, ::-1]
        elif filt_type == "sin":
            bp_left = ((mx.cos(ft_low) - mx.cos(ft_high)) / (self.n_ / 2)) * self.window_
            bp_center = mx.zeros_like(band.reshape(-1, 1))
            bp_right = -bp_left[:, ::-1]
        else:
            raise ValueError(f"Unknown filter type: {filt_type}")

        band_pass = mx.concatenate([bp_left, bp_center, bp_right], axis=1)
        band_pass = band_pass / (2 * band[:, None])
        return band_pass.reshape(self.n_filters // 2, 1, self.kernel_size)

    def get_filters(self) -> mx.array:
        low = self.min_low_hz + mx.abs(self.low_hz_)
        high = mx.clip(
            low + self.min_band_hz + mx.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        cos_filters = self.make_filters(low, high, filt_type="cos")
        sin_filters = self.make_filters(low, high, filt_type="sin")
        return mx.concatenate([cos_filters, sin_filters], axis=0)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=1)

        x_mlx = mx.transpose(x, (0, 2, 1))
        if self._static_filters is not None:
            # Already in MLX [C_out, K, C_in] format — no transpose needed.
            filters_mlx = self._static_filters
        else:
            filters = self.get_filters()
            filters_mlx = mx.transpose(filters, (0, 2, 1))
        output = mx.conv1d(x_mlx, filters_mlx, stride=self.stride, padding=0)
        output = mx.transpose(output, (0, 2, 1))
        return output


class SincNet(nn.Module):
    def __init__(self, sample_rate: int = 16000, stride: int = 10):
        super().__init__()
        self.wav_norm = nn.InstanceNorm(1, affine=True)
        self.sinc_conv = SincConv1d(
            n_filters=80,
            kernel_size=251,
            sample_rate=sample_rate,
            min_low_hz=50,
            min_band_hz=50,
            stride=stride,
        )
        self.norm1 = nn.InstanceNorm(80, affine=True)
        self.conv2 = nn.Conv1d(80, 60, kernel_size=5, stride=1)
        self.norm2 = nn.InstanceNorm(60, affine=True)
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, stride=1)
        self.norm3 = nn.InstanceNorm(60, affine=True)

    def maxpool1d(self, x: mx.array, pool_size: int = 3) -> mx.array:
        batch, time, channels = x.shape
        time_pooled = time // pool_size
        x_reshaped = x[:, : time_pooled * pool_size, :].reshape(
            batch, time_pooled, pool_size, channels
        )
        return mx.max(x_reshaped, axis=2)

    def __call__(self, waveforms: mx.array) -> mx.array:
        if waveforms.ndim == 2:
            waveforms = mx.expand_dims(waveforms, axis=1)

        x = mx.transpose(waveforms, (0, 2, 1))
        x = self.wav_norm(x)
        x = mx.transpose(x, (0, 2, 1))

        x = self.sinc_conv(x)
        x = mx.abs(x)

        x = mx.transpose(x, (0, 2, 1))
        x = self.maxpool1d(x, pool_size=3)
        x = self.norm1(x)
        x = mx.maximum(x, 0.01 * x)

        x = self.conv2(x)
        x = self.maxpool1d(x, pool_size=3)
        x = self.norm2(x)
        x = mx.maximum(x, 0.01 * x)

        x = self.conv3(x)
        x = self.maxpool1d(x, pool_size=3)
        x = self.norm3(x)
        x = mx.maximum(x, 0.01 * x)

        return x


class PyannoteSegmentationModel(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        sincnet_stride: int = 10,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 4,
        num_classes: int = 7,
    ):
        super().__init__()
        self.sincnet = SincNet(sample_rate=sample_rate, stride=sincnet_stride)
        self.lstm_forward = []
        self.lstm_backward = []
        for i in range(num_lstm_layers):
            input_dim = 60 if i == 0 else lstm_hidden_dim * 2
            self.lstm_forward.append(nn.LSTM(input_dim, lstm_hidden_dim))
            self.lstm_backward.append(nn.LSTM(input_dim, lstm_hidden_dim))

        self.linear1 = nn.Linear(lstm_hidden_dim * 2, 128)
        self.linear2 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)

    def __call__(self, waveforms: mx.array) -> mx.array:
        features = self.sincnet(waveforms)
        h = features
        for lstm_fwd, lstm_bwd in zip(self.lstm_forward, self.lstm_backward):
            h_fwd, _ = lstm_fwd(h)
            h_rev = h[:, ::-1, :]
            h_bwd, _ = lstm_bwd(h_rev)
            h_bwd = h_bwd[:, ::-1, :]
            h = mx.concatenate([h_fwd, h_bwd], axis=-1)

        h = self.linear1(h)
        h = mx.maximum(h, 0)
        h = self.linear2(h)
        h = mx.maximum(h, 0)
        logits = self.classifier(h)
        return nn.log_softmax(logits, axis=-1)

    def load_weights(self, weights_path: Path) -> None:
        weights = mx.load(str(weights_path))

        self.sincnet.sinc_conv.low_hz_ = weights["sincnet.conv1d.0.filterbank.low_hz_"]
        self.sincnet.sinc_conv.band_hz_ = weights["sincnet.conv1d.0.filterbank.band_hz_"]

        if "sincnet.wav_norm1d.weight" in weights:
            self.sincnet.wav_norm.weight = weights["sincnet.wav_norm1d.weight"]
            self.sincnet.wav_norm.bias = weights["sincnet.wav_norm1d.bias"]

        if "sincnet.norm1d.0.weight" in weights:
            self.sincnet.norm1.weight = weights["sincnet.norm1d.0.weight"]
            self.sincnet.norm1.bias = weights["sincnet.norm1d.0.bias"]

        conv2_weight = weights["sincnet.conv1d.1.weight"]
        self.sincnet.conv2.weight = mx.transpose(conv2_weight, (0, 2, 1))
        self.sincnet.conv2.bias = weights["sincnet.conv1d.1.bias"]
        if "sincnet.norm1d.1.weight" in weights:
            self.sincnet.norm2.weight = weights["sincnet.norm1d.1.weight"]
            self.sincnet.norm2.bias = weights["sincnet.norm1d.1.bias"]

        conv3_weight = weights["sincnet.conv1d.2.weight"]
        self.sincnet.conv3.weight = mx.transpose(conv3_weight, (0, 2, 1))
        self.sincnet.conv3.bias = weights["sincnet.conv1d.2.bias"]
        if "sincnet.norm1d.2.weight" in weights:
            self.sincnet.norm3.weight = weights["sincnet.norm1d.2.weight"]
            self.sincnet.norm3.bias = weights["sincnet.norm1d.2.bias"]

        for i in range(4):
            self.lstm_forward[i].Wx = weights[f"lstm.weight_ih_l{i}"]
            self.lstm_forward[i].Wh = weights[f"lstm.weight_hh_l{i}"]
            bias_ih = weights[f"lstm.bias_ih_l{i}"]
            bias_hh = weights[f"lstm.bias_hh_l{i}"]
            self.lstm_forward[i].bias = bias_ih + bias_hh

            self.lstm_backward[i].Wx = weights[f"lstm.weight_ih_l{i}_reverse"]
            self.lstm_backward[i].Wh = weights[f"lstm.weight_hh_l{i}_reverse"]
            bias_ih_rev = weights[f"lstm.bias_ih_l{i}_reverse"]
            bias_hh_rev = weights[f"lstm.bias_hh_l{i}_reverse"]
            self.lstm_backward[i].bias = bias_ih_rev + bias_hh_rev

        self.linear1.weight = weights["linear.0.weight"]
        self.linear1.bias = weights["linear.0.bias"]
        self.linear2.weight = weights["linear.1.weight"]
        self.linear2.bias = weights["linear.1.bias"]
        self.classifier.weight = weights["classifier.weight"]
        self.classifier.bias = weights["classifier.bias"]


    def load_weights_safetensors(self, weights_path: Path) -> None:
        """Load weights from aufklarer/Pyannote-Segmentation-MLX (safetensors / MLX key names)."""
        weights = mx.load(str(weights_path))

        # SincConv: pre-computed filters stored in MLX [C_out, K, C_in] format.
        if 'sincnet.conv.0.weight' in weights:
            self.sincnet.sinc_conv._static_filters = weights['sincnet.conv.0.weight']

        # Conv2 / Conv3 — already in MLX [C_out, K, C_in] format (no transpose).
        for attr, key in (
            (self.sincnet.conv2, 'sincnet.conv.1'),
            (self.sincnet.conv3, 'sincnet.conv.2'),
        ):
            if f'{key}.weight' in weights:
                attr.weight = weights[f'{key}.weight']
            if f'{key}.bias' in weights:
                attr.bias = weights[f'{key}.bias']

        # Instance-norm layers.
        for norm, key in (
            (self.sincnet.norm1, 'sincnet.norm.0'),
            (self.sincnet.norm2, 'sincnet.norm.1'),
            (self.sincnet.norm3, 'sincnet.norm.2'),
        ):
            if f'{key}.weight' in weights:
                norm.weight = weights[f'{key}.weight']
            if f'{key}.bias' in weights:
                norm.bias = weights[f'{key}.bias']

        # LSTM layers — aufklarer stores Wx/Wh/bias per direction in lstm_fwd/lstm_bwd.
        for i in range(4):
            for lst, base in (
                (self.lstm_forward, f'lstm_fwd.layers.{i}'),
                (self.lstm_backward, f'lstm_bwd.layers.{i}'),
            ):
                if f'{base}.Wx' in weights:
                    lst[i].Wx = weights[f'{base}.Wx']
                if f'{base}.Wh' in weights:
                    lst[i].Wh = weights[f'{base}.Wh']
                if f'{base}.bias' in weights:
                    lst[i].bias = weights[f'{base}.bias']

        # Linear layers and classifier.
        for attr, key in (
            (self.linear1, 'linear.0'),
            (self.linear2, 'linear.1'),
        ):
            if f'{key}.weight' in weights:
                attr.weight = weights[f'{key}.weight']
            if f'{key}.bias' in weights:
                attr.bias = weights[f'{key}.bias']

        if 'classifier.weight' in weights:
            self.classifier.weight = weights['classifier.weight']
        if 'classifier.bias' in weights:
            self.classifier.bias = weights['classifier.bias']


def load_pyannote_model(weights_path: Path) -> PyannoteSegmentationModel:
    model = PyannoteSegmentationModel()
    model.load_weights(weights_path)
    return model


def load_aufklarer_model(weights_path: Path) -> PyannoteSegmentationModel:
    model = PyannoteSegmentationModel()
    model.load_weights_safetensors(weights_path)
    return model


def resolve_diarization_weights(model_ref: str) -> Path:
    path = Path(model_ref).expanduser()
    if path.exists():
        if path.is_dir():
            weights = path / "weights.npz"
            if not weights.exists():
                raise FileNotFoundError(f"Missing weights.npz in {path}")
            return weights
        if path.suffix == ".npz":
            return path
        raise FileNotFoundError(f"Expected .npz file or directory, got {path}")

    weights_path = hf_hub_download(repo_id=model_ref, filename="weights.npz")
    return Path(weights_path)


def resolve_diarization_weights_safetensors(model_ref: str) -> Path:
    path = Path(model_ref).expanduser()
    if path.exists():
        if path.is_dir():
            weights = path / "model.safetensors"
            if not weights.exists():
                raise FileNotFoundError(f"Missing model.safetensors in {path}")
            return weights
        if path.suffix == ".safetensors":
            return path
        raise FileNotFoundError(f"Expected .safetensors file or directory, got {path}")

    weights_path = hf_hub_download(repo_id=model_ref, filename="model.safetensors")
    return Path(weights_path)


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.numpy(), target_sr


def process_audio_chunks(
    audio: np.ndarray,
    model: PyannoteSegmentationModel,
    chunk_size: int = 160000,
    overlap: int = 16000,
) -> Tuple[mx.array, np.ndarray]:
    total_samples = audio.shape[1]
    sr = 16000
    all_logits = []
    frame_times: List[float] = []

    start = 0
    current_time = 0.0

    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = audio[:, start:end]
        if chunk.shape[1] < chunk_size:
            pad_size = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_size)), mode="constant")

        chunk_mx = mx.array(chunk, dtype=mx.float32)
        logits = model(chunk_mx)

        if start == 0:
            all_logits.append(logits[0])
            frame_duration = (end - start) / sr / logits.shape[1]
            for i in range(logits.shape[1]):
                frame_times.append(current_time + i * frame_duration)
        else:
            overlap_frames = int(logits.shape[1] * overlap / chunk_size)
            all_logits.append(logits[0, overlap_frames:])
            frame_duration = (end - start) / sr / logits.shape[1]
            for i in range(overlap_frames, logits.shape[1]):
                frame_times.append(current_time + (i - overlap_frames) * frame_duration)

        start += chunk_size - overlap
        current_time = start / sr

    full_logits = mx.concatenate(all_logits, axis=0)
    return full_logits, np.array(frame_times)


def logits_to_segments(
    logits: mx.array,
    frame_times: np.ndarray,
    min_duration: float = 0.5,
    silence_threshold: float = 0.0,
) -> List[SpeakerSegment]:
    if logits.shape[0] == 0:
        return []

    probs = mx.softmax(logits, axis=-1)
    predictions = np.array(mx.argmax(probs, axis=-1))
    confidences = np.array(mx.max(probs, axis=-1))
    if silence_threshold > 0:
        predictions = np.where(confidences < silence_threshold, -1, predictions)

    segments: List[SpeakerSegment] = []
    current_speaker: Optional[int] = None
    start_time: Optional[float] = None

    for i, pred in enumerate(predictions):
        time = float(frame_times[i])
        if pred == -1:
            if current_speaker is not None and start_time is not None:
                duration = time - start_time
                if duration >= min_duration:
                    segments.append(
                        SpeakerSegment(
                            start=start_time,
                            end=time,
                            speaker=f"RAW_{current_speaker:02d}",
                        )
                    )
            current_speaker = None
            start_time = None
            continue

        if current_speaker is None:
            current_speaker = int(pred)
            start_time = time
            continue

        if pred != current_speaker:
            duration = time - (start_time or time)
            if duration >= min_duration:
                segments.append(
                    SpeakerSegment(
                        start=start_time or time,
                        end=time,
                        speaker=f"RAW_{current_speaker:02d}",
                    )
                )
            current_speaker = int(pred)
            start_time = time

    if current_speaker is not None and start_time is not None:
        end_time = float(frame_times[-1])
        duration = end_time - start_time
        if duration >= min_duration:
            segments.append(
                SpeakerSegment(
                    start=start_time,
                    end=end_time,
                    speaker=f"RAW_{current_speaker:02d}",
                )
            )

    return segments


def diarize_audio_raw(
    audio_path: str,
    model_ref: str,
    min_duration: float,
    chunk_size: int,
    overlap: int,
    silence_threshold: float,
) -> List[SpeakerSegment]:
    weights_path = resolve_diarization_weights(model_ref)
    model = load_pyannote_model(weights_path)
    waveform, _ = load_audio(audio_path)
    logits, frame_times = process_audio_chunks(
        waveform,
        model,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return logits_to_segments(
        logits,
        frame_times,
        min_duration=min_duration,
        silence_threshold=silence_threshold,
    )


def diarize_audio_raw_aufklarer(
    audio_path: str,
    model_ref: str,
    min_duration: float,
    chunk_size: int,
    overlap: int,
    silence_threshold: float,
) -> List[SpeakerSegment]:
    weights_path = resolve_diarization_weights_safetensors(model_ref)
    model = load_aufklarer_model(weights_path)
    waveform, _ = load_audio(audio_path)
    logits, frame_times = process_audio_chunks(
        waveform,
        model,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return logits_to_segments(
        logits,
        frame_times,
        min_duration=min_duration,
        silence_threshold=silence_threshold,
    )


def compute_mfcc_embedding(
    waveform: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    mfcc_transform,
) -> Optional[np.ndarray]:
    start_sample = max(0, int(start * sample_rate))
    end_sample = min(waveform.shape[1], int(end * sample_rate))
    if end_sample <= start_sample:
        return None
    duration = (end_sample - start_sample) / sample_rate
    if duration < DEFAULT_EMBEDDING_MIN_DURATION:
        return None

    segment = waveform[:, start_sample:end_sample]
    segment_tensor = torch.from_numpy(segment).float()
    mfcc = mfcc_transform(segment_tensor)
    if mfcc.shape[-1] == 0:
        return None
    mean = mfcc.mean(dim=-1).squeeze(0)
    std = mfcc.std(dim=-1).squeeze(0)
    embedding = torch.cat([mean, std], dim=0).numpy()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def load_speaker_embedding_model(model_id: str, device: str) -> Any:
    import huggingface_hub
    if "use_auth_token" not in huggingface_hub.hf_hub_download.__code__.co_varnames:
        original_download = huggingface_hub.hf_hub_download

        def hf_hub_download_compat(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return original_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = hf_hub_download_compat
    from speechbrain.inference import interfaces as sb_interfaces
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils import fetching as sb_fetching

    original_fetch = sb_fetching.fetch

    def fetch_compat(filename, *args, **kwargs):
        try:
            return original_fetch(filename, *args, **kwargs)
        except RemoteEntryNotFoundError as exc:
            if filename == "custom.py":
                raise ValueError("optional custom.py not found") from exc
            raise

    logging.info("Embeddings: loading model %s (%s)", model_id, device)
    sb_fetching.fetch = fetch_compat
    sb_interfaces.fetch = fetch_compat
    # Audio is loaded externally; speechbrain never loads files itself in this pipeline.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='SpeechBrain could not find any working torchaudio backend',
            module='speechbrain',
        )
        return EncoderClassifier.from_hparams(
            source=model_id,
            run_opts={"device": device},
        )


def compute_ecapa_embedding(
    waveform: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    classifier: Any,
) -> Optional[np.ndarray]:
    start_sample = max(0, int(start * sample_rate))
    end_sample = min(waveform.shape[1], int(end * sample_rate))
    if end_sample <= start_sample:
        return None
    duration = (end_sample - start_sample) / sample_rate
    if duration < DEFAULT_EMBEDDING_MIN_DURATION:
        return None

    segment = waveform[:, start_sample:end_sample]
    segment_tensor = torch.from_numpy(segment).float()
    with torch.no_grad():
        embedding = classifier.encode_batch(segment_tensor)
    embedding = embedding.squeeze().cpu().numpy()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def init_kmeans_plus(embeddings: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = embeddings.shape[0]
    centers = [embeddings[rng.integers(0, n)]]
    for _ in range(1, k):
        distances = np.min(
            np.linalg.norm(embeddings[:, None, :] - np.array(centers)[None, :, :], axis=2) ** 2,
            axis=1,
        )
        total = distances.sum()
        if total == 0:
            centers.append(embeddings[rng.integers(0, n)])
            continue
        probs = distances / total
        centers.append(embeddings[rng.choice(n, p=probs)])
    return np.vstack(centers)


def kmeans_cluster(
    embeddings: np.ndarray,
    k: int,
    max_iter: int = DEFAULT_KMEANS_ITERS,
    seed: int = 42,
) -> np.ndarray:
    n = embeddings.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if k <= 1:
        return np.zeros(n, dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    centers = init_kmeans_plus(embeddings, k, rng)
    for _ in range(max_iter):
        distances = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = []
        for i in range(k):
            cluster = embeddings[labels == i]
            if cluster.size == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(cluster.mean(axis=0))
        new_centers = np.vstack(new_centers)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels


def _log_cluster_separation(embeddings: np.ndarray, labels: np.ndarray) -> None:
    mask0, mask1 = labels == 0, labels == 1
    if mask0.sum() > 1 and mask1.sum() > 1:
        within = (
            (embeddings[mask0] @ embeddings[mask0].T).mean()
            + (embeddings[mask1] @ embeddings[mask1].T).mean()
        ) / 2
        between = (embeddings[mask0] @ embeddings[mask1].T).mean()
        logging.info(
            'Cluster sep: within=%.3f between=%.3f delta=%.3f',
            within,
            between,
            within - between,
        )


def spectral_cluster_2(embeddings: np.ndarray) -> np.ndarray:
    """Spectral clustering for k=2 using eigenvector k-means on 2D spectral embedding."""
    n = embeddings.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    # Cosine affinity: for normalized embeddings = dot product
    W = embeddings @ embeddings.T        # [-1, 1]
    W = (W + 1.0) / 2.0                 # → [0, 1]
    np.fill_diagonal(W, 0.0)

    # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    L_sym = np.eye(n) - d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]

    _, vecs = np.linalg.eigh(L_sym)
    # Take 2 smallest eigenvectors (sklearn standard for spectral clustering)
    spectral_embedding = vecs[:, :2]
    # Row-normalize (sklearn standard)
    norms = np.linalg.norm(spectral_embedding, axis=1, keepdims=True)
    spectral_embedding = spectral_embedding / np.where(norms > 1e-10, norms, 1.0)

    # K-means on 2D spectral embedding — more robust than gap-based threshold
    labels = kmeans_cluster(spectral_embedding, k=2)

    fiedler = vecs[:, 1]
    sorted_vals = np.sort(fiedler)
    gaps = np.diff(sorted_vals)
    logging.info(
        'Spectral: max_gap=%.4f, cluster_sizes=%s',
        gaps.max(),
        str(np.bincount(labels).tolist()),
    )
    _log_cluster_separation(embeddings, labels)
    return labels


def agglomerative_cluster_2(embeddings: np.ndarray) -> np.ndarray:
    """Agglomerative clustering with average linkage and cosine distance for k=2."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = embeddings.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    dist = pdist(embeddings, metric='cosine')
    Z = linkage(dist, method='average')
    labels = fcluster(Z, t=2, criterion='maxclust') - 1  # fcluster starts from 1
    logging.info('Agglomerative: cluster_sizes=%s', str(np.bincount(labels).tolist()))
    _log_cluster_separation(embeddings, labels)
    return labels


def remap_labels_by_first_occurrence(labels: np.ndarray) -> np.ndarray:
    mapping: Dict[int, int] = {}
    next_id = 0
    remapped = []
    for label in labels:
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        remapped.append(mapping[label])
    return np.array(remapped, dtype=int)


def assign_missing_labels_by_nearest(
    segments: List[SpeakerSegment],
    labeled_indices: List[int],
    labeled_labels: np.ndarray,
) -> List[int]:
    if not labeled_indices:
        return [0 for _ in segments]

    labeled_midpoints = np.array(
        [(segments[idx].start + segments[idx].end) / 2 for idx in labeled_indices]
    )
    labeled_labels_list = labeled_labels.tolist()
    index_to_label = {idx: labeled_labels_list[pos] for pos, idx in enumerate(labeled_indices)}

    full_labels: List[int] = []
    for idx, seg in enumerate(segments):
        if idx in index_to_label:
            full_labels.append(index_to_label[idx])
            continue
        midpoint = (seg.start + seg.end) / 2
        nearest = int(np.argmin(np.abs(labeled_midpoints - midpoint)))
        full_labels.append(labeled_labels_list[nearest])
    return full_labels


def cluster_segments_by_embeddings(
    waveform: np.ndarray,
    sample_rate: int,
    segments: List[SpeakerSegment],
    num_speakers: int,
    embedding_model_id: str,
    embedding_device: str,
    clustering_method: str = 'kmeans',
) -> List[SpeakerSegment]:
    if not segments:
        return []

    logging.info("Clustering: input segments %d", len(segments))
    classifier = load_speaker_embedding_model(embedding_model_id, embedding_device)

    embeddings: List[np.ndarray] = []
    indices: List[int] = []
    for idx, seg in enumerate(segments):
        embedding = compute_ecapa_embedding(
            waveform,
            sample_rate,
            seg.start,
            seg.end,
            classifier,
        )
        if embedding is None:
            continue
        embeddings.append(embedding)
        indices.append(idx)

    logging.info("Clustering: embeddings count %d", len(embeddings))
    if not embeddings:
        logging.warning("Clustering: no embeddings obtained, fallback to MFCC")
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=DEFAULT_EMBEDDING_N_MFCC,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": DEFAULT_EMBEDDING_N_MELS,
            },
        )
        for idx, seg in enumerate(segments):
            embedding = compute_mfcc_embedding(
                waveform,
                sample_rate,
                seg.start,
                seg.end,
                mfcc_transform,
            )
            if embedding is None:
                continue
            embeddings.append(embedding)
            indices.append(idx)
        if not embeddings:
            return [
                SpeakerSegment(seg.start, seg.end, "SPEAKER_00")
                for seg in segments
            ]

    embeddings_np = np.vstack(embeddings)
    k = num_speakers if num_speakers and num_speakers > 0 else 1
    k = min(k, embeddings_np.shape[0])
    if k == 2 and clustering_method == 'spectral':
        logging.info('Clustering: using spectral (eigvec k-means, k=2)')
        labels = spectral_cluster_2(embeddings_np)
    elif k == 2 and clustering_method == 'agglomerative':
        logging.info('Clustering: using agglomerative (average cosine, k=2)')
        labels = agglomerative_cluster_2(embeddings_np)
    else:
        logging.info('Clustering: using k-means (k=%d)', k)
        labels = kmeans_cluster(embeddings_np, k)
    labels = remap_labels_by_first_occurrence(labels)

    full_labels = assign_missing_labels_by_nearest(segments, indices, labels)
    return [
        SpeakerSegment(seg.start, seg.end, f"SPEAKER_{label:02d}")
        for seg, label in zip(segments, full_labels)
    ]


def merge_segments(segments: List[SpeakerSegment], gap: float) -> List[SpeakerSegment]:
    if not segments:
        return []
    sorted_segments = sorted(segments, key=lambda s: s.start)
    merged = [sorted_segments[0]]
    for seg in sorted_segments[1:]:
        last = merged[-1]
        if seg.speaker == last.speaker and seg.start - last.end <= gap:
            last.end = max(last.end, seg.end)
        else:
            merged.append(seg)
    return merged


def merge_sandwiched_segments(
    segments: List[SpeakerSegment],
    max_duration: float,
    max_gap: float,
) -> List[SpeakerSegment]:
    if len(segments) < 3:
        return segments
    segments = sorted(segments, key=lambda s: s.start)
    changed = True
    while changed:
        changed = False
        merged: List[SpeakerSegment] = []
        i = 0
        while i < len(segments):
            if 0 < i < len(segments) - 1:
                prev_seg = segments[i - 1]
                curr_seg = segments[i]
                next_seg = segments[i + 1]
                duration = curr_seg.end - curr_seg.start
                if (
                    prev_seg.speaker == next_seg.speaker
                    and curr_seg.speaker != prev_seg.speaker
                    and duration <= max_duration
                    and curr_seg.start - prev_seg.end <= max_gap
                    and next_seg.start - curr_seg.end <= max_gap
                ):
                    prev_seg.end = max(prev_seg.end, next_seg.end)
                    segments[i - 1] = prev_seg
                    i += 2
                    changed = True
                    continue
            merged.append(segments[i])
            i += 1
        segments = merged
    return segments


def diarize_with_pyannote_pipeline(
    audio_path: str,
    num_speakers: int = 2,
    pipeline_model: str = 'pyannote/speaker-diarization-3.1',
    hf_token: Optional[str] = None,
) -> List[SpeakerSegment]:
    from pyannote.audio import Pipeline

    logging.info('Pyannote pipeline: loading %s', pipeline_model)
    pipeline = Pipeline.from_pretrained(pipeline_model)
    pipeline.to(torch.device("mps"))

    # Load via torchaudio to avoid sample-count mismatch with compressed formats (ogg, mp3).
    # Pyannote reads duration from metadata and expects an exact sample count; decoding
    # compressed audio often yields slightly fewer samples, causing a ValueError.
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000

    logging.info('Pyannote pipeline: running diarization')
    diarization = pipeline({'waveform': waveform, 'sample_rate': sr}, num_speakers=num_speakers)

    # pyannote 4.x returns DiarizeOutput; older versions return Annotation directly
    annotation = getattr(diarization, 'speaker_diarization', diarization)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(SpeakerSegment(start=turn.start, end=turn.end, speaker=speaker))

    if segments:
        label_map = {}
        next_id = 0
        for seg in segments:
            if seg.speaker not in label_map:
                label_map[seg.speaker] = f'SPEAKER_{next_id:02d}'
                next_id += 1
        segments = [SpeakerSegment(s.start, s.end, label_map[s.speaker]) for s in segments]

    logging.info('Pyannote pipeline: %d segments', len(segments))
    return segments


def post_process_diarization(
    segments: List[SpeakerSegment],
    merge_gap: float,
    sandwich_max_duration: float,
) -> List[SpeakerSegment]:
    for _ in range(10):
        prev_count = len(segments)
        segments = merge_segments(segments, gap=merge_gap)
        segments = merge_sandwiched_segments(
            segments,
            max_duration=sandwich_max_duration,
            max_gap=merge_gap,
        )
        if len(segments) == prev_count:
            break
    return segments
