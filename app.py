#!/usr/bin/env python3
from psy_protocol.cli import main


if __name__ == "__main__":
    main()
    raise SystemExit(0)


DEFAULT_WHISPER_MODEL = "~/.cache/mlx/large-v3-turbo"
DEFAULT_DIARIZATION_MODEL = "mlx-community/pyannote-segmentation-3.0-mlx"
DEFAULT_FONT_NAME = "Times New Roman"
TABLE_FONT_NAME = "Calibri"
DEFAULT_FONT_SIZE_PT = 12
DEFAULT_LINE_SPACING = 1.0
DEFAULT_SPACE_BEFORE_PT = 1
DEFAULT_SPACE_AFTER_PT = 0
DEFAULT_EMBEDDING_MIN_DURATION = 0.3
DEFAULT_EMBEDDING_N_MFCC = 20
DEFAULT_EMBEDDING_N_MELS = 40
DEFAULT_KMEANS_ITERS = 20
DEFAULT_SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SPEAKER_EMBEDDING_DEVICE = "cpu"
DEFAULT_WORD_SMOOTH_MIN_WORDS = 2
DEFAULT_WORD_PROB_THRESHOLD = 0.2
LOG_FORMAT = "%(levelname)s %(message)s"
TABLE_HEADERS = [
    "№",
    "К или Т",
    "Текст диалога",
    "Феномены клиента",
    "Комментарии к процессу клиента",
    "Комментарии к интервенциям терапевта",
    "Комментарии к внутренней личной динамике терапевта",
]
A4_LANDSCAPE_WIDTH_CM = 29.7
A4_LANDSCAPE_HEIGHT_CM = 21.0
MARGIN_TOP_CM = 1.5
MARGIN_BOTTOM_CM = 3.0
MARGIN_LEFT_CM = 2.0
MARGIN_RIGHT_CM = 2.0


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

        filters = self.get_filters()
        x_mlx = mx.transpose(x, (0, 2, 1))
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


def load_pyannote_model(weights_path: Path) -> PyannoteSegmentationModel:
    model = PyannoteSegmentationModel()
    model.load_weights(weights_path)
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


def load_speaker_embedding_model(
    model_id: str,
    device: str,
) -> Any:
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []
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
    from huggingface_hub.errors import RemoteEntryNotFoundError

    original_fetch = sb_fetching.fetch

    def fetch_compat(filename, *args, **kwargs):
        try:
            return original_fetch(filename, *args, **kwargs)
        except RemoteEntryNotFoundError as exc:
            if filename == "custom.py":
                raise ValueError("optional custom.py not found") from exc
            raise

    logging.info("Эмбеддинги: загрузка модели %s (%s)", model_id, device)
    sb_fetching.fetch = fetch_compat
    sb_interfaces.fetch = fetch_compat
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
) -> List[SpeakerSegment]:
    if not segments:
        return []

    logging.info("Кластеризация: входных сегментов %d", len(segments))
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

    logging.info("Кластеризация: эмбеддингов %d", len(embeddings))
    if not embeddings:
        logging.warning("Кластеризация: эмбеддинги не получены, fallback MFCC")
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
    labels = kmeans_cluster(embeddings_np, k)
    labels = remap_labels_by_first_occurrence(labels)

    full_labels = assign_missing_labels_by_nearest(segments, indices, labels)
    return [
        SpeakerSegment(seg.start, seg.end, f"SPEAKER_{label:02d}")
        for seg, label in zip(segments, full_labels)
    ]


def transcribe_audio(audio_path: str, model_path: str, word_timestamps: bool) -> Dict:
    model_path = str(Path(model_path).expanduser())
    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_path,
        word_timestamps=word_timestamps,
    )


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
        if key.startswith("SPEAKER_"):
            key = key
        mapping[key] = role
    return mapping


def assign_speakers_to_spans(
    spans: List[Tuple[float, float]],
    diarization_segments: List[SpeakerSegment],
) -> List[str]:
    diarization_segments = sorted(diarization_segments, key=lambda s: s.start)
    assigned: List[str] = []
    idx = 0
    last_speaker = diarization_segments[0].speaker if diarization_segments else "SPEAKER_00"

    for start, end in spans:
        while idx < len(diarization_segments) and diarization_segments[idx].end <= start:
            idx += 1

        overlaps: Dict[str, float] = {}
        j = idx
        while j < len(diarization_segments) and diarization_segments[j].start < end:
            dseg = diarization_segments[j]
            overlap = max(0.0, min(end, dseg.end) - max(start, dseg.start))
            if overlap > 0:
                overlaps[dseg.speaker] = overlaps.get(dseg.speaker, 0.0) + overlap
            j += 1

        if overlaps:
            speaker = max(overlaps.items(), key=lambda item: item[1])[0]
        else:
            speaker = last_speaker
        assigned.append(speaker)
        last_speaker = speaker

    return assigned


def assign_speakers_to_segments(
    whisper_segments: List[Dict],
    diarization_segments: List[SpeakerSegment],
) -> List[str]:
    spans = [
        (float(seg.get("start", 0.0)), float(seg.get("end", seg.get("start", 0.0))))
        for seg in whisper_segments
    ]
    return assign_speakers_to_spans(spans, diarization_segments)


def build_replicas(
    whisper_segments: List[Dict],
    diarization_segments: List[SpeakerSegment],
) -> List[Dict[str, str]]:
    speakers = assign_speakers_to_segments(whisper_segments, diarization_segments)
    replicas: List[Dict[str, str]] = []

    for seg, speaker in zip(whisper_segments, speakers):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if not replicas or replicas[-1]["speaker"] != speaker:
            replicas.append({"speaker": speaker, "text": text})
        else:
            replicas[-1]["text"] = f"{replicas[-1]['text']} {text}"

    return replicas


def extract_words(
    whisper_result: Dict,
    prob_threshold: float,
) -> List[Dict[str, float]]:
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


def build_replicas_from_words(
    words: List[Dict[str, float]],
    diarization_segments: List[SpeakerSegment],
    smooth_min_words: int,
) -> List[Dict[str, str]]:
    if not words:
        return []
    spans = [(w["start"], w["end"]) for w in words]
    speakers = assign_speakers_to_spans(spans, diarization_segments)
    speakers = smooth_word_speakers(words, speakers, min_words=smooth_min_words)

    replicas: List[Dict[str, str]] = []
    current_speaker: Optional[str] = None
    tokens: List[str] = []

    for word, speaker in zip(words, speakers):
        token = word["word"]
        if current_speaker is None:
            current_speaker = speaker
        if speaker != current_speaker:
            text = "".join(tokens).strip()
            if text:
                replicas.append({"speaker": current_speaker, "text": text})
            tokens = []
            current_speaker = speaker
        tokens.append(token)

    text = "".join(tokens).strip()
    if text and current_speaker is not None:
        replicas.append({"speaker": current_speaker, "text": text})

    return replicas


def smooth_word_speakers(
    words: List[Dict[str, float]],
    speakers: List[str],
    min_words: int = 2,
) -> List[str]:
    if min_words <= 1 or len(speakers) < 3:
        return speakers
    smoothed = speakers[:]
    runs: List[Tuple[int, int, str]] = []
    start = 0
    current = speakers[0]
    for idx, speaker in enumerate(speakers[1:], start=1):
        if speaker != current:
            runs.append((start, idx - 1, current))
            start = idx
            current = speaker
    runs.append((start, len(speakers) - 1, current))

    for i in range(1, len(runs) - 1):
        run_start, run_end, run_speaker = runs[i]
        run_len = run_end - run_start + 1
        prev_speaker = runs[i - 1][2]
        next_speaker = runs[i + 1][2]
        if run_len <= min_words and prev_speaker == next_speaker:
            for idx in range(run_start, run_end + 1):
                smoothed[idx] = prev_speaker

    return smoothed


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


def post_process_diarization(
    segments: List[SpeakerSegment],
    merge_gap: float,
    sandwich_max_duration: float,
) -> List[SpeakerSegment]:
    segments = merge_segments(segments, gap=merge_gap)
    segments = merge_sandwiched_segments(
        segments,
        max_duration=sandwich_max_duration,
        max_gap=merge_gap,
    )
    return segments


def map_speakers_to_roles(
    replicas: List[Dict[str, str]],
    explicit_map: Dict[str, str],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    order: List[str] = []
    for replica in replicas:
        speaker = replica["speaker"]
        if speaker in mapping:
            continue
        if speaker in explicit_map:
            mapping[speaker] = explicit_map[speaker]
        else:
            order.append(speaker)

    roles = ["К", "Т"]
    for idx, speaker in enumerate(order):
        mapping[speaker] = roles[idx % len(roles)]

    return mapping


def create_docx(
    output_path: str,
    replicas: List[Dict[str, str]],
    metadata: Dict[str, str],
) -> None:
    doc = Document()
    section = doc.sections[-1]
    section.orientation = WD_ORIENT.LANDSCAPE
    set_page_a4_landscape(section)
    set_default_font(doc, DEFAULT_FONT_NAME)

    for label, value in metadata.items():
        line = f"{label}: {value}" if value else label
        paragraph = doc.add_paragraph()
        set_paragraph_spacing(
            paragraph,
            line_spacing=DEFAULT_LINE_SPACING,
            space_before=DEFAULT_SPACE_BEFORE_PT,
            space_after=DEFAULT_SPACE_AFTER_PT,
        )
        run = paragraph.add_run(line)
        set_run_font(
            run,
            DEFAULT_FONT_NAME,
            bold=True,
            size_pt=DEFAULT_FONT_SIZE_PT,
        )

    table = doc.add_table(rows=1 + len(replicas), cols=len(TABLE_HEADERS))
    set_table_layout_fixed(table)
    set_column_widths(table, [0.91, 1.24, 9.02, 3.06, 3.48, 3.57, 3.31])
    set_table_borders(table)
    set_table_rows_layout(table)
    for col_idx, header in enumerate(TABLE_HEADERS):
        set_cell_text_with_alignment(
            table.rows[0].cells[col_idx],
            header,
            bold=True,
            font_name=TABLE_FONT_NAME,
            alignment=WD_ALIGN_PARAGRAPH.CENTER,
        )

    for idx, replica in enumerate(replicas):
        row = table.rows[idx + 1]
        set_cell_text(row.cells[0], str(idx + 1), bold=True, font_name=TABLE_FONT_NAME)
        set_cell_text(row.cells[1], replica["role"], bold=True, font_name=TABLE_FONT_NAME)
        set_cell_text(row.cells[2], replica["text"], bold=False, font_name=TABLE_FONT_NAME)

    doc.save(output_path)


def set_default_font(doc: Document, font_name: str) -> None:
    style = doc.styles["Normal"]
    style.font.name = font_name
    style.font.size = Pt(DEFAULT_FONT_SIZE_PT)
    r_pr = style._element.get_or_add_rPr()
    r_fonts = r_pr.get_or_add_rFonts()
    r_fonts.set(qn("w:eastAsia"), font_name)


def set_page_a4_landscape(section) -> None:
    section.page_width = Cm(A4_LANDSCAPE_WIDTH_CM)
    section.page_height = Cm(A4_LANDSCAPE_HEIGHT_CM)
    section.top_margin = Cm(MARGIN_TOP_CM)
    section.bottom_margin = Cm(MARGIN_BOTTOM_CM)
    section.left_margin = Cm(MARGIN_LEFT_CM)
    section.right_margin = Cm(MARGIN_RIGHT_CM)


def set_run_font(
    run,
    font_name: str,
    bold: Optional[bool] = None,
    size_pt: Optional[int] = None,
) -> None:
    run.font.name = font_name
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.get_or_add_rFonts()
    r_fonts.set(qn("w:eastAsia"), font_name)
    if bold is not None:
        run.bold = bold


def set_cell_text(
    cell,
    text: str,
    bold: bool = False,
    font_name: str = DEFAULT_FONT_NAME,
) -> None:
    cell.text = text
    if not cell.paragraphs:
        return
    for paragraph in cell.paragraphs:
        set_paragraph_spacing(
            paragraph,
            line_spacing=DEFAULT_LINE_SPACING,
            space_before=DEFAULT_SPACE_BEFORE_PT,
            space_after=DEFAULT_SPACE_AFTER_PT,
        )
        for run in paragraph.runs:
            set_run_font(
                run,
                font_name,
                bold=bold,
                size_pt=DEFAULT_FONT_SIZE_PT,
            )


def set_cell_text_with_alignment(
    cell,
    text: str,
    bold: bool,
    font_name: str,
    alignment: WD_ALIGN_PARAGRAPH,
) -> None:
    set_cell_text(cell, text, bold=bold, font_name=font_name)
    for paragraph in cell.paragraphs:
        paragraph.alignment = alignment


def set_column_widths(table, widths_cm: List[float]) -> None:
    table.autofit = False
    if hasattr(table, "allow_autofit"):
        table.allow_autofit = False
    total_width = sum(widths_cm)
    set_table_width(table, total_width)
    set_table_grid(table, widths_cm)
    for idx, width in enumerate(widths_cm):
        table.columns[idx].width = Cm(width)
    for row in table.rows:
        for idx, width in enumerate(widths_cm):
            set_cell_width(row.cells[idx], width)


def set_cell_width(cell, width_cm: float) -> None:
    width = Cm(width_cm)
    cell.width = width
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:type"), "dxa")
    tc_w.set(qn("w:w"), str(width.twips))


def set_table_layout_fixed(table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")


def set_table_width(table, width_cm: float) -> None:
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(Cm(width_cm).twips))


def set_table_grid(table, widths_cm: List[float]) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_grid = tbl.tblGrid
    if tbl_grid is None:
        tbl_grid = OxmlElement("w:tblGrid")
        tbl.insert(tbl.index(tbl_pr) + 1, tbl_grid)
    else:
        tbl_grid.clear()
    for width in widths_cm:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(Cm(width).twips))
        tbl_grid.append(grid_col)


def set_table_rows_layout(table, min_height_pt: Optional[int] = None) -> None:
    for row in table.rows:
        row.allow_break_across_pages = True
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        if min_height_pt is not None:
            row.height = Pt(min_height_pt)


def set_paragraph_spacing(
    paragraph,
    line_spacing: float,
    space_before: int,
    space_after: int,
) -> None:
    paragraph_format = paragraph.paragraph_format
    paragraph_format.line_spacing = line_spacing
    paragraph_format.space_before = Pt(space_before)
    paragraph_format.space_after = Pt(space_after)


def set_table_borders(table, size: int = 4) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        elem = OxmlElement(f"w:{edge}")
        elem.set(qn("w:val"), "single")
        elem.set(qn("w:sz"), str(size))
        elem.set(qn("w:space"), "0")
        elem.set(qn("w:color"), "auto")
        borders.append(elem)
    tbl_pr.append(borders)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_dialogue_txt(path: Path, replicas: List[Dict[str, str]]) -> None:
    lines = [f"{replica['role']}: {replica['text']}" for replica in replicas]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Транскрибировать аудио, определить спикеров и собрать DOCX."
    )
    parser.add_argument("--audio", required=True, help="Путь к аудиофайлу")
    parser.add_argument(
        "--output-docx",
        default=None,
        help="Путь к выходному DOCX (по умолчанию: <audio>.docx)",
    )
    parser.add_argument(
        "--transcript-dir",
        default="transcripts",
        help="Каталог для сохранения транскриптов и диаризации",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Путь или HF-репозиторий модели Whisper-MLX",
    )
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help="Путь или HF-репозиторий модели pyannote MLX",
    )
    parser.add_argument(
        "--speaker-map",
        default=None,
        help="Связать спикеров и роли: SPEAKER_00=К,SPEAKER_01=Т",
    )
    parser.add_argument(
        "--fio",
        default="",
        help="ФИО",
    )
    parser.add_argument(
        "--group",
        default="",
        help="Номер группы",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Дата",
    )
    parser.add_argument(
        "--topic",
        default="",
        help="Тема протокола",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Задание",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=0.5,
        help="Минимальная длительность сегмента (сек)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.35,
        help="Порог уверенности для тишины (0 = отключить)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=2,
        help="Число спикеров для кластеризации",
    )
    parser.add_argument(
        "--speaker-embedding-model",
        default=DEFAULT_SPEAKER_EMBEDDING_MODEL,
        help="Модель спикер-эмбеддингов (SpeechBrain)",
    )
    parser.add_argument(
        "--speaker-embedding-device",
        default=DEFAULT_SPEAKER_EMBEDDING_DEVICE,
        help="Устройство для спикер-эмбеддингов (cpu/mps/cuda)",
    )
    parser.add_argument(
        "--word-smooth-min-words",
        type=int,
        default=DEFAULT_WORD_SMOOTH_MIN_WORDS,
        help="Сглаживание word-диаризации: минимум слов в вставке",
    )
    parser.add_argument(
        "--word-prob-threshold",
        type=float,
        default=DEFAULT_WORD_PROB_THRESHOLD,
        help="Порог вероятности слова (0 = не фильтровать)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.2,
        help="Сшивать соседние сегменты одного спикера (сек)",
    )
    parser.add_argument(
        "--sandwich-max-duration",
        type=float,
        default=1.0,
        help="Макс. длительность «вставки» между одинаковыми спикерами (сек)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=160000,
        help="Размер чанка для диаризации (семплы, 10с при 16кГц)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=16000,
        help="Перекрытие чанков (семплы, 1с при 16кГц)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Уровень логирования (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--force-whisper",
        action="store_true",
        help="Перезапустить Whisper и обновить кэш",
    )
    parser.add_argument(
        "--no-word-timestamps",
        action="store_false",
        dest="word_timestamps",
        default=True,
        help="Отключить word timestamps в Whisper",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format=LOG_FORMAT)
    logging.info("Старт обработки")
    logging.info("Аудио: %s", args.audio)
    logging.info("Модель Whisper: %s", args.whisper_model)
    logging.info("Модель диаризации: %s", args.diarization_model)
    logging.info(
        "Параметры диаризации: min_duration=%.2f, silence_threshold=%.2f, chunk=%d, overlap=%d",
        args.min_segment_duration,
        args.silence_threshold,
        args.chunk_size,
        args.overlap,
    )
    logging.info(
        "Кластеризация: num_speakers=%d",
        args.max_speakers,
    )
    logging.info(
        "Эмбеддинги: model=%s device=%s",
        args.speaker_embedding_model,
        args.speaker_embedding_device,
    )
    logging.info(
        "Word-smoothing: min_words=%d",
        args.word_smooth_min_words,
    )
    logging.info(
        "Word-filter: prob_threshold=%.2f",
        args.word_prob_threshold,
    )
    logging.info(
        "Постобработка: merge_gap=%.2f, sandwich_max_duration=%.2f",
        args.merge_gap,
        args.sandwich_max_duration,
    )

    audio_path = Path(args.audio)
    output_docx = (
        Path(args.output_docx)
        if args.output_docx
        else audio_path.with_suffix(".docx")
    )

    transcript_root = Path(args.transcript_dir).expanduser()
    transcript_dir = transcript_root / audio_path.stem
    transcript_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Кэш: %s", transcript_dir)

    transcript_json_path = transcript_dir / "transcript.json"
    transcript_txt_path = transcript_dir / "transcript.txt"
    whisper_segments_path = transcript_dir / "whisper_segments.json"

    if transcript_json_path.exists() and not args.force_whisper:
        logging.info("Whisper: загрузка из кэша %s", transcript_json_path)
        whisper_result = load_json(transcript_json_path)
    else:
        logging.info("Whisper: запуск распознавания (word_timestamps=%s)", args.word_timestamps)
        whisper_result = transcribe_audio(
            str(audio_path),
            args.whisper_model,
            word_timestamps=args.word_timestamps,
        )
        save_json(transcript_json_path, whisper_result)
        logging.info("Whisper: сохранен кэш %s", transcript_json_path)

    if args.force_whisper or not transcript_txt_path.exists():
        save_text(transcript_txt_path, whisper_result.get("text", ""))
        logging.info("Whisper: сохранен текст %s", transcript_txt_path)

    if args.force_whisper or not whisper_segments_path.exists():
        whisper_segments = whisper_result.get("segments", [])
        save_json(whisper_segments_path, {"segments": whisper_segments})
        logging.info("Whisper: сегменты сохранены %s", whisper_segments_path)
    else:
        logging.info("Whisper: сегменты из кэша %s", whisper_segments_path)
        whisper_segments_payload = load_json(whisper_segments_path)
        whisper_segments = whisper_segments_payload.get("segments", [])

    diarization_post_path = transcript_dir / "diarization_post.json"
    diarization_payload = None
    if diarization_post_path.exists():
        diarization_payload = load_json(diarization_post_path)
        params = diarization_payload.get("params", {})
        if (
            diarization_payload.get("method") != "embedding_kmeans_v2"
            or params.get("num_speakers") != args.max_speakers
            or params.get("merge_gap") != args.merge_gap
            or params.get("sandwich_max_duration") != args.sandwich_max_duration
            or params.get("silence_threshold") != args.silence_threshold
            or params.get("min_segment_duration") != args.min_segment_duration
            or params.get("embedding_min_duration") != DEFAULT_EMBEDDING_MIN_DURATION
            or params.get("embedding_model") != args.speaker_embedding_model
            or params.get("embedding_device") != args.speaker_embedding_device
        ):
            diarization_payload = None

    if diarization_payload:
        logging.info("Диаризация: загрузка из кэша %s", diarization_post_path)
        diarization_segments = [
            SpeakerSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=str(seg["speaker"]),
            )
            for seg in diarization_payload.get("segments", [])
        ]
    else:
        diarization_raw_path = transcript_dir / "diarization.json"
        if diarization_raw_path.exists():
            logging.info("Диаризация: сырые сегменты из кэша %s", diarization_raw_path)
            diarization_raw_payload = load_json(diarization_raw_path)
            raw_segments = [
                SpeakerSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    speaker=str(seg.get("speaker", "RAW_00")),
                )
                for seg in diarization_raw_payload.get("segments", [])
            ]
        else:
            logging.info("Диаризация: запуск сегментации")
            raw_segments = diarize_audio_raw(
                str(audio_path),
                args.diarization_model,
                min_duration=args.min_segment_duration,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                silence_threshold=args.silence_threshold,
            )
            save_json(
                diarization_raw_path,
                {
                    "segments": [
                        {"start": s.start, "end": s.end, "speaker": s.speaker}
                        for s in raw_segments
                    ]
                },
            )
            logging.info("Диаризация: сырые сегменты сохранены %s", diarization_raw_path)

        waveform, sample_rate = load_audio(str(audio_path))
        logging.info("Диаризация: сырых сегментов %d", len(raw_segments))
        diarization_segments = cluster_segments_by_embeddings(
            waveform,
            sample_rate,
            raw_segments,
            num_speakers=args.max_speakers,
            embedding_model_id=args.speaker_embedding_model,
            embedding_device=args.speaker_embedding_device,
        )
        logging.info("Диаризация: сегментов после кластеризации %d", len(diarization_segments))
    diarization_segments = post_process_diarization(
        diarization_segments,
        merge_gap=args.merge_gap,
        sandwich_max_duration=args.sandwich_max_duration,
    )
    logging.info("Диаризация: сегментов после постобработки %d", len(diarization_segments))
    save_json(
        diarization_post_path,
        {
                "method": "embedding_kmeans_v2",
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker}
                for s in diarization_segments
            ],
            "params": {
                "num_speakers": args.max_speakers,
                "merge_gap": args.merge_gap,
                "sandwich_max_duration": args.sandwich_max_duration,
                "silence_threshold": args.silence_threshold,
                "min_segment_duration": args.min_segment_duration,
                "embedding_min_duration": DEFAULT_EMBEDDING_MIN_DURATION,
                    "embedding_model": args.speaker_embedding_model,
                    "embedding_device": args.speaker_embedding_device,
            },
        },
    )
    logging.info("Диаризация: сохранена постобработка %s", diarization_post_path)
    words = (
        extract_words(whisper_result, prob_threshold=args.word_prob_threshold)
        if args.word_timestamps
        else []
    )
    if args.word_timestamps and words:
        logging.info("Whisper: word timestamps %d", len(words))
        replicas = build_replicas_from_words(
            words,
            diarization_segments,
            smooth_min_words=args.word_smooth_min_words,
        )
    else:
        if args.word_timestamps:
            logging.warning(
                "Whisper: word timestamps отсутствуют в кэше, использую сегменты"
            )
        replicas = build_replicas(whisper_segments, diarization_segments)
    logging.info("Реплики: %d", len(replicas))

    explicit_map = parse_speaker_map(args.speaker_map)
    role_map = map_speakers_to_roles(replicas, explicit_map)
    for replica in replicas:
        replica["role"] = role_map.get(replica["speaker"], "К")

    metadata = {
        "ФИО": args.fio,
        "Номер группы": args.group,
        "Дата": args.date,
        "Тема протокола": args.topic,
        "Задание": args.task,
    }
    logging.info("Генерация DOCX: %s", output_docx)
    create_docx(
        output_path=str(output_docx),
        replicas=replicas,
        metadata=metadata,
    )
    dialogue_txt_path = output_docx.with_suffix(".txt")
    save_dialogue_txt(dialogue_txt_path, replicas)
    logging.info("Диалог TXT: %s", dialogue_txt_path)
    logging.info("Готово")


if __name__ == "__main__":
    main()
