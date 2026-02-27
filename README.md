# psy-protocol

CLI tool to transcribe audio with Whisper-MLX, diarize speakers with the
`pyannote-segmentation-3.0-mlx` model, and generate `.docx` + dialogue `.txt`.

## Setup

- Install ffmpeg (required by Whisper):
  - macOS: `brew install ffmpeg`
- Install dependencies:
  - `python -m pip install -r requirements.txt`

## Usage

```sh
python app.py --audio /path/to/audio.wav
```

Outputs:

- `<audio>.docx` (filled table)
- `<audio>.txt` (dialogue with roles)
- `transcripts/<audio-stem>/transcript.txt`
- `transcripts/<audio-stem>/transcript.json`
- `transcripts/<audio-stem>/whisper_segments.json`
- `transcripts/<audio-stem>/diarization.json`
- `transcripts/<audio-stem>/diarization_post.json`

## Common options

- Use a local Whisper model path:
  - `--whisper-model ~/.cache/mlx/large-v3-turbo`
- Override diarization model weights:
  - `--diarization-model /path/to/weights.npz`
- Map speakers to roles explicitly:
  - `--speaker-map SPEAKER_00=C,SPEAKER_01=T`

## Programmatic API

You can run the processing pipeline directly from Python without CLI parsing:

```python
from psy_protocol import ProcessingOptions, process_audio_file

options = ProcessingOptions(max_speakers=2)
docx_path, txt_path = process_audio_file("/path/to/audio.wav", options)
```

## Telegram Bot

This repo includes `bot.py` (aiogram v3) that accepts Telegram `voice`, `audio`,
or audio `document` messages and returns generated `.txt` + `.docx`.

### Configure

Create `.env` in project root:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_LOCAL_API_PORT=8081
TELEGRAM_BOT_API_BASE_URL=http://localhost:8081
TELEGRAM_LOCAL_SERVER_FILE_ROOT=/var/lib/telegram-bot-api
TELEGRAM_LOCAL_HOST_FILE_ROOT=./telegram-bot-api-data

# Optional overrides:
# PSY_WHISPER_MODEL=~/.cache/mlx/large-v3-turbo
# PSY_DIARIZATION_MODEL=mlx-community/pyannote-segmentation-3.0-mlx
# PSY_MAX_SPEAKERS=2
```

### Run local Telegram Bot API server

```sh
docker compose up -d telegram-bot-api
```

If you change `TELEGRAM_LOCAL_HOST_FILE_ROOT`, recreate the service:

```sh
docker compose down
docker compose up -d telegram-bot-api
```

### Run VictoriaMetrics and scrape Telegram metrics (8082)

```sh
docker compose up -d victoria-metrics
```

VictoriaMetrics UI/API will be available at:

- `http://localhost:8428`

Scrape target is configured as:

- `telegram-bot-api:8082` via `monitoring/prometheus.yml`

### Run Grafana with VictoriaMetrics datasource

```sh
docker compose up -d grafana
```

Grafana will be available at:

- `http://localhost:3000`

Default credentials (from `.env`):

- login: `admin`
- password: `admin`

Datasource `VictoriaMetrics` is provisioned automatically from:

- `monitoring/grafana/provisioning/datasources/victoriametrics.yml`

Dashboard provisioning is enabled from:

- `monitoring/grafana/provisioning/dashboards/default.yml`
- `monitoring/grafana/dashboards/telegram-bot-api-overview.json`

### Run bot

```sh
python bot.py
```