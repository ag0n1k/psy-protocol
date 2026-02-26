# psy-protocol

CLI tool to transcribe audio with Whisper-MLX, diarize speakers with the
`pyannote-segmentation-3.0-mlx` model, and populate the `template.docx` table.

## Setup

- Install ffmpeg (required by Whisper):
  - macOS: `brew install ffmpeg`
- Install dependencies:
  - `python -m pip install -r requirements.txt`

## Usage

```sh
python app.py --audio /path/to/audio.wav --template template.docx
```

Outputs:

- `<audio>.docx` (filled table)
- `transcripts/<audio-stem>/transcript.txt`
- `transcripts/<audio-stem>/transcript.json`
- `transcripts/<audio-stem>/whisper_segments.json`
- `transcripts/<audio-stem>/diarization.json`

## Common options

- Use a local Whisper model path:
  - `--whisper-model ~/.cache/mlx/large-v3-turbo`
- Override diarization model weights:
  - `--diarization-model /path/to/weights.npz`
- Mapls speakers to roles explicitly:
  - `--speaker-map SPEAKER_00=C,SPEAKER_01=T`
- Table targeting:
  - `--table-index 0 --data-start-row 1`