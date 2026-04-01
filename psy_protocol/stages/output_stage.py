import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..docx_writer import create_docx
from ..io_utils import save_text
from ..models import Replica
from ..text_outputs import save_dialogue_txt, save_timed_dialogue_txt


def run_output(
    replicas: List[Replica],
    opts,
    audio_path: Path,
    cache_dir: Path,
) -> Tuple[Path, Path]:
    """Stage 5: generate DOCX and TXT output files."""
    output_docx = (
        Path(opts.output_docx).expanduser()
        if opts.output_docx
        else audio_path.with_suffix('.docx')
    )

    replicas_dicts = [r.to_dict() for r in replicas]

    metadata = {
        'ФИО': opts.fio,
        'Номер группы': opts.group,
        'Дата': opts.date,
        'Тема протокола': opts.topic,
        'Задание': opts.task,
    }

    logging.info('Generating DOCX: %s', output_docx)
    try:
        create_docx(
            output_path=str(output_docx),
            replicas=replicas_dicts,
            metadata=metadata,
        )
    except Exception:
        logging.exception('DOCX generation failed: %s', output_docx)
        raise
    if not output_docx.exists():
        raise RuntimeError(f'DOCX output file was not created: {output_docx}')
    logging.info('DOCX ready: %s (%d bytes)', output_docx, output_docx.stat().st_size)

    dialogue_txt_path = output_docx.with_suffix('.txt')
    save_dialogue_txt(dialogue_txt_path, replicas_dicts)
    logging.info('Dialogue TXT: %s', dialogue_txt_path)

    timed_txt_path = cache_dir / 'timed_dialogue.txt'
    save_timed_dialogue_txt(timed_txt_path, replicas_dicts)
    logging.info('Timed dialogue TXT: %s', timed_txt_path)

    transcript_txt_path = cache_dir / 'transcript.txt'
    logging.info('Whisper TXT (raw): %s', transcript_txt_path)

    return output_docx, dialogue_txt_path
