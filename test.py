#!/usr/bin/env python3
"""Test script for evaluating audio processing quality."""
import argparse
import dataclasses
import difflib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psy_protocol import ProcessingOptions, process_audio_file

DEFAULT_TESTS_DIR = Path(__file__).parent / 'tests'
TEST_NUMBERS = list(range(1, 6))
TRANSCRIPTION_METHODS = ['whisper', 'qwen_asr']

CONFIGS = {
    'default': ProcessingOptions(),
    'noisy': ProcessingOptions(
        silence_threshold=0.5,
        merge_gap=0.3,
    ),
    'interrupts': ProcessingOptions(
        sandwich_max_duration=2.0,
        word_smooth_min_words=3,
    ),
    'sensitive': ProcessingOptions(
        silence_threshold=0.25,
        merge_gap=0.1,
        min_segment_duration=0.3,
    ),
}

CLUSTERING_METHODS = ['kmeans', 'spectral', 'agglomerative']
DIARIZATION_METHODS = ['pyannote_pipeline', 'custom_mlx', 'aufklarer_mlx', 'llm']


def parse_dialogue(path: Path) -> List[Tuple[str, str]]:
    turns = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        if ': ' in line:
            speaker, text = line.split(': ', 1)
            turns.append((speaker.strip(), text.strip()))
    return turns


def seq_ratio(a: str, b: str) -> float:
    a_words = a.lower().split()
    b_words = b.lower().split()
    if not b_words:
        return 1.0 if not a_words else 0.0
    matcher = difflib.SequenceMatcher(None, a_words, b_words)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    return matching / len(b_words)


def text_accuracy(
    output_turns: List[Tuple[str, str]],
    goal_turns: List[Tuple[str, str]],
) -> float:
    out_text = ' '.join(text for _, text in output_turns)
    goal_text = ' '.join(text for _, text in goal_turns)
    return seq_ratio(out_text, goal_text) * 100


def speaker_accuracy(
    output_turns: List[Tuple[str, str]],
    goal_turns: List[Tuple[str, str]],
) -> Tuple[float, bool]:
    def group_text(turns: List[Tuple[str, str]], swap: bool = False) -> Tuple[str, str]:
        k_parts, t_parts = [], []
        for speaker, text in turns:
            is_k = (speaker == 'К') if not swap else (speaker != 'К')
            if is_k:
                k_parts.append(text)
            else:
                t_parts.append(text)
        return ' '.join(k_parts), ' '.join(t_parts)

    goal_k, goal_t = group_text(goal_turns)
    best_score = -1.0
    best_swap = False
    for swap in (False, True):
        out_k, out_t = group_text(output_turns, swap=swap)
        score = (seq_ratio(out_k, goal_k) + seq_ratio(out_t, goal_t)) / 2
        if score > best_score:
            best_score = score
            best_swap = swap
    return best_score * 100, best_swap


def find_audio(tests_dir: Path, test_num: int) -> Optional[Path]:
    for ext in ('.wav', '.ogg', '.mp3', '.m4a', '.flac'):
        p = tests_dir / f'{test_num}{ext}'
        if p.exists():
            return p
    return None


def run_test(
    test_num: int,
    config_name: str,
    options: ProcessingOptions,
    no_process: bool,
    tests_dir: Optional[Path] = None,
) -> Dict:
    tests_dir = tests_dir or DEFAULT_TESTS_DIR
    output_dir = tests_dir / 'output' / config_name
    audio_path = find_audio(tests_dir, test_num)
    if audio_path is None:
        return {'test': test_num, 'error': 'audio not found', 'duration': None}
    goal_path = tests_dir / f'{test_num}_goal.txt'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_docx = output_dir / f'{test_num}.docx'
    output_txt = output_dir / f'{test_num}.txt'

    duration: Optional[float] = None
    if not no_process:
        opts = dataclasses.replace(
            options,
            output_docx=output_docx,
            transcript_dir=tests_dir / 'transcripts',
        )
        t0 = time.time()
        process_audio_file(audio_path, opts)
        duration = time.time() - t0

    if not output_txt.exists():
        return {'test': test_num, 'error': 'output not found', 'duration': duration}

    output_turns = parse_dialogue(output_txt)
    goal_turns = parse_dialogue(goal_path)

    if not output_turns:
        return {'test': test_num, 'error': 'empty output', 'duration': duration}

    text_acc = text_accuracy(output_turns, goal_turns)
    spk_acc, swap = speaker_accuracy(output_turns, goal_turns)
    overall = (text_acc + spk_acc) / 2

    return {
        'test': test_num,
        'text': text_acc,
        'speaker': spk_acc,
        'overall': overall,
        'swap': swap,
        'duration': duration,
    }


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return '—'
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def print_results(config_name: str, results: List[Dict]) -> None:
    print(f'\n=== Config: {config_name} ===\n')
    header = f'{"Test":>4}  {"Text%":>6}  {"Speaker%":>8}  {"Overall%":>8}  {"Swap":>4}  {"Duration":>8}'
    separator = '-' * len(header)
    print(header)
    print(separator)

    valid = [r for r in results if 'error' not in r]
    for r in results:
        if 'error' in r:
            print(f'{r["test"]:>4}  ERROR: {r["error"]}')
            continue
        swap_str = 'YES' if r['swap'] else 'NO'
        print(
            f'{r["test"]:>4}  {r["text"]:>6.1f}  {r["speaker"]:>8.1f}'
            f'  {r["overall"]:>8.1f}  {swap_str:>4}  {format_duration(r["duration"]):>8}'
        )

    if valid:
        avg_text = sum(r['text'] for r in valid) / len(valid)
        avg_spk = sum(r['speaker'] for r in valid) / len(valid)
        avg_overall = sum(r['overall'] for r in valid) / len(valid)
        print(separator)
        print(f'{"AVG":>4}  {avg_text:>6.1f}  {avg_spk:>8.1f}  {avg_overall:>8.1f}')


def print_comparison(all_results: Dict[str, List[Dict]]) -> None:
    config_names = list(all_results.keys())
    print(f'\n=== Config comparison (Overall%) ===\n')

    header = f'{"Test":>4}' + ''.join(f'  {c:>10}' for c in config_names)
    separator = '-' * len(header)
    print(header)
    print(separator)

    test_nums = sorted({r['test'] for results in all_results.values() for r in results})
    avgs: Dict[str, List[float]] = {c: [] for c in config_names}

    for test_num in test_nums:
        row = f'{test_num:>4}'
        for config_name in config_names:
            r = next((x for x in all_results[config_name] if x['test'] == test_num), None)
            if r is None or 'error' in r:
                row += f'  {"—":>10}'
            else:
                avgs[config_name].append(r['overall'])
                row += f'  {r["overall"]:>10.1f}'
        print(row)

    avg_row = f'{"AVG":>4}'
    for config_name in config_names:
        vals = avgs[config_name]
        avg_row += f'  {sum(vals) / len(vals):>10.1f}' if vals else f'  {"—":>10}'
    print(separator)
    print(avg_row)


def _print_clustering_comparison(all_results: Dict[str, List[Dict]]) -> None:
    method_keys = list(all_results.keys())
    col_w = 14

    for metric, label in (('speaker', 'Speaker%'), ('overall', 'Overall%')):
        print(f'\n=== Diarization comparison ({label}) ===\n')
        header = f'{"Test":>4}' + ''.join(f'  {k.split(":")[-1]:>{col_w}}' for k in method_keys)
        separator = '-' * len(header)
        print(header)
        print(separator)

        test_nums = sorted({r['test'] for results in all_results.values() for r in results})
        avgs: Dict[str, List[float]] = {k: [] for k in method_keys}

        for test_num in test_nums:
            row = f'{test_num:>4}'
            for key in method_keys:
                r = next((x for x in all_results[key] if x['test'] == test_num), None)
                if r is None or 'error' in r:
                    row += f'  {"—":>{col_w}}'
                else:
                    avgs[key].append(r[metric])
                    row += f'  {r[metric]:>{col_w}.1f}'
            print(row)

        avg_row = f'{"AVG":>4}'
        for key in method_keys:
            vals = avgs[key]
            avg_row += f'  {sum(vals)/len(vals):>{col_w}.1f}' if vals else f'  {"—":>{col_w}}'
        print(separator)
        print(avg_row)

    print('\n=== Diarization comparison (Duration) ===\n')
    header = f'{"Test":>4}' + ''.join(f'  {k.split(":")[-1]:>{col_w}}' for k in method_keys)
    separator = '-' * len(header)
    print(header)
    print(separator)
    test_nums = sorted({r['test'] for results in all_results.values() for r in results})
    for test_num in test_nums:
        row = f'{test_num:>4}'
        for key in method_keys:
            r = next((x for x in all_results[key] if x['test'] == test_num), None)
            if r is None or 'error' in r:
                row += f'  {"—":>{col_w}}'
            else:
                row += f'  {format_duration(r["duration"]):>{col_w}}'
        print(row)
    print(separator)


def _print_transcription_comparison(all_results: Dict[str, List[Dict]]) -> None:
    method_keys = list(all_results.keys())
    print('\n=== Transcription method comparison (Text%) ===\n')

    header = f'{"Test":>4}' + ''.join(f'  {k.split(":")[-1]:>12}' for k in method_keys)
    separator = '-' * len(header)
    print(header)
    print(separator)

    test_nums = sorted({r['test'] for results in all_results.values() for r in results})
    avgs: Dict[str, List[float]] = {k: [] for k in method_keys}

    for test_num in test_nums:
        row = f'{test_num:>4}'
        for key in method_keys:
            r = next((x for x in all_results[key] if x['test'] == test_num), None)
            if r is None or 'error' in r:
                row += f'  {"—":>12}'
            else:
                avgs[key].append(r['text'])
                row += f'  {r["text"]:>12.1f}'
        print(row)

    avg_row = f'{"AVG":>4}'
    for key in method_keys:
        vals = avgs[key]
        avg_row += f'  {sum(vals) / len(vals):>12.1f}' if vals else f'  {"—":>12}'
    print(separator)
    print(avg_row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate audio processing quality')
    parser.add_argument(
        '--config',
        choices=list(CONFIGS.keys()),
        default='default',
        help='Config to use (default: default)',
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Run all configs and compare results in a summary table',
    )
    parser.add_argument(
        '--test',
        nargs='+',
        type=int,
        choices=TEST_NUMBERS,
        metavar='N',
        help='Run only specific tests (e.g. --test 1 3)',
    )
    parser.add_argument(
        '--no-process',
        action='store_true',
        help='Skip pipeline, recompute metrics from existing output files',
    )
    parser.add_argument(
        '--compare-clustering',
        action='store_true',
        help='Run all clustering methods and compare Speaker%% results',
    )
    parser.add_argument(
        '--compare-diarization',
        action='store_true',
        help='Compare pyannote_pipeline vs custom_mlx diarization methods (Speaker%%)',
    )
    parser.add_argument(
        '--diarization-method',
        nargs='+',
        choices=DIARIZATION_METHODS,
        metavar='METHOD',
        help='Run only specific diarization methods (e.g. --diarization-method llm)',
    )
    parser.add_argument(
        '--compare-transcription',
        action='store_true',
        help='Compare whisper vs qwen_asr transcription methods (Text%%)',
    )
    parser.add_argument(
        '--transcription-method',
        nargs='+',
        choices=TRANSCRIPTION_METHODS,
        metavar='METHOD',
        help='Run only specific transcription methods (e.g. --transcription-method qwen_asr)',
    )
    parser.add_argument(
        '--tests-dir',
        default=None,
        help='Path to directory with test audio and goal files',
    )
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir).expanduser() if args.tests_dir else DEFAULT_TESTS_DIR
    test_nums = args.test or TEST_NUMBERS

    if args.compare_transcription:
        all_results: Dict[str, List[Dict]] = {}
        methods = args.transcription_method or TRANSCRIPTION_METHODS
        for i, method in enumerate(methods):
            options = dataclasses.replace(
                ProcessingOptions(),
                transcription_method=method,
                force_whisper=True,
                # diarization cache is reused across transcription methods
                force_diarization=(i == 0),
            )
            config_key = f'transcription:{method}'
            config_dir = f'transcription_{method}'
            results = []
            for test_num in test_nums:
                print(f'[{method}] test {test_num}...', end=' ', flush=True)
                result = run_test(test_num, config_dir, options, args.no_process, tests_dir)
                if 'error' in result:
                    print(f'ERROR: {result["error"]}')
                else:
                    print(f'OK  text={result["text"]:.1f}%')
                results.append(result)
            all_results[config_key] = results

        _print_transcription_comparison(all_results)
        return

    if args.compare_diarization:
        all_results: Dict[str, List[Dict]] = {}
        methods = args.diarization_method or DIARIZATION_METHODS
        transcription_method = (args.transcription_method or [None])[0]
        for i, method in enumerate(methods):
            extra = {}
            if transcription_method:
                extra['transcription_method'] = transcription_method
                # first diarization method runs transcription; rest reuse cache
                extra['force_whisper'] = (i == 0)
            options = dataclasses.replace(
                ProcessingOptions(),
                diarization_method=method,
                force_diarization=True,
                **extra,
            )
            suffix = f'_{transcription_method}' if transcription_method else ''
            config_key = f'diarization:{method}'
            config_dir = f'diarization_{method}{suffix}'
            results = []
            for test_num in test_nums:
                print(f'[{method}] test {test_num}...', end=' ', flush=True)
                result = run_test(test_num, config_dir, options, args.no_process, tests_dir)
                if 'error' in result:
                    print(f'ERROR: {result["error"]}')
                else:
                    print(f'OK  overall={result["overall"]:.1f}%  {format_duration(result["duration"])}')
                results.append(result)
            all_results[config_key] = results

        _print_clustering_comparison(all_results)
        return

    if args.compare_clustering:
        all_results: Dict[str, List[Dict]] = {}
        for method in CLUSTERING_METHODS:
            options = dataclasses.replace(
                ProcessingOptions(), clustering_method=method, force_diarization=True,
            )
            config_key = f'clustering:{method}'
            config_dir = f'clustering_{method}'
            results = []
            for test_num in test_nums:
                print(f'[{method}] test {test_num}...', end=' ', flush=True)
                result = run_test(test_num, config_dir, options, args.no_process, tests_dir)
                if 'error' in result:
                    print(f'ERROR: {result["error"]}')
                else:
                    print(f'OK  speaker={result["speaker"]:.1f}%')
                results.append(result)
            all_results[config_key] = results

        _print_clustering_comparison(all_results)
        return

    configs_to_run = list(CONFIGS.keys()) if args.compare_all else [args.config]

    all_results_configs: Dict[str, List[Dict]] = {}
    for config_name in configs_to_run:
        options = CONFIGS[config_name]
        results = []
        for test_num in test_nums:
            print(f'[{config_name}] test {test_num}...', end=' ', flush=True)
            result = run_test(test_num, config_name, options, args.no_process, tests_dir)
            if 'error' in result:
                print(f'ERROR: {result["error"]}')
            else:
                print(f'OK  overall={result["overall"]:.1f}%')
            results.append(result)
        all_results_configs[config_name] = results

    for config_name, results in all_results_configs.items():
        print_results(config_name, results)

    if args.compare_all and len(all_results_configs) > 1:
        print_comparison(all_results_configs)


if __name__ == '__main__':
    main()
