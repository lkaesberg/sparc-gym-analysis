#!/usr/bin/env python3
"""
Script to calculate average number of tokens for model answers in JSONL files.
Uses tiktoken for accurate token counting.
Each row in the JSONL is one puzzle - calculates per-puzzle and overall averages.
"""

import os
import json
import glob
from pathlib import Path
import re
import statistics
import tiktoken

# Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    if not text:
        return 0
    return len(ENCODING.encode(text))


def extract_model_name(filename: str) -> str:
    """Extract model name from filename."""
    basename = os.path.basename(filename)
    model_name = re.sub(r'_gym(_traceback|_visual)?\.jsonl$', '', basename)
    model_name = model_name.replace('_', '/', 1)
    return model_name


def get_file_type(filename: str) -> str:
    """Determine file type from filename."""
    if '_traceback' in filename:
        return 'traceback'
    elif '_visual' in filename:
        return 'visual'
    else:
        return 'standard'


def analyze_jsonl_file(filepath: str) -> dict:
    """
    Analyze a JSONL file and return token statistics.
    Each row = one puzzle. We sum all messages per puzzle to get tokens per puzzle.
    """
    stats = {
        'total_puzzles': 0,
        'puzzles_with_messages': 0,
        'total_tokens': 0,
        'tokens_per_puzzle': [],
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    stats['total_puzzles'] += 1

                    result = record.get('result', {})
                    messages = result.get('message', [])

                    puzzle_tokens = 0

                    if messages:
                        if isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, str) and msg:
                                    puzzle_tokens += count_tokens(msg)
                        elif isinstance(messages, str):
                            puzzle_tokens = count_tokens(messages)

                    if puzzle_tokens > 0:
                        stats['puzzles_with_messages'] += 1
                        stats['total_tokens'] += puzzle_tokens
                        stats['tokens_per_puzzle'].append(puzzle_tokens)
                    else:
                        stats['tokens_per_puzzle'].append(0)

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return stats


def main():
    results_dir = Path(__file__).resolve().parent / "results" / "spatial_gym"

    jsonl_files = glob.glob(os.path.join(results_dir, "*_gym.jsonl"))
    jsonl_files += glob.glob(os.path.join(results_dir, "*_gym_traceback.jsonl"))
    jsonl_files += glob.glob(os.path.join(results_dir, "*_gym_visual.jsonl"))

    jsonl_files = sorted(set(jsonl_files))

    print("=" * 130)
    print("TOKEN ANALYSIS FOR MODEL ANSWERS (using tiktoken cl100k_base encoding)")
    print("=" * 130)
    print()

    all_stats = []

    for filepath in jsonl_files:
        model_name = extract_model_name(filepath)
        file_type = get_file_type(filepath)
        stats = analyze_jsonl_file(filepath)

        num_puzzles = stats['total_puzzles']
        tokens_list = stats['tokens_per_puzzle']

        if num_puzzles > 0 and tokens_list:
            avg_tokens = statistics.mean(tokens_list)
            median_tokens = statistics.median(tokens_list)
            min_tokens = min(tokens_list)
            max_tokens = max(tokens_list)
            total_tokens = stats['total_tokens']

            if len(tokens_list) > 1:
                std_tokens = statistics.stdev(tokens_list)
            else:
                std_tokens = 0
        else:
            avg_tokens = median_tokens = min_tokens = max_tokens = total_tokens = std_tokens = 0

        all_stats.append({
            'model': model_name,
            'file_type': file_type,
            'num_puzzles': num_puzzles,
            'avg_tokens': avg_tokens,
            'median_tokens': median_tokens,
            'std_tokens': std_tokens,
            'min_tokens': min_tokens,
            'max_tokens': max_tokens,
            'total_tokens': total_tokens,
        })

    all_stats.sort(key=lambda x: (x['model'], x['file_type']))

    print("STANDARD GYM FILES - Tokens per Puzzle")
    print("-" * 130)
    header = f"{'Model':<50} {'Puzzles':<10} {'Avg Tokens':<15} {'Median':<12} {'Std Dev':<12} {'Min':<10} {'Max':<12} {'Total Tokens':<15}"
    print(header)
    print("-" * 130)

    standard_stats = [s for s in all_stats if s['file_type'] == 'standard' and s['num_puzzles'] > 0]
    standard_stats.sort(key=lambda x: x['avg_tokens'], reverse=True)

    for stat in standard_stats:
        row = f"{stat['model']:<50} {stat['num_puzzles']:<10} {stat['avg_tokens']:<15.1f} {stat['median_tokens']:<12.1f} {stat['std_tokens']:<12.1f} {stat['min_tokens']:<10} {stat['max_tokens']:<12} {stat['total_tokens']:<15,}"
        print(row)

    print()

    print("TRACEBACK GYM FILES - Tokens per Puzzle")
    print("-" * 130)
    print(header)
    print("-" * 130)

    traceback_stats = [s for s in all_stats if s['file_type'] == 'traceback' and s['num_puzzles'] > 0]
    traceback_stats.sort(key=lambda x: x['avg_tokens'], reverse=True)

    for stat in traceback_stats:
        row = f"{stat['model']:<50} {stat['num_puzzles']:<10} {stat['avg_tokens']:<15.1f} {stat['median_tokens']:<12.1f} {stat['std_tokens']:<12.1f} {stat['min_tokens']:<10} {stat['max_tokens']:<12} {stat['total_tokens']:<15,}"
        print(row)

    print()

    visual_stats = [s for s in all_stats if s['file_type'] == 'visual' and s['num_puzzles'] > 0]
    if visual_stats:
        print("VISUAL GYM FILES - Tokens per Puzzle")
        print("-" * 130)
        print(header)
        print("-" * 130)

        visual_stats.sort(key=lambda x: x['avg_tokens'], reverse=True)
        for stat in visual_stats:
            row = f"{stat['model']:<50} {stat['num_puzzles']:<10} {stat['avg_tokens']:<15.1f} {stat['median_tokens']:<12.1f} {stat['std_tokens']:<12.1f} {stat['min_tokens']:<10} {stat['max_tokens']:<12} {stat['total_tokens']:<15,}"
            print(row)
        print()

    csv_path = os.path.join(results_dir, "token_analysis.csv")
    with open(csv_path, 'w') as f:
        f.write("Model,File Type,Num Puzzles,Avg Tokens per Puzzle,Median Tokens,Std Dev,Min Tokens,Max Tokens,Total Tokens\n")
        for stat in all_stats:
            f.write(f"{stat['model']},{stat['file_type']},{stat['num_puzzles']},{stat['avg_tokens']:.1f},{stat['median_tokens']:.1f},{stat['std_tokens']:.1f},{stat['min_tokens']},{stat['max_tokens']},{stat['total_tokens']}\n")

    print(f"Results exported to: {csv_path}")


if __name__ == "__main__":
    main()
