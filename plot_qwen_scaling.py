"""
Script to create a comparison of Qwen 3 model scaling (0.6B, 4B, 14B, 32B) 
across SPARC and SPARC-Gym variants.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
)

# Model sizes in order
QWEN_SIZES = ['0.6B', '4B', '14B', '32B']
QWEN_PARAMS = [0.6, 4, 14, 32]  # For x-axis scaling

# Colors
COLORS = {
    'sparc': '#7B68EE',      # Medium slate blue for SPARC
    'gym': '#A47AFF',        # Lighter purple for SPARC-Gym
}


def extract_accuracy_from_stats(stats_file):
    """Extract accuracy percentage from a stats CSV file."""
    df = pd.read_csv(stats_file)
    
    solved_row = df[df['Metric'] == 'Correctly Solved']
    if len(solved_row) > 0:
        percentage_str = solved_row['Percentage'].values[0]
        match = re.search(r'([\d.]+)%', str(percentage_str))
        if match:
            return float(match.group(1))
    return 0.0


def extract_difficulty_stats(stats_file):
    """Extract accuracy by difficulty level."""
    df = pd.read_csv(stats_file)
    
    difficulty_pcts = {}
    for d in range(1, 6):
        diff_row = df[df['Metric'] == f'Difficulty {d} Solved']
        if len(diff_row) > 0:
            value_str = diff_row['Value'].values[0]
            match = re.match(r'(\d+)/(\d+)', str(value_str))
            if match:
                solved = int(match.group(1))
                total = int(match.group(2))
                difficulty_pcts[d] = 100.0 * solved / total if total > 0 else 0
    return difficulty_pcts


def load_qwen_data(results_dir):
    """Load SPARC and SPARC-Gym data for all Qwen sizes."""
    results_path = Path(results_dir)
    
    data = {
        'sparc': {},
        'gym': {},
        'sparc_diff': {},
        'gym_diff': {},
        'gym_path_stats': {},  # Path length statistics
        'sparc_tokens': {},  # Tokens per puzzle for SPARC
        'gym_tokens': {},  # Tokens per puzzle for SPARC-Gym
    }
    
    # Load token cache
    token_cache = results_path / "token_cache.csv"
    if token_cache.exists():
        token_df = pd.read_csv(token_cache)
        for size in QWEN_SIZES:
            # Match Qwen/Qwen3-{size} in gym
            row_gym = token_df[(token_df['Model'] == f'Qwen/Qwen3-{size}') & (token_df['File Type'] == 'gym')]
            if len(row_gym) > 0:
                data['gym_tokens'][size] = row_gym['Avg Tokens per Puzzle'].values[0]
            
            # Match Qwen/Qwen3-{size} in sparc
            row_sparc = token_df[(token_df['Model'] == f'Qwen/Qwen3-{size}') & (token_df['File Type'] == 'sparc')]
            if len(row_sparc) > 0:
                data['sparc_tokens'][size] = row_sparc['Avg Tokens per Puzzle'].values[0]
    
    for size in QWEN_SIZES:
        # SPARC stats
        sparc_file = results_path / f"Qwen_Qwen3-{size}_stats.csv"
        if sparc_file.exists():
            data['sparc'][size] = extract_accuracy_from_stats(sparc_file)
            data['sparc_diff'][size] = extract_difficulty_stats(sparc_file)
        
        # SPARC-Gym stats
        gym_file = results_path / f"Qwen_Qwen3-{size}_gym_stats.csv"
        if gym_file.exists():
            data['gym'][size] = extract_accuracy_from_stats(gym_file)
            data['gym_diff'][size] = extract_difficulty_stats(gym_file)
        
        # Load JSONL for path length analysis
        gym_jsonl = results_path / f"Qwen_Qwen3-{size}_gym.jsonl"
        if gym_jsonl.exists():
            data['gym_path_stats'][size] = extract_path_stats(gym_jsonl)
    
    return data


def extract_path_stats(jsonl_file):
    """Extract path length statistics from JSONL file."""
    path_lengths = []
    steps_taken = []
    solved_path_lengths = []
    solved_steps = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                
                # Get true solution path length
                if 'solutions' in entry and len(entry['solutions']) > 0:
                    path_len = len(entry['solutions'][0]['path']) - 1  # edges = nodes - 1
                    path_lengths.append(path_len)
                
                # Get steps taken by model
                result = entry.get('result', {})
                steps = result.get('steps_taken', 0)
                steps_taken.append(steps)
                
                # For solved puzzles
                if result.get('solved', False):
                    if 'solutions' in entry and len(entry['solutions']) > 0:
                        solved_path_lengths.append(len(entry['solutions'][0]['path']) - 1)
                    solved_steps.append(steps)
    
    return {
        'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
        'avg_steps': np.mean(steps_taken) if steps_taken else 0,
        'avg_solved_path': np.mean(solved_path_lengths) if solved_path_lengths else 0,
        'avg_solved_steps': np.mean(solved_steps) if solved_steps else 0,
        'efficiency': np.mean(solved_path_lengths) / np.mean(solved_steps) if solved_steps and np.mean(solved_steps) > 0 else 0,
    }


def create_qwen_scaling_plot(results_dir, output_path=None):
    """Create a plot showing Qwen scaling across SPARC and SPARC-Gym."""
    setup_plot_style(use_latex=True)
    
    data = load_qwen_data(results_dir)
    
    # Accuracy values for convenience
    sparc_accs = [data['sparc'].get(size, 0) for size in QWEN_SIZES]
    gym_accs = [data['gym'].get(size, 0) for size in QWEN_SIZES]
    
    # --- Single subplot: Accuracy vs compute (FLOPs) for both variants ---
    fig, ax2 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_INCHES, 2.5))
    
    # Get tokens per puzzle for both variants
    sparc_tokens = [data['sparc_tokens'].get(size, 0) for size in QWEN_SIZES]
    gym_tokens = [data['gym_tokens'].get(size, 0) for size in QWEN_SIZES]
    
    # Estimate FLOPs: ~2 × params × tokens (standard transformer inference estimate)
    # Use raw FLOPs (log-scaled axis keeps values readable)
    sparc_flops = [2 * QWEN_PARAMS[i] * sparc_tokens[i] for i in range(len(QWEN_SIZES))]
    gym_flops = [2 * QWEN_PARAMS[i] * gym_tokens[i] for i in range(len(QWEN_SIZES))]
    
    # Plot accuracy vs FLOPs
    # Use lines with markers so scaling trend is easier to see
    ax2.plot(sparc_flops, sparc_accs, label='SPARC',
             color=COLORS['sparc'], marker='o', linewidth=1.5)
    ax2.plot(gym_flops, gym_accs, label='SPARC-Gym',
             color=COLORS['gym'], marker='s', linewidth=1.5)
    
    # Annotate points with model size and a thin white outline for readability
    for size, x_val, y_val in zip(QWEN_SIZES, sparc_flops, sparc_accs):
        if x_val > 0 and y_val > 0:
            txt = ax2.annotate(
                size,
                xy=(x_val, y_val),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=COLORS["sparc"],
            )
            txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])

    for size, x_val, y_val in zip(QWEN_SIZES, gym_flops, gym_accs):
        if x_val > 0 and y_val > 0:
            txt = ax2.annotate(
                size,
                xy=(x_val, y_val),
                xytext=(4, -10),
                textcoords="offset points",
                fontsize=8,
                color=COLORS["gym"],
            )
            txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
    
    ax2.set_xscale('log')
    ax2.set_ylabel('Accuracy (\\%)')
    ax2.set_xlabel('Estimated Compute (FLOPs, log scale)')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.close(fig)
    
    return data


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "qwen_scaling.pdf"
    output_png = Path(__file__).parent / "qwen_scaling.png"
    
    print("=" * 60)
    print("Qwen 3 Model Scaling Comparison (SPARC vs SPARC-Gym)")
    print("=" * 60)
    
    data = create_qwen_scaling_plot(results_dir, output_pdf)
    create_qwen_scaling_plot(results_dir, output_png)
    
    # Print summary
    print("\nAccuracy by model size:")
    print("-" * 50)
    print(f"{'Size':<10} {'SPARC':<12} {'SPARC-Gym':<12} {'Diff':<10}")
    print("-" * 50)
    for size in QWEN_SIZES:
        sparc = data['sparc'].get(size, 0)
        gym = data['gym'].get(size, 0)
        diff = gym - sparc
        print(f"{size:<10} {sparc:>6.1f}%      {gym:>6.1f}%      {diff:>+6.1f}%")
    print("-" * 50)
    
    print("\nKey observations:")
    print(f"  - SPARC-Gym helps smaller models (0.6B: {data['gym']['0.6B'] - data['sparc']['0.6B']:+.1f}%)")
    print(f"  - SPARC-Gym helps larger models (32B: {data['gym']['32B'] - data['sparc']['32B']:+.1f}%)")


if __name__ == "__main__":
    main()
