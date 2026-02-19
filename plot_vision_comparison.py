"""
Script to create a comparison between Qwen3-VL-32B (vision), Qwen3-VL-32B (text), and Qwen3-32B (text-only).
Three subplots: overall accuracy, accuracy by difficulty (line plot), and navigation outcome.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
)

MODELS = {
    'text': {
        'file': 'Qwen_Qwen3-32B_gym_stats.csv',
        'label': 'Qwen3-32B',
        'short_label': 'Qwen3-32B (Text)',
        'color': '#A47AFF',
        'marker': 'D',
    },
    'vl_text': {
        'file': 'Qwen_Qwen3-VL-32B-Thinking_gym_stats.csv',
        'label': 'Qwen3-VL-32B\n(Text)',
        'short_label': 'Qwen3-VL-32B (Text)',
        'color': '#C78EF0',
        'marker': 's',
    },
    'vision': {
        'file': 'Qwen_Qwen3-VL-32B-Thinking_gym_visual_stats.csv',
        'label': 'Qwen3-VL-32B\n(Vision)',
        'short_label': 'Qwen3-VL-32B (Vision)',
        'color': '#9B59B6',
        'marker': 'o',
    },
}


def extract_stats_from_csv(stats_file):
    """Extract key statistics from a stats CSV file."""
    df = pd.read_csv(stats_file)
    stats = {}

    solved_row = df[df['Metric'] == 'Correctly Solved']
    if len(solved_row) > 0:
        match = re.search(r'([\d.]+)%', str(solved_row['Percentage'].values[0]))
        if match:
            stats['accuracy'] = float(match.group(1))

    for d in range(1, 6):
        diff_row = df[df['Metric'] == f'Difficulty {d} Solved']
        if len(diff_row) > 0:
            match = re.match(r'(\d+)/(\d+)', str(diff_row['Value'].values[0]))
            if match:
                solved = int(match.group(1))
                total = int(match.group(2))
                stats[f'd{d}_pct'] = 100.0 * solved / total if total > 0 else 0

    reached_row = df[df['Metric'] == 'Reached End']
    if len(reached_row) > 0:
        match = re.search(r'([\d.]+)%', str(reached_row['Percentage'].values[0]))
        if match:
            stats['reached_end_pct'] = float(match.group(1))

    stuck_row = df[df['Metric'] == 'No Legal Actions']
    if len(stuck_row) > 0:
        match = re.search(r'([\d.]+)%', str(stuck_row['Percentage'].values[0]))
        if match:
            stats['stuck_pct'] = float(match.group(1))

    avg_steps_row = df[df['Metric'] == 'Avg Steps Taken']
    if len(avg_steps_row) > 0:
        match = re.search(r'([\d.]+)', str(avg_steps_row['Value'].values[0]))
        if match:
            stats['avg_steps'] = float(match.group(1))

    return stats


def create_vision_comparison(results_dir, output_path=None):
    """Create a comparison chart between vision and text-only models."""
    setup_plot_style(use_latex=True)

    results_path = Path(results_dir)

    model_stats = {}
    for key, cfg in MODELS.items():
        model_stats[key] = extract_stats_from_csv(results_path / cfg['file'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 2.5))

    # --- Subplot 1: Overall accuracy ---
    keys = list(MODELS.keys())
    labels = [MODELS[k]['label'] for k in keys]
    accuracies = [model_stats[k]['accuracy'] for k in keys]
    colors = [MODELS[k]['color'] for k in keys]

    x_pos = np.arange(len(labels))
    bars = ax1.bar(x_pos, accuracies, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8)
    for bar, acc, color in zip(bars, accuracies, colors):
        ax1.annotate(f'{acc:.1f}\\%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=color)

    ax1.set_ylabel('Accuracy (\\%)')
    ax1.set_ylim(0, max(accuracies) * 1.45)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_title('(a) Overall Accuracy', fontsize=10, fontweight='bold')

    # --- Subplot 2: Accuracy by difficulty (line plot) ---
    difficulties = np.array([1, 2, 3, 4, 5])
    for key in keys:
        vals = [model_stats[key].get(f'd{d}_pct', 0) for d in difficulties]
        cfg = MODELS[key]
        ax2.plot(difficulties, vals,
                 color=cfg['color'], marker=cfg['marker'], markersize=5,
                 linewidth=1.5, label=cfg['short_label'],
                 markeredgecolor='white', markeredgewidth=0.4)

    ax2.set_ylabel('Accuracy (\\%)')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_xticks(difficulties)
    ax2.set_xlim(0.7, 5.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax2.set_title('(b) Accuracy by Difficulty', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")

    plt.close(fig)
    return model_stats


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "vision_comparison.pdf"
    output_png = Path(__file__).parent / "vision_comparison.png"

    print("=" * 60)
    print("Vision vs Text-Only Model Comparison")
    print("=" * 60)

    stats = create_vision_comparison(results_dir, output_pdf)
    create_vision_comparison(results_dir, output_png)

    for key, cfg in MODELS.items():
        s = stats[key]
        print(f"\n{cfg['short_label']}:")
        print(f"  Overall accuracy: {s['accuracy']:.1f}%")
        print(f"  Avg steps: {s.get('avg_steps', 'N/A')}")
        print(f"  Reached end: {s.get('reached_end_pct', 'N/A'):.1f}%")
        print(f"  Deadlocked: {s.get('stuck_pct', 'N/A'):.1f}%")


if __name__ == "__main__":
    main()
