"""
Script to compare the largest model (GPT-OSS 120B) and smallest model (Qwen 0.6B)
against two baselines: Random Agent and A* Search.
Shows both solve rate and finish rate (reached end vs got stuck).
Reads all data from stats CSV files.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path
from matplotlib.patches import Patch

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    get_model_color,
    MODEL_COLORS,
)

RESULTS_DIR = Path(__file__).parent / "results" / "sparc"

MODELS = [
    {"label": "Random", "file": "random_ablation_baseline1_stats.csv",
     "color": MODEL_COLORS.get("Random Agent", "#A0A0A0")},
    {"label": "A*", "file": "astar_ablation_baseline2_stats.csv",
     "color": MODEL_COLORS.get("A*", "#505050")},
    {"label": "Qwen 3\n0.6B", "file": "Qwen_Qwen3-0.6B_gym_stats.csv",
     "color": get_model_color("Qwen 3 0.6B")},
    {"label": "GPT-OSS\n120B", "file": "openai_gpt-oss-120b_gym_stats.csv",
     "color": "#F79F1F"},
]


def extract_stats(stats_file):
    """Extract solve rate, finish rate, and stuck rate from a stats CSV file."""
    df = pd.read_csv(stats_file)
    stats = {}

    solved_row = df[df['Metric'] == 'Correctly Solved']
    if len(solved_row) > 0:
        match = re.search(r'([\d.]+)%', str(solved_row['Percentage'].values[0]))
        if match:
            stats['solve_rate'] = float(match.group(1))

    reached_row = df[df['Metric'] == 'Reached End']
    if len(reached_row) > 0:
        match = re.search(r'([\d.]+)%', str(reached_row['Percentage'].values[0]))
        if match:
            stats['finish_rate'] = float(match.group(1))

    stuck_row = df[df['Metric'] == 'No Legal Actions']
    if len(stuck_row) > 0:
        match = re.search(r'([\d.]+)%', str(stuck_row['Percentage'].values[0]))
        if match:
            stats['stuck_rate'] = float(match.group(1))

    return stats


def create_baseline_comparison():
    """Create a grouped bar chart comparing models against baselines."""
    setup_plot_style()

    all_stats = []
    for m in MODELS:
        filepath = RESULTS_DIR / m["file"]
        if filepath.exists():
            stats = extract_stats(filepath)
            print(f"  {m['label'].replace(chr(10), ' ')}: {stats}")
        else:
            print(f"  Warning: {filepath} not found")
            stats = {}
        all_stats.append(stats)

    labels = [m["label"] for m in MODELS]
    bar_colors = [m["color"] for m in MODELS]
    solve_rates = [s.get('solve_rate', 0) for s in all_stats]
    finish_rates = [s.get('finish_rate', 0) for s in all_stats]
    stuck_rates = [s.get('stuck_rate', 0) for s in all_stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES * 0.85, 2.4))

    x = np.arange(len(labels))
    width = 0.6

    # === Left plot: Accuracy ===
    bars1 = ax1.bar(x, solve_rates, width, color=bar_colors, edgecolor='black', linewidth=0.5)

    for bar, value in zip(bars1, solve_rates):
        height = bar.get_height()
        ax1.annotate(f'{value:.1f}\\%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax1.axvline(x=1.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_ylabel('Rate (\\%)')
    ax1.set_title('Accuracy', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, max(solve_rates) * 1.35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)

    # === Right plot: Navigation Outcome (stacked) ===
    bars_finish = ax2.bar(x, finish_rates, width, color=bar_colors, edgecolor='black',
                          linewidth=0.5)
    ax2.bar(x, stuck_rates, width, bottom=finish_rates, color='#E8E8E8',
            edgecolor='black', linewidth=0.5, hatch='////')

    for bar, value in zip(bars_finish, finish_rates):
        if value > 15:
            ax2.annotate(f'{value:.0f}\\%',
                        xy=(bar.get_x() + bar.get_width() / 2, value / 2),
                        ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white')

    ax2.axvline(x=1.5, ymin=0, ymax=100/125, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.set_ylabel('Rate (\\%)')
    ax2.set_title('Navigation Outcome', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, 125)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Finished'),
        Patch(facecolor='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////', label='Deadlocked')
    ]
    ax2.legend(handles=legend_elements, loc='upper center', fontsize=7,
               framealpha=0.9, ncol=2)

    y_top = ax1.get_ylim()[1]
    ax1.text(0.5, y_top * 0.97, 'Baselines', ha='center', va='top',
            fontsize=7, fontstyle='italic', color='#555555')
    ax1.text(2.5, y_top * 0.97, 'LLMs', ha='center', va='top',
            fontsize=7, fontstyle='italic', color='#555555')

    plt.tight_layout()

    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "baseline_comparison.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / "baseline_comparison.png", bbox_inches='tight', dpi=300)
    print("Saved baseline_comparison.pdf and baseline_comparison.png")

    plt.close()


if __name__ == "__main__":
    create_baseline_comparison()
