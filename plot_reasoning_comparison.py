"""
Script to compare Qwen 3 model performance with and without reasoning (thinking).
Compares across SPARC and SPARC-Gym for both 14B and 32B sizes.
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
)

RESULTS_DIR = Path(__file__).parent / "results" / "sparc"

MODEL_SIZES = ["14B", "32B"]

STATS_FILES = {
    "14B": {
        "sparc_reason": "Qwen_Qwen3-14B_stats.csv",
        "sparc_no_reason": "Qwen_Qwen3-14B_no-reason_stats.csv",
        "gym_reason": "Qwen_Qwen3-14B_gym_stats.csv",
        "gym_no_reason": "Qwen_Qwen3-14B_gym_no-reason_stats.csv",
    },
    "32B": {
        "sparc_reason": "Qwen_Qwen3-32B_stats.csv",
        "sparc_no_reason": "Qwen_Qwen3-32B_no-reason_stats.csv",
        "gym_reason": "Qwen_Qwen3-32B_gym_stats.csv",
        "gym_no_reason": "Qwen_Qwen3-32B_gym_no-reason_stats.csv",
    },
}


def extract_stats(stats_file):
    """Extract solve rate and navigation stats from a stats CSV file."""
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
            stats['finished'] = float(match.group(1))

    stuck_row = df[df['Metric'] == 'No Legal Actions']
    if len(stuck_row) > 0:
        match = re.search(r'([\d.]+)%', str(stuck_row['Percentage'].values[0]))
        if match:
            stats['stuck'] = float(match.group(1))

    return stats


def load_all_stats():
    """Load all stats from CSV files."""
    all_stats = {}
    for size, files in STATS_FILES.items():
        all_stats[size] = {}
        for key, filename in files.items():
            filepath = RESULTS_DIR / filename
            if filepath.exists():
                all_stats[size][key] = extract_stats(filepath)
            else:
                print(f"Warning: {filepath} not found")
                all_stats[size][key] = {}
    return all_stats


def create_reasoning_comparison():
    """Create a grouped bar chart comparing reasoning vs no-reasoning."""
    setup_plot_style()

    all_stats = load_all_stats()

    # Print loaded data
    for size in MODEL_SIZES:
        for key, stats in all_stats[size].items():
            print(f"  {size} {key}: {stats}")

    color_reasoning = get_model_color("Qwen 3 32B")
    color_no_reasoning = "#C8B8E8"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TEXT_WIDTH_INCHES, 2.4))

    x = np.arange(len(MODEL_SIZES))
    width = 0.35

    # === Left: SPARC Solve Rate ===
    sparc_reason = [all_stats[s]['sparc_reason'].get('solve_rate', 0) for s in MODEL_SIZES]
    sparc_no_reason = [all_stats[s]['sparc_no_reason'].get('solve_rate', 0) for s in MODEL_SIZES]

    bars1a = ax1.bar(x - width/2, sparc_reason, width, color=color_reasoning,
                     edgecolor='black', linewidth=0.5, label='Reasoning')
    bars1b = ax1.bar(x + width/2, sparc_no_reason, width, color=color_no_reasoning,
                     edgecolor='black', linewidth=0.5, label='No Reasoning')

    for bar, val in zip(bars1a, sparc_reason):
        ax1.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar, val in zip(bars1b, sparc_no_reason):
        ax1.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.set_title('SPARC', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Solve Rate (\\%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Qwen 3\n{s}' for s in MODEL_SIZES], fontsize=8)
    ax1.set_ylim(0, max(max(sparc_reason), max(sparc_no_reason)) * 1.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=6, loc='upper right', ncol=1)

    # === Middle: SPARC-Gym Solve Rate ===
    gym_reason = [all_stats[s]['gym_reason'].get('solve_rate', 0) for s in MODEL_SIZES]
    gym_no_reason = [all_stats[s]['gym_no_reason'].get('solve_rate', 0) for s in MODEL_SIZES]

    bars2a = ax2.bar(x - width/2, gym_reason, width, color=color_reasoning,
                     edgecolor='black', linewidth=0.5, label='Reasoning')
    bars2b = ax2.bar(x + width/2, gym_no_reason, width, color=color_no_reasoning,
                     edgecolor='black', linewidth=0.5, label='No Reasoning')

    for bar, val in zip(bars2a, gym_reason):
        ax2.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar, val in zip(bars2b, gym_no_reason):
        ax2.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax2.set_title('SPARC-Gym', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Solve Rate (\\%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Qwen 3\n{s}' for s in MODEL_SIZES], fontsize=8)
    ax2.set_ylim(0, max(max(gym_reason), max(gym_no_reason)) * 1.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=6, loc='upper right', ncol=1)

    # === Right: SPARC-Gym Navigation Outcome ===
    nav_labels = ["14B\nReason", "14B\nNo Reason", "32B\nReason", "32B\nNo Reason"]
    finished = [
        all_stats["14B"]["gym_reason"].get("finished", 0),
        all_stats["14B"]["gym_no_reason"].get("finished", 0),
        all_stats["32B"]["gym_reason"].get("finished", 0),
        all_stats["32B"]["gym_no_reason"].get("finished", 0),
    ]
    stuck = [
        all_stats["14B"]["gym_reason"].get("stuck", 0),
        all_stats["14B"]["gym_no_reason"].get("stuck", 0),
        all_stats["32B"]["gym_reason"].get("stuck", 0),
        all_stats["32B"]["gym_no_reason"].get("stuck", 0),
    ]
    nav_colors = [color_reasoning, color_no_reasoning, color_reasoning, color_no_reasoning]

    x3 = np.arange(len(nav_labels))
    bars3a = ax3.bar(x3, finished, 0.6, color=nav_colors, edgecolor='black', linewidth=0.5)
    ax3.bar(x3, stuck, 0.6, bottom=finished, color='#E8E8E8',
            edgecolor='black', linewidth=0.5, hatch='////')

    for bar, val in zip(bars3a, finished):
        if val > 15:
            ax3.annotate(f'{val:.0f}\\%', xy=(bar.get_x() + bar.get_width()/2, val/2),
                        ha='center', va='center', fontsize=6, fontweight='bold', color='white')

    ax3.axvline(x=1.5, ymin=0, ymax=100/120, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax3.set_title('Navigation Outcome', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Rate (\\%)')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(nav_labels, fontsize=6)
    ax3.set_ylim(0, 120)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax3.set_axisbelow(True)

    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Finished'),
        Patch(facecolor='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////', label='Deadlocked')
    ]
    ax3.legend(handles=legend_elements, loc='upper center', fontsize=6, ncol=2)

    plt.tight_layout()

    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "reasoning_comparison.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / "reasoning_comparison.png", bbox_inches='tight', dpi=300)
    print("Saved reasoning_comparison.pdf and reasoning_comparison.png")
    plt.close()


if __name__ == "__main__":
    create_reasoning_comparison()
