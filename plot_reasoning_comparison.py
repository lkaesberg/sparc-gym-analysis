"""
Script to compare Qwen 3 model performance with and without reasoning (thinking).
Compares across SPaRC and Spatial Gym for both 14B and 32B sizes.
Reads all data from stats CSV files.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnnotationBbox

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    get_model_color,
    get_model_imagebox,
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TEXT_WIDTH_INCHES, 1.6),
                                        gridspec_kw={'width_ratios': [1, 1, 1.2]})

    x = np.arange(len(MODEL_SIZES))
    width = 0.35

    # === Left: SPaRC Accuracy ===
    sparc_reason = [all_stats[s]['sparc_reason'].get('solve_rate', 0) for s in MODEL_SIZES]
    sparc_no_reason = [all_stats[s]['sparc_no_reason'].get('solve_rate', 0) for s in MODEL_SIZES]

    bars1a = ax1.bar(x - width/2, sparc_reason, width, color=color_reasoning,
                     edgecolor='black', linewidth=0.5, label='Reasoning')
    bars1b = ax1.bar(x + width/2, sparc_no_reason, width, color=color_no_reasoning,
                     edgecolor='black', linewidth=0.5, label='No Reasoning')

    for bar, val in zip(bars1a, sparc_reason):
        ax1.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, fontweight='bold')
    for bar, val in zip(bars1b, sparc_no_reason):
        ax1.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax1.set_title('Baseline', fontweight='bold')
    ax1.set_ylabel('Accuracy (\\%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Qwen 3\n{s}' for s in MODEL_SIZES], fontsize=7)

    ax1.set_ylim(0, max(max(sparc_reason), max(sparc_no_reason)) * 1.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='y', labelsize=7)

    # === Middle: Spatial Gym Accuracy ===
    gym_reason = [all_stats[s]['gym_reason'].get('solve_rate', 0) for s in MODEL_SIZES]
    gym_no_reason = [all_stats[s]['gym_no_reason'].get('solve_rate', 0) for s in MODEL_SIZES]

    bars2a = ax2.bar(x - width/2, gym_reason, width, color=color_reasoning,
                     edgecolor='black', linewidth=0.5, label='Reasoning')
    bars2b = ax2.bar(x + width/2, gym_no_reason, width, color=color_no_reasoning,
                     edgecolor='black', linewidth=0.5, label='No Reasoning')

    for bar, val in zip(bars2a, gym_reason):
        ax2.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, fontweight='bold')
    for bar, val in zip(bars2b, gym_no_reason):
        ax2.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax2.set_title('Gym', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Qwen 3\n{s}' for s in MODEL_SIZES], fontsize=7)
    ax2.tick_params(labelleft=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    # === Right: Spatial Gym Navigation Outcome (grouped like ax1/ax2) ===
    finished_reason = [all_stats[s]['gym_reason'].get('finished', 0) for s in MODEL_SIZES]
    finished_no = [all_stats[s]['gym_no_reason'].get('finished', 0) for s in MODEL_SIZES]
    stuck_reason = [all_stats[s]['gym_reason'].get('stuck', 0) for s in MODEL_SIZES]
    stuck_no = [all_stats[s]['gym_no_reason'].get('stuck', 0) for s in MODEL_SIZES]

    x3 = np.arange(len(MODEL_SIZES))
    bars3a = ax3.bar(x3 - width/2, finished_reason, width, color=color_reasoning,
                     edgecolor='black', linewidth=0.5)
    ax3.bar(x3 - width/2, stuck_reason, width, bottom=finished_reason,
            color='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////')
    bars3b = ax3.bar(x3 + width/2, finished_no, width, color=color_no_reasoning,
                     edgecolor='black', linewidth=0.5)
    ax3.bar(x3 + width/2, stuck_no, width, bottom=finished_no,
            color='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////')

    for bar, val in zip(bars3a, finished_reason):
        if val > 15:
            ax3.annotate(f'{val:.0f}\\%', xy=(bar.get_x() + bar.get_width()/2, val/2),
                        ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    for bar, val in zip(bars3b, finished_no):
        if val > 15:
            ax3.annotate(f'{val:.0f}\\%', xy=(bar.get_x() + bar.get_width()/2, val/2),
                        ha='center', va='center', fontsize=6, fontweight='bold', color='white')

    ax3.set_title('Navigation Outcome', fontweight='bold')
    ax3.set_ylabel('Rate (\\%)')
    ax3.set_xticks(x3)
    ax3.set_xticklabels([f'Qwen 3\n{s}' for s in MODEL_SIZES], fontsize=7)
    ax3.set_ylim(0, 135)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.tick_params(axis='y', labelsize=7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax3.set_axisbelow(True)

    # Navigation Outcome finished/deadlocked legend inside ax3
    legend_nav = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Finished'),
        Patch(facecolor='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////', label='Deadlocked'),
    ]
    ax3.legend(handles=legend_nav, loc='upper center', fontsize=6, ncol=2, framealpha=0.9, columnspacing=1, handlelength=1.4, handleheight=0.8)

    # Unified Reasoning / No Reasoning legend below all panels
    legend_reason = [
        Patch(facecolor=color_reasoning, edgecolor='black', linewidth=0.5, label='Reasoning'),
        Patch(facecolor=color_no_reasoning, edgecolor='black', linewidth=0.5, label='No Reasoning'),
    ]
    fig.legend(handles=legend_reason, loc='lower center', fontsize=6, ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # Tighten gap between first two panels; pull ax3 along with ax2
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    gap = 0.01
    shift = pos2.x0 - (pos1.x1 + gap)
    ax2.set_position([pos1.x1 + gap, pos2.y0, pos2.width, pos2.height])
    ax3.set_position([pos3.x0 - shift, pos3.y0, pos3.width, pos3.height])

    output_dir = Path(__file__).parent

    # First pass: stabilise layout
    fig.savefig(output_dir / "reasoning_comparison.pdf", bbox_inches='tight', dpi=300)

    # Place logos above every individual bar

    def place_logo(ax, x_data, y_top, logo_key, zoom=0.75, offset_pts=14):
        imagebox = get_model_imagebox(logo_key, zoom_factor=zoom)
        if not imagebox:
            return
        ab = AnnotationBbox(imagebox, (x_data, y_top),
                            xybox=(0, offset_pts),
                            xycoords='data',
                            boxcoords='offset points',
                            frameon=False,
                            pad=0,
                            box_alignment=(0.5, 0.0),
                            zorder=10)
        ax.add_artist(ab)

    for i, size in enumerate(MODEL_SIZES):
        logo_r  = f'Qwen 3 {size}'
        logo_nr = f'Qwen No Reason {size}'  # maps to qwen-no-reason.png

        # ax1 (SPaRC accuracy)
        place_logo(ax1, x[i] - width/2, sparc_reason[i]-0.5,    logo_r)
        place_logo(ax1, x[i] + width/2, sparc_no_reason[i]-0.5, logo_nr)

        # ax2 (Gym accuracy)
        place_logo(ax2, x[i] - width/2, gym_reason[i]-0.5,    logo_r)
        place_logo(ax2, x[i] + width/2, gym_no_reason[i]-0.5, logo_nr)

    # Set shared ylim to accommodate logos above tallest bars
    all_acc = sparc_reason + sparc_no_reason + gym_reason + gym_no_reason
    shared_ylim = (0, max(all_acc) * 1.45)
    ax1.set_ylim(shared_ylim)
    ax2.set_ylim(shared_ylim)
    ax3.set_ylim(0, 140)

    # Second pass: save with logos
    fig.savefig(output_dir / "reasoning_comparison.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / "reasoning_comparison.png", bbox_inches='tight', dpi=300)
    print("Saved reasoning_comparison.pdf and reasoning_comparison.png")
    plt.close()


if __name__ == "__main__":
    create_reasoning_comparison()
