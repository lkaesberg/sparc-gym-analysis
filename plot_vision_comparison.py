"""
Script to create a comparison between Qwen3-VL-32B (vision) and Qwen3-32B (text-only).
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
)

# Colors for the two models
COLORS = {
    'vision': '#9B59B6',    # Purple for vision
    'text': '#A47AFF',      # Lighter purple for text-only Qwen
}


def extract_stats_from_csv(stats_file):
    """Extract key statistics from a stats CSV file."""
    df = pd.read_csv(stats_file)
    
    stats = {}
    
    # Overall accuracy
    solved_row = df[df['Metric'] == 'Correctly Solved']
    if len(solved_row) > 0:
        percentage_str = solved_row['Percentage'].values[0]
        match = re.search(r'([\d.]+)%', str(percentage_str))
        if match:
            stats['accuracy'] = float(match.group(1))
    
    # Difficulty breakdown
    for d in range(1, 6):
        diff_row = df[df['Metric'] == f'Difficulty {d} Solved']
        if len(diff_row) > 0:
            value_str = diff_row['Value'].values[0]
            # Parse "X/Y" format
            match = re.match(r'(\d+)/(\d+)', str(value_str))
            if match:
                solved = int(match.group(1))
                total = int(match.group(2))
                stats[f'd{d}_solved'] = solved
                stats[f'd{d}_total'] = total
                stats[f'd{d}_pct'] = 100.0 * solved / total if total > 0 else 0
    
    # Steps info
    avg_steps_row = df[df['Metric'] == 'Avg Steps Taken']
    if len(avg_steps_row) > 0:
        value_str = avg_steps_row['Value'].values[0]
        match = re.search(r'([\d.]+)', str(value_str))
        if match:
            stats['avg_steps'] = float(match.group(1))
    
    # Reached end vs no legal actions
    reached_row = df[df['Metric'] == 'Reached End']
    if len(reached_row) > 0:
        percentage_str = reached_row['Percentage'].values[0]
        match = re.search(r'([\d.]+)%', str(percentage_str))
        if match:
            stats['reached_end_pct'] = float(match.group(1))
    
    return stats


def create_vision_comparison(results_dir, output_path=None):
    """Create a comparison chart between vision and text-only models."""
    setup_plot_style(use_latex=True)
    
    results_path = Path(results_dir)
    
    # Load stats for both models
    vision_stats = extract_stats_from_csv(results_path / "Qwen_Qwen3-VL-32B-Thinking_gym_visual_stats.csv")
    text_stats = extract_stats_from_csv(results_path / "Qwen_Qwen3-32B_gym_stats.csv")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 2.5))
    
    # --- Subplot 1: Overall accuracy comparison ---
    ax1 = axes[0]
    models = ['Qwen3-32B\n(Text)', 'Qwen3-VL-32B\n(Vision)']
    accuracies = [text_stats['accuracy'], vision_stats['accuracy']]
    colors = [COLORS['text'], COLORS['vision']]
    
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.annotate(f'{acc:.1f}\\%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (\\%)')
    ax1.set_ylim(0, max(accuracies) * 1.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # --- Subplot 2: Accuracy by difficulty ---
    ax2 = axes[1]
    difficulties = ['1', '2', '3', '4', '5']
    vision_by_diff = [vision_stats.get(f'd{i}_pct', 0) for i in range(1, 6)]
    text_by_diff = [text_stats.get(f'd{i}_pct', 0) for i in range(1, 6)]
    
    x = np.arange(len(difficulties))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, text_by_diff, width, label='Text', color=COLORS['text'], edgecolor='white')
    bars2 = ax2.bar(x + width/2, vision_by_diff, width, label='Vision', color=COLORS['vision'], edgecolor='white')
    
    ax2.set_ylabel('Accuracy (\\%)')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulties)
    ax2.legend(loc='upper right', framealpha=0.9)
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
    
    return vision_stats, text_stats


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "vision_comparison.pdf"
    output_png = Path(__file__).parent / "vision_comparison.png"
    
    print("=" * 60)
    print("Vision vs Text-Only Model Comparison")
    print("=" * 60)
    
    vision_stats, text_stats = create_vision_comparison(results_dir, output_pdf)
    create_vision_comparison(results_dir, output_png)
    
    # Print summary
    print(f"\nQwen3-VL-32B (Vision):")
    print(f"  Overall accuracy: {vision_stats['accuracy']:.1f}%")
    print(f"  Avg steps: {vision_stats.get('avg_steps', 'N/A')}")
    print(f"  Reached end: {vision_stats.get('reached_end_pct', 'N/A'):.1f}%")
    
    print(f"\nQwen3-32B (Text):")
    print(f"  Overall accuracy: {text_stats['accuracy']:.1f}%")
    print(f"  Avg steps: {text_stats.get('avg_steps', 'N/A')}")
    print(f"  Reached end: {text_stats.get('reached_end_pct', 'N/A'):.1f}%")
    
    print(f"\nDifference: {text_stats['accuracy'] - vision_stats['accuracy']:+.1f}% (text - vision)")
    
    print("\nBy difficulty:")
    for d in range(1, 6):
        v_pct = vision_stats.get(f'd{d}_pct', 0)
        t_pct = text_stats.get(f'd{d}_pct', 0)
        print(f"  D{d}: Vision {v_pct:.1f}% vs Text {t_pct:.1f}% (diff: {t_pct - v_pct:+.1f}%)")


if __name__ == "__main__":
    main()
