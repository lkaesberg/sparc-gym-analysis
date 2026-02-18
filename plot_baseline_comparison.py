"""
Script to compare the largest model (GPT-OSS 120B) and smallest model (Qwen 0.6B)
against two baselines: Random Agent and A* Search.
Shows both solve rate and finish rate (reached end vs got stuck).
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plot_config import (
    setup_plot_style,
    COLUMN_WIDTH_INCHES,
    TEXT_WIDTH_INCHES,
    get_model_color,
    MODEL_COLORS,
)


def create_baseline_comparison():
    """Create a grouped bar chart comparing models against baselines."""
    
    # Data from stats files (SPARC-Gym results)
    # Labels with line breaks to avoid overlap
    labels = ["Random", "A*", "Qwen 3\n0.6B", "GPT-OSS\n120B"]
    
    # Solve rates (correctly solved)
    solve_rates = [2.4, 6.4, 2.7, 16.0]
    
    # Finish rates (reached end node, regardless of rule satisfaction)
    finish_rates = [31.2, 100.0, 42.7, 85.4]
    
    # Got stuck rates (no legal actions available)
    stuck_rates = [68.8, 0.0, 57.3, 14.6]
    
    # Define colors
    colors = {
        "Random": MODEL_COLORS.get("Random Agent", "#A0A0A0"),
        "A*": MODEL_COLORS.get("A*", "#505050"),
        "Qwen 3\n0.6B": get_model_color("Qwen 3 0.6B"),
        "GPT-OSS\n120B": "#F79F1F",  # Orange for OpenAI
    }
    bar_colors = [colors[label] for label in labels]
    
    # Create figure with two subplots
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES * 0.85, 2.4))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # === Left plot: Solve Rate ===
    bars1 = ax1.bar(x, solve_rates, width, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
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
    ax1.set_title('Solve Rate', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, max(solve_rates) * 1.35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # === Right plot: Finish Rate (stacked: reached end vs stuck) ===
    # Bottom bar: reached end (success in navigation) - colored by model, no pattern
    bars_finish = ax2.bar(x, finish_rates, width, color=bar_colors, edgecolor='black', 
                          linewidth=0.5)
    # Top bar: got stuck - with very subtle pattern
    bars_stuck = ax2.bar(x, stuck_rates, width, bottom=finish_rates, color='#E8E8E8', 
                         edgecolor='black', linewidth=0.5, hatch='////')
    
    # Add value labels for finish rates
    for bar, value in zip(bars_finish, finish_rates):
        if value > 15:  # Only label if bar is large enough
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
    
    # Create custom legend handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Finished'),
        Patch(facecolor='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////', label='Deadlocked')
    ]
    # Position legend at the top inside the extended ylim area
    ax2.legend(handles=legend_elements, loc='upper center', fontsize=7, 
               framealpha=0.9, ncol=2)
    
    # Add category labels - only on left plot to avoid overlap with legend
    y_top = ax1.get_ylim()[1]
    ax1.text(0.5, y_top * 0.97, 'Baselines', ha='center', va='top', 
            fontsize=7, fontstyle='italic', color='#555555')
    ax1.text(2.5, y_top * 0.97, 'LLMs', ha='center', va='top', 
            fontsize=7, fontstyle='italic', color='#555555')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figures
    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "baseline_comparison.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / "baseline_comparison.png", bbox_inches='tight', dpi=300)
    print(f"Saved baseline_comparison.pdf and baseline_comparison.png")
    
    plt.close()


if __name__ == "__main__":
    create_baseline_comparison()
