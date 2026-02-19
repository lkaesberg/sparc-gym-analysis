"""
Script to compare Qwen 3 model performance with and without reasoning (thinking).
Compares across SPARC and SPARC-Gym for both 14B and 32B sizes.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
    get_model_color,
)


def create_reasoning_comparison():
    """Create a grouped bar chart comparing reasoning vs no-reasoning."""
    
    setup_plot_style()
    
    # Data structure: {model_size: {variant: {reasoning: solve_rate}}}
    # SPARC (direct path output)
    sparc_data = {
        "14B": {"Reasoning": 12.6, "No Reasoning": None},  # No 14B no-reason SPARC data
        "32B": {"Reasoning": 5.2, "No Reasoning": 1.4},
    }
    
    # SPARC-Gym (interactive)
    gym_data = {
        "14B": {"Reasoning": 10.2, "No Reasoning": 3.4},
        "32B": {"Reasoning": 10.6, "No Reasoning": 2.2},
    }
    
    # Navigation outcomes for SPARC-Gym
    gym_nav = {
        "14B": {"Reasoning": {"finished": 68.0, "stuck": 32.0}, 
                "No Reasoning": {"finished": 44.6, "stuck": 55.4}},
        "32B": {"Reasoning": {"finished": 60.2, "stuck": 39.8}, 
                "No Reasoning": {"finished": 38.6, "stuck": 61.4}},
    }
    
    # Colors
    color_reasoning = get_model_color("Qwen 3 32B")     # Dark purple
    color_no_reasoning = "#C8B8E8"                        # Light purple
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TEXT_WIDTH_INCHES, 2.4))
    
    sizes = ["14B", "32B"]
    x = np.arange(len(sizes))
    width = 0.35
    
    # === Left: SPARC Solve Rate ===
    sparc_reason = [sparc_data[s]["Reasoning"] for s in sizes]
    sparc_no_reason = [sparc_data[s]["No Reasoning"] if sparc_data[s]["No Reasoning"] is not None else 0 for s in sizes]
    
    bars1a = ax1.bar(x - width/2, sparc_reason, width, color=color_reasoning, 
                     edgecolor='black', linewidth=0.5, label='Reasoning')
    bars1b = ax1.bar(x + width/2, sparc_no_reason, width, color=color_no_reasoning, 
                     edgecolor='black', linewidth=0.5, label='No Reasoning')
    
    # Value labels
    for bar, val in zip(bars1a, sparc_reason):
        if val is not None:
            ax1.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar, val, orig in zip(bars1b, sparc_no_reason, [sparc_data[s]["No Reasoning"] for s in sizes]):
        if orig is not None:
            ax1.annotate(f'{val:.1f}\\%', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
        else:
            ax1.annotate('N/A', xy=(bar.get_x() + bar.get_width()/2, 0.5),
                        ha='center', va='bottom', fontsize=6, color='gray')
    
    ax1.set_title('SPARC', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Solve Rate (\\%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Qwen 3\n{s}' for s in sizes], fontsize=8)
    ax1.set_ylim(0, max(max(sparc_reason), max(sparc_no_reason)) * 1.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=6, loc='upper right', ncol=1)
    
    # === Middle: SPARC-Gym Solve Rate ===
    gym_reason = [gym_data[s]["Reasoning"] for s in sizes]
    gym_no_reason = [gym_data[s]["No Reasoning"] for s in sizes]
    
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
    ax2.set_xticklabels([f'Qwen 3\n{s}' for s in sizes], fontsize=8)
    ax2.set_ylim(0, max(max(gym_reason), max(gym_no_reason)) * 1.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=6, loc='upper right', ncol=1)
    
    # === Right: SPARC-Gym Navigation Outcome ===
    # Show all 4 bars: 14B reason, 14B no-reason, 32B reason, 32B no-reason
    nav_labels = ["14B\nReason", "14B\nNo Reason", "32B\nReason", "32B\nNo Reason"]
    finished = [gym_nav["14B"]["Reasoning"]["finished"], gym_nav["14B"]["No Reasoning"]["finished"],
                gym_nav["32B"]["Reasoning"]["finished"], gym_nav["32B"]["No Reasoning"]["finished"]]
    stuck = [gym_nav["14B"]["Reasoning"]["stuck"], gym_nav["14B"]["No Reasoning"]["stuck"],
             gym_nav["32B"]["Reasoning"]["stuck"], gym_nav["32B"]["No Reasoning"]["stuck"]]
    nav_colors = [color_reasoning, color_no_reasoning, color_reasoning, color_no_reasoning]
    
    x3 = np.arange(len(nav_labels))
    bars3a = ax3.bar(x3, finished, 0.6, color=nav_colors, edgecolor='black', linewidth=0.5)
    bars3b = ax3.bar(x3, stuck, 0.6, bottom=finished, color='#E8E8E8', 
                     edgecolor='black', linewidth=0.5, hatch='////')
    
    for bar, val in zip(bars3a, finished):
        if val > 15:
            ax3.annotate(f'{val:.0f}\\%', xy=(bar.get_x() + bar.get_width()/2, val/2),
                        ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    
    # Separator between 14B and 32B, limited to y=100
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
    
    # Add legend for navigation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Finished'),
        Patch(facecolor='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////', label='Deadlocked')
    ]
    ax3.legend(handles=legend_elements, loc='upper center', fontsize=6, ncol=2)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "reasoning_comparison.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / "reasoning_comparison.png", bbox_inches='tight', dpi=300)
    print("Saved reasoning_comparison.pdf and reasoning_comparison.png")
    plt.close()


if __name__ == "__main__":
    create_reasoning_comparison()
