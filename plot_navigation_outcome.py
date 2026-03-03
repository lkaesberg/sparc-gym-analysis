"""
Script to compare navigation outcomes (finished vs deadlocked) for all models.
Left panel: SPaRC-Gym, Right panel: SPaRC-Gym Traceback
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re

from matplotlib.offsetbox import AnnotationBbox

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    get_model_color,
    get_model_imagebox,
    MODEL_COLORS,
)

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "allenai_Olmo-3.1-32B-Think": "OLMo 3.1 32B",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron 49B",
    "Qwen_Qwen3-32B": "Qwen 3 32B",
    "Qwen_Qwen3-0.6B": "Qwen 3 0.6B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "R1 Distill 32B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "mistralai_Magistral-Small-2507": "Magistral Small",
}

# Models to include (same as accuracy plot)
INCLUDED_MODELS = {
    "openai_gpt-oss-120b",
    "allenai_Olmo-3.1-32B-Think",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5",
    "Qwen_Qwen3-32B",
    "Qwen_Qwen3-0.6B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
    "google_gemma-3-27b-it",
    "mistralai_Magistral-Small-2507",
}

# Logo-inspired fallback colors keyed on tokens in the internal model path/name
MODEL_FAMILY_COLORS = {
    "openai":    "#10A37F",  # GPT / OpenAI → ChatGPT teal-green
    "gpt":       "#10A37F",
    "google":    "#4E84C4",  # Gemma → blue (Gemma logo)
    "gemma":     "#4E84C4",
    "Qwen":      "#6040E0",  # Qwen → purple-indigo (Qwen logo)
    "qwen":      "#6040E0",
    "deepseek":  "#4A6EA8",  # R1 / DeepSeek → cobalt blue
    "nvidia":    "#76B900",  # Nemotron → NVIDIA lime green
    "allenai":   "#D43870",  # OLMo → hot pink (OLMo logo)
    "mistralai": "#D96818",  # Magistral → warm orange (Mistral logo)
}


def get_model_family_color(model_name, display_name=None):
    """Get color based on model family/provider, checking MODEL_COLORS first."""
    if display_name and display_name in MODEL_COLORS:
        return MODEL_COLORS[display_name]
    for family, color in MODEL_FAMILY_COLORS.items():
        if family.lower() in model_name.lower():
            return color
    return "#808080"


def extract_navigation_stats(stats_file):
    """Extract navigation outcome stats from a stats CSV file."""
    df = pd.read_csv(stats_file)

    reached_pct = 0.0
    stuck_pct = 0.0
    solve_rate = 0.0

    reached_row = df[df['Metric'] == 'Reached End']
    if len(reached_row) > 0:
        match = re.search(r'([\d.]+)%', str(reached_row['Percentage'].values[0]))
        if match:
            reached_pct = float(match.group(1))

    stuck_row = df[df['Metric'] == 'No Legal Actions']
    if len(stuck_row) > 0:
        match = re.search(r'([\d.]+)%', str(stuck_row['Percentage'].values[0]))
        if match:
            stuck_pct = float(match.group(1))

    solved_row = df[df['Metric'] == 'Correctly Solved']
    if len(solved_row) > 0:
        match = re.search(r'([\d.]+)%', str(solved_row['Percentage'].values[0]))
        if match:
            solve_rate = float(match.group(1))

    return reached_pct, stuck_pct, solve_rate


def load_navigation_stats(results_dir, variant='gym'):
    """Load navigation stats for all models of a specific variant."""
    results_path = Path(results_dir)
    
    if variant == 'gym':
        pattern = "*_gym_stats.csv"
        exclude_pattern = 'traceback'
    else:  # traceback
        pattern = "*_gym_traceback_stats.csv"
        exclude_pattern = None
    
    stats_files = list(results_path.glob(pattern))
    
    # Filter
    if variant == 'gym':
        stats_files = [f for f in stats_files if 'traceback' not in f.name and 'visual' not in f.name]
    
    model_data = []
    for stats_file in stats_files:
        # Extract model name from filename
        if variant == 'gym':
            model_name = stats_file.name.replace("_gym_stats.csv", "")
        else:
            model_name = stats_file.name.replace("_gym_traceback_stats.csv", "")
        
        # Only include selected models
        if model_name not in INCLUDED_MODELS:
            continue
        
        reached_pct, stuck_pct, solve_rate = extract_navigation_stats(stats_file)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = get_model_family_color(model_name, display_name)
        
        model_data.append({
            'model_name': model_name,
            'display_name': display_name,
            'reached': reached_pct,
            'stuck': stuck_pct,
            'solve_rate': solve_rate,
            'color': color
        })
    
    return model_data


def create_navigation_comparison():
    """Create the navigation outcome comparison chart."""
    setup_plot_style()
    
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    # Load data for both variants
    gym_data = load_navigation_stats(results_dir, 'gym')
    traceback_data_unsorted = load_navigation_stats(results_dir, 'traceback')

    # Sort gym data by solve rate (descending)
    gym_data.sort(key=lambda x: x['solve_rate'], reverse=True)

    # Reorder traceback data to match gym ordering
    gym_order = [m['model_name'] for m in gym_data]
    tb_by_name = {m['model_name']: m for m in traceback_data_unsorted}
    traceback_data = [tb_by_name[name] for name in gym_order if name in tb_by_name]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 2.4), sharey=True)
    
    width = 0.75
    
    # === Left plot: SPaRC-Gym ===
    labels1 = [m['display_name'] for m in gym_data]
    reached1 = [m['reached'] for m in gym_data]
    stuck1 = [m['stuck'] for m in gym_data]
    colors1 = [m['color'] for m in gym_data]
    
    x1 = np.arange(len(labels1))
    
    # Stacked bars
    bars_finish1 = ax1.bar(x1, reached1, width, color=colors1, edgecolor='black', linewidth=0.5)
    bars_stuck1 = ax1.bar(x1, stuck1, width, bottom=reached1, color='#E8E8E8', 
                          edgecolor='black', linewidth=0.5, hatch='////')
    
    # Add value labels
    for bar, value in zip(bars_finish1, reached1):
        if value > 15:
            ax1.annotate(f'{value:.0f}\\%',
                        xy=(bar.get_x() + bar.get_width() / 2, value / 2),
                        ha='center', va='center',
                        fontsize=6, fontweight='bold', color='white')
    
    ax1.set_ylabel('Rate (\\%)')
    ax1.set_title('Gym w/o traceback', fontsize=10, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels1, fontsize=7, rotation=45, ha='right')
    ax1.set_ylim(0, 120)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Add legend to left plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Finished'),
        Patch(facecolor='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='////', label='Deadlocked')
    ]
    ax1.legend(handles=legend_elements, loc='upper center', fontsize=7, framealpha=0.9, ncol=2, bbox_to_anchor=(0.5, 1.04))
    
    # === Right plot: SPaRC-Gym Traceback ===
    labels2 = [m['display_name'] for m in traceback_data]
    reached2 = [m['reached'] for m in traceback_data]
    stuck2 = [m['stuck'] for m in traceback_data]
    colors2 = [m['color'] for m in traceback_data]
    
    x2 = np.arange(len(labels2))
    
    # Stacked bars
    bars_finish2 = ax2.bar(x2, reached2, width, color=colors2, edgecolor='black', linewidth=0.5)
    bars_stuck2 = ax2.bar(x2, stuck2, width, bottom=reached2, color='#E8E8E8', 
                          edgecolor='black', linewidth=0.5, hatch='////')
    
    # Add value labels
    for bar, value in zip(bars_finish2, reached2):
        if value > 15:
            ax2.annotate(f'{value:.0f}\\%',
                        xy=(bar.get_x() + bar.get_width() / 2, value / 2),
                        ha='center', va='center',
                        fontsize=6, fontweight='bold', color='white')
    
    ax2.set_ylabel('Rate (\\%)')
    ax2.set_title('Gym w/ traceback', fontsize=10, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=7, rotation=45, ha='right')
    ax2.set_ylim(0, 120)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.set_ylabel('')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add legend to right plot
    ax2.legend(handles=legend_elements, loc='upper center', fontsize=7, framealpha=0.9, ncol=2, bbox_to_anchor=(0.5, 1.04))
    
    # Tight layout
    plt.tight_layout()

    # First pass: save to finalise layout so text positions are stable
    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "navigation_outcome.pdf", bbox_inches='tight', dpi=300)

    renderer = fig.canvas.get_renderer()

    # X-axis logos on both axes
    for ax in [ax1, ax2]:
        for tick_label in ax.get_xticklabels():
            name = tick_label.get_text()
            imagebox = get_model_imagebox(name, zoom_factor=0.65, rotation=45)
            if not imagebox:
                continue
            bbox = tick_label.get_window_extent(renderer)
            fig_x, fig_y = fig.transFigure.inverted().transform(
                [bbox.x0, bbox.y0]
            )
            ab = AnnotationBbox(imagebox, (fig_x - 0.001, fig_y),
                               xycoords='figure fraction',
                               frameon=False,
                               box_alignment=(1.0, 0.5),
                               pad=0)
            fig.add_artist(ab)

    # Second pass: save with logos in place
    fig.savefig(output_dir / "navigation_outcome.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / "navigation_outcome.png", bbox_inches='tight', dpi=300)
    print(f"Saved navigation_outcome.pdf and navigation_outcome.png")

    plt.close()


if __name__ == "__main__":
    create_navigation_comparison()
