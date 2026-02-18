"""
Script to compare navigation outcomes (finished vs deadlocked) for all models.
Left panel: SPARC-Gym, Right panel: SPARC-Gym Traceback
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    get_model_color,
)

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "allenai_Olmo-3.1-32B-Think": "OLMo 3.1 32B",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron 49B",
    "Qwen_Qwen3-32B": "Qwen 3 32B",
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
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
    "google_gemma-3-27b-it",
    "mistralai_Magistral-Small-2507",
}

# Model family colors
MODEL_FAMILY_COLORS = {
    "openai": "#F79F1F",
    "Qwen": "#A47AFF",
    "deepseek": "#61DB7E",
    "google": "#4285F4",
    "nvidia": "#76B900",
    "allenai": "#FF615C",
    "mistralai": "#FF69B4",
}


def get_model_family_color(model_name):
    """Get color based on model family/provider."""
    for family, color in MODEL_FAMILY_COLORS.items():
        if family.lower() in model_name.lower():
            return color
    return "#808080"


def extract_navigation_stats(stats_file):
    """Extract navigation outcome stats from a stats CSV file."""
    df = pd.read_csv(stats_file)
    
    # Find the "Reached End" row
    reached_row = df[df['Metric'] == 'Reached End']
    stuck_row = df[df['Metric'] == 'No Legal Actions']
    
    reached_pct = 0.0
    stuck_pct = 0.0
    
    if len(reached_row) > 0:
        pct_str = reached_row['Percentage'].values[0]
        match = re.search(r'([\d.]+)%', str(pct_str))
        if match:
            reached_pct = float(match.group(1))
    
    if len(stuck_row) > 0:
        pct_str = stuck_row['Percentage'].values[0]
        match = re.search(r'([\d.]+)%', str(pct_str))
        if match:
            stuck_pct = float(match.group(1))
    
    return reached_pct, stuck_pct


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
        
        reached_pct, stuck_pct = extract_navigation_stats(stats_file)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = get_model_family_color(model_name)
        
        model_data.append({
            'model_name': model_name,
            'display_name': display_name,
            'reached': reached_pct,
            'stuck': stuck_pct,
            'color': color
        })
    
    # Sort by reached percentage (descending)
    model_data.sort(key=lambda x: x['reached'], reverse=True)
    
    return model_data


def create_navigation_comparison():
    """Create the navigation outcome comparison chart."""
    setup_plot_style()
    
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    # Load data for both variants
    gym_data = load_navigation_stats(results_dir, 'gym')
    traceback_data = load_navigation_stats(results_dir, 'traceback')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 2.8))
    
    width = 0.6
    
    # === Left plot: SPARC-Gym ===
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
                        fontsize=7, fontweight='bold', color='white')
    
    ax1.set_ylabel('Rate (\\%)')
    ax1.set_title('SPARC-Gym', fontsize=10, fontweight='bold')
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
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
    
    # === Right plot: SPARC-Gym Traceback ===
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
                        fontsize=7, fontweight='bold', color='white')
    
    ax2.set_ylabel('Rate (\\%)')
    ax2.set_title('SPARC-Gym Traceback', fontsize=10, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=7, rotation=45, ha='right')
    ax2.set_ylim(0, 120)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add legend to right plot
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figures
    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "navigation_outcome.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / "navigation_outcome.png", bbox_inches='tight', dpi=300)
    print(f"Saved navigation_outcome.pdf and navigation_outcome.png")
    
    plt.close()


if __name__ == "__main__":
    create_navigation_comparison()
