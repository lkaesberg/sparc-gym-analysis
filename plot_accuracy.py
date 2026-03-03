"""
Script to create an accuracy comparison bar chart for SPaRC-Gym models.
Uses the plot_config.py style settings.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re

from matplotlib.offsetbox import AnnotationBbox

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
    get_model_color,
    get_model_imagebox,
    MODEL_COLORS,
)

# Define colors for different model families (logo-inspired fallback)
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
    "9Tobi":     "#9070F0",  # Fine-tuned Qwen → Qwen purple
    "mistralai": "#D96818",  # Magistral → warm orange (Mistral logo)
}

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "allenai_Olmo-3.1-32B-Think": "OLMo 3.1 32B",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron 49B",
    "Qwen_Qwen3-32B": "Qwen 3 32B",
    "Qwen_Qwen3-14B": "Qwen 3 14B",
    "Qwen_Qwen3-4B": "Qwen 3 4B",
    "Qwen_Qwen3-0.6B": "Qwen 3 0.6B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "R1 Distill 32B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "9Tobi_ragen_sparc_qwen3_4B_CW3": "Qwen 3 4B (FT)",
    "mistralai_Magistral-Small-2507": "Magistral Small",
}


def get_model_family_color(model_name, display_name=None):
    """Get color based on model family/provider, checking MODEL_COLORS first."""
    if display_name and display_name in MODEL_COLORS:
        return MODEL_COLORS[display_name]
    for family, color in MODEL_FAMILY_COLORS.items():
        if family.lower() in model_name.lower():
            return color
    return "#808080"  # Gray fallback


def extract_accuracy_from_stats(stats_file):
    """Extract accuracy percentage from a stats CSV file."""
    df = pd.read_csv(stats_file)
    
    # Find the "Correctly Solved" row
    solved_row = df[df['Metric'] == 'Correctly Solved']
    if len(solved_row) > 0:
        percentage_str = solved_row['Percentage'].values[0]
        # Extract number from percentage string (e.g., "10.6%" -> 10.6)
        match = re.search(r'([\d.]+)%', str(percentage_str))
        if match:
            return float(match.group(1))
    return 0.0


# Models to exclude (only keep Qwen 32B and 0.6B from Qwen family)
EXCLUDED_MODELS = {
    "Qwen_Qwen3-14B",
    "Qwen_Qwen3-4B",
    "9Tobi_ragen_sparc_qwen3_4B_CW3",
    "Qwen_Qwen3-VL-32B-Thinking"
}

# Human solve rate for reference
HUMAN_SOLVE_RATE = 98.0


def load_all_model_stats(results_dir):
    """Load accuracy stats for all non-traceback models."""
    results_path = Path(results_dir)
    
    # Find all stats files (exclude traceback and visual)
    stats_files = list(results_path.glob("*_gym_stats.csv"))
    
    # Filter out traceback and visual files
    stats_files = [f for f in stats_files if 'traceback' not in f.name and 'visual' not in f.name]
    
    model_data = []
    for stats_file in stats_files:
        # Extract model name from filename
        model_name = stats_file.name.replace("_gym_stats.csv", "")
        
        # Skip excluded models (non-32B Qwen variants)
        if model_name in EXCLUDED_MODELS:
            continue
        
        accuracy = extract_accuracy_from_stats(stats_file)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = get_model_family_color(model_name, display_name)
        
        model_data.append({
            'model_name': model_name,
            'display_name': display_name,
            'accuracy': accuracy,
            'color': color
        })
    
    # Sort by accuracy (descending)
    model_data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return model_data


def create_accuracy_bar_chart(model_data, output_path=None):
    """Create the accuracy comparison bar chart."""
    setup_plot_style(use_latex=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES*1.3, 2.5))
    
    # Add human as first entry
    all_display_names = ['Human'] + [m['display_name'] for m in model_data]
    all_accuracies = [HUMAN_SOLVE_RATE] + [m['accuracy'] for m in model_data]
    all_colors = ['#2E86AB'] + [m['color'] for m in model_data]  # Blue for human
    
    # Create bar positions
    x_pos = np.arange(len(all_display_names))
    
    # Create bars
    bars = ax.bar(x_pos, all_accuracies, color=all_colors, edgecolor='white', linewidth=0.5, zorder=2)
    
    # Add dashed vertical line to separate human from models
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # Add percentage labels and logos on top of bars
    for i, (bar, acc, color) in enumerate(zip(bars, all_accuracies, all_colors)):
        height = bar.get_height()
        
        if i == 0:
            # Human bar: put text and icon at the top of the bar
            ax.annotate(f'{acc:.0f}\\%',
                        xy=(bar.get_x() + bar.get_width() / 2, height - 18),
                        ha='center', va='top',
                        fontsize=8,
                        fontweight='bold',
                        color='white')
            # Add human logo inside the bar (above the text)
            imagebox = get_model_imagebox('Human')
            if imagebox:
                ab = AnnotationBbox(imagebox, (bar.get_x() + bar.get_width() / 2, height - 8),
                                   xycoords='data',
                                   frameon=False,
                                   pad=0)
                ax.add_artist(ab)
        else:
            # Model bars: percentage label above
            ax.annotate(f'{acc:.1f}\\%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8,
                        fontweight='bold',
                        color=color)
            
            # Add logo above the percentage
            m = model_data[i - 1]
            imagebox = get_model_imagebox(m['display_name'])
            if imagebox:
                ab = AnnotationBbox(imagebox, (bar.get_x() + bar.get_width() / 2, height),
                                   xybox=(0, 20),  # Offset above percentage
                                   xycoords='data',
                                   boxcoords="offset points",
                                   frameon=False,
                                   pad=0)
                ax.add_artist(ab)
    
    # Customize axes
    ax.set_ylabel('Accuracy (\\%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_display_names, rotation=45, ha='right', fontsize=9)
    
    # Add horizontal dashed line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Set axis limits
    ax.set_xlim(-0.5, len(all_display_names) - 0.5)
    ax.set_ylim(0, 100)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.close(fig)
    
    return fig, ax


def main():
    # Define paths
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "accuracy_comparison.pdf"
    output_png = Path(__file__).parent / "accuracy_comparison.png"
    
    # Load data
    print("Loading model statistics...")
    model_data = load_all_model_stats(results_dir)
    
    # Print summary
    print("\nModel Accuracies (sorted by accuracy):")
    print("-" * 50)
    for m in model_data:
        print(f"{m['display_name']:30s} {m['accuracy']:6.1f}%")
    print("-" * 50)
    
    # Create chart and save as PDF
    print("\nCreating bar chart...")
    create_accuracy_bar_chart(model_data, output_pdf)
    
    # Also save as PNG
    create_accuracy_bar_chart(model_data, output_png)


if __name__ == "__main__":
    main()
