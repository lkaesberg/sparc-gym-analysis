"""
Script to create a bar chart showing the difference between traceback and non-traceback variants.
Positive values mean traceback performed better, negative means it performed worse.
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
    COLUMN_WIDTH_INCHES,
    get_model_imagebox,
    desaturate_color,
)

# Define colors for different model families
MODEL_FAMILY_COLORS = {
    "openai": "#F79F1F",      # Orange for OpenAI
    "Qwen": "#A47AFF",        # Purple for Qwen
    "deepseek": "#61DB7E",    # Green for DeepSeek
    "google": "#4285F4",      # Google Blue
    "nvidia": "#76B900",      # Nvidia Green
    "allenai": "#FF615C",     # Red for AllenAI
}

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "allenai_Olmo-3.1-32B-Think": "OLMo 3.1 32B",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron 49B",
    "Qwen_Qwen3-32B": "Qwen 3 32B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "R1 Distill 32B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
}


def get_model_family_color(model_name):
    """Get color based on model family/provider."""
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


def load_traceback_comparison(results_dir):
    """Load accuracy stats for models with both traceback and non-traceback variants."""
    results_path = Path(results_dir)
    
    # Find all non-traceback stats files
    main_stats_files = list(results_path.glob("*_gym_stats.csv"))
    main_stats_files = [f for f in main_stats_files if 'traceback' not in f.name and 'visual' not in f.name]
    
    model_data = []
    for main_file in main_stats_files:
        # Extract model name from filename
        model_name = main_file.name.replace("_gym_stats.csv", "")
        
        # Check if traceback variant exists
        traceback_file = results_path / f"{model_name}_gym_traceback_stats.csv"
        if not traceback_file.exists():
            continue
        
        # Get accuracies
        main_accuracy = extract_accuracy_from_stats(main_file)
        traceback_accuracy = extract_accuracy_from_stats(traceback_file)
        
        # Calculate difference (traceback - main)
        diff = traceback_accuracy - main_accuracy
        
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = get_model_family_color(model_name)
        
        model_data.append({
            'model_name': model_name,
            'display_name': display_name,
            'main_accuracy': main_accuracy,
            'traceback_accuracy': traceback_accuracy,
            'difference': diff,
            'color': color
        })
    
    # Sort by difference (descending - best improvement first)
    model_data.sort(key=lambda x: x['difference'], reverse=True)
    
    return model_data


def create_traceback_diff_chart(model_data, output_path=None):
    """Create the traceback difference bar chart."""
    setup_plot_style(use_latex=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, 3.5))
    
    # Extract data for plotting
    display_names = [m['display_name'] for m in model_data]
    differences = [m['difference'] for m in model_data]
    colors = [m['color'] if d >= 0 else desaturate_color(m['color'], 0.5) 
              for m, d in zip(model_data, differences)]
    
    # Create bar positions
    x_pos = np.arange(len(display_names))
    
    # Create bars
    bars = ax.bar(x_pos, differences, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add percentage labels and logos
    for i, (bar, diff, m) in enumerate(zip(bars, differences, model_data)):
        height = bar.get_height()
        
        # Position label above or below bar depending on sign
        if diff >= 0:
            va = 'bottom'
            y_offset = 3
            logo_offset = 18
        else:
            va = 'top'
            y_offset = -3
            logo_offset = -18
        
        # Add percentage label with sign
        sign = '+' if diff > 0 else ''
        ax.annotate(f'{sign}{diff:.1f}\\%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', va=va,
                    fontsize=9,
                    fontweight='bold',
                    color=m['color'])
        
        # Add logo
        imagebox = get_model_imagebox(m['display_name'])
        if imagebox:
            ab = AnnotationBbox(imagebox, (bar.get_x() + bar.get_width() / 2, height),
                               xybox=(0, logo_offset + (5 if diff >= 0 else -5)),
                               xycoords='data',
                               boxcoords="offset points",
                               frameon=False,
                               pad=0)
            ax.add_artist(ab)
    
    # Customize axes
    ax.set_ylabel('Accuracy Difference (\\%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Set y-axis limits with padding for logos
    y_max = max(differences) if max(differences) > 0 else 0
    y_min = min(differences) if min(differences) < 0 else 0
    padding = max(abs(y_max), abs(y_min)) * 0.5
    ax.set_ylim(y_min - padding, y_max + padding)
    
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
    output_pdf = Path(__file__).parent / "traceback_comparison.pdf"
    output_png = Path(__file__).parent / "traceback_comparison.png"
    
    # Load data
    print("Loading model statistics...")
    model_data = load_traceback_comparison(results_dir)
    
    # Print summary
    print("\nTraceback vs Main Comparison (sorted by improvement):")
    print("-" * 70)
    print(f"{'Model':<25} {'Main':>10} {'Traceback':>10} {'Diff':>10}")
    print("-" * 70)
    for m in model_data:
        sign = '+' if m['difference'] > 0 else ''
        print(f"{m['display_name']:<25} {m['main_accuracy']:>9.1f}% {m['traceback_accuracy']:>9.1f}% {sign}{m['difference']:>9.1f}%")
    print("-" * 70)
    
    # Create chart
    print("\nCreating bar chart...")
    create_traceback_diff_chart(model_data, output_pdf)
    create_traceback_diff_chart(model_data, output_png)


if __name__ == "__main__":
    main()
