"""
Script to create a combined figure with two subplots:
1. SPaRC-Gym vs SPaRC difference
2. SPaRC-Gym Traceback vs SPaRC-Gym difference
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
    get_model_imagebox,
    get_model_color,
    desaturate_color,
    MODEL_COLORS,
)

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


EXCLUDED_MODELS = {
    "Qwen_Qwen3-14B",
    "Qwen_Qwen3-4B",
    "9Tobi_ragen_sparc_qwen3_4B_CW3",
}


def get_model_family_color(model_name, display_name=None):
    """Get color based on model family/provider, checking MODEL_COLORS first."""
    if display_name and display_name in MODEL_COLORS:
        return MODEL_COLORS[display_name]
    for family, color in MODEL_FAMILY_COLORS.items():
        if family.lower() in model_name.lower():
            return color
    return "#808080"


def extract_accuracy_from_stats(stats_file):
    """Extract accuracy percentage from a stats CSV file."""
    df = pd.read_csv(stats_file)
    solved_row = df[df['Metric'] == 'Correctly Solved']
    if len(solved_row) > 0:
        percentage_str = solved_row['Percentage'].values[0]
        match = re.search(r'([\d.]+)%', str(percentage_str))
        if match:
            return float(match.group(1))
    return 0.0


def load_sparc_gym_comparison(results_dir):
    """Load accuracy stats for models with both SPaRC and SPaRC-Gym variants."""
    results_path = Path(results_dir)
    
    sparc_files = list(results_path.glob("*_stats.csv"))
    sparc_files = [f for f in sparc_files 
                   if '_gym_' not in f.name 
                   and 'archive' not in str(f)
                   and 'visual' not in f.name]
    
    model_data = []
    for sparc_file in sparc_files:
        model_name = sparc_file.name.replace("_stats.csv", "")
        if model_name in EXCLUDED_MODELS:
            continue
        gym_file = results_path / f"{model_name}_gym_stats.csv"
        if not gym_file.exists():
            continue
        
        sparc_accuracy = extract_accuracy_from_stats(sparc_file)
        gym_accuracy = extract_accuracy_from_stats(gym_file)
        diff = gym_accuracy - sparc_accuracy
        
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = get_model_family_color(model_name, display_name)
        
        model_data.append({
            'model_name': model_name,
            'display_name': display_name,
            'base_accuracy': sparc_accuracy,
            'variant_accuracy': gym_accuracy,
            'difference': diff,
            'color': color
        })
    
    model_data.sort(key=lambda x: x['difference'], reverse=True)
    return model_data


def load_traceback_comparison(results_dir):
    """Load accuracy stats for models with both traceback and non-traceback variants."""
    results_path = Path(results_dir)
    
    main_stats_files = list(results_path.glob("*_gym_stats.csv"))
    main_stats_files = [f for f in main_stats_files if 'traceback' not in f.name and 'visual' not in f.name]
    
    model_data = []
    for main_file in main_stats_files:
        model_name = main_file.name.replace("_gym_stats.csv", "")
        if model_name in EXCLUDED_MODELS:
            continue
        traceback_file = results_path / f"{model_name}_gym_traceback_stats.csv"
        if not traceback_file.exists():
            continue
        
        main_accuracy = extract_accuracy_from_stats(main_file)
        traceback_accuracy = extract_accuracy_from_stats(traceback_file)
        diff = traceback_accuracy - main_accuracy
        
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = get_model_family_color(model_name, display_name)
        
        model_data.append({
            'model_name': model_name,
            'display_name': display_name,
            'base_accuracy': main_accuracy,
            'variant_accuracy': traceback_accuracy,
            'difference': diff,
            'color': color
        })
    
    model_data.sort(key=lambda x: x['difference'], reverse=True)
    return model_data


def add_bars_to_subplot(ax, model_data, title):
    """Add bars and annotations to a subplot."""
    display_names = [m['display_name'] for m in model_data]
    differences = [m['difference'] for m in model_data]
    colors = [m['color'] if d >= 0 else desaturate_color(m['color'], 0.5) 
              for m, d in zip(model_data, differences)]
    
    x_pos = np.arange(len(display_names))
    bars = ax.bar(x_pos, differences, color=colors, edgecolor='white', linewidth=0.5)
    
    for i, (bar, diff, m) in enumerate(zip(bars, differences, model_data)):
        height = bar.get_height()
        
        if diff >= 0:
            va = 'bottom'
            y_offset = 3
            logo_offset = 13
        else:
            va = 'top'
            y_offset = -3
            logo_offset = -11

        sign = '+' if diff > 0 else ''
        ax.annotate(f'{sign}{diff:.1f}\\%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', va=va,
                    fontsize=7,
                    fontweight='bold',
                    color=m['color'])

        imagebox = get_model_imagebox(m['display_name'])
        if imagebox:
            ab = AnnotationBbox(imagebox, (bar.get_x() + bar.get_width() / 2, height),
                               xybox=(0, logo_offset + (5 if diff >= 0 else -5)),
                               xycoords='data',
                               boxcoords="offset points",
                               frameon=False,
                               pad=0)
            ax.add_artist(ab)
    
    ax.set_ylabel('$\Delta$ Accuracy (\\%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(title, fontweight='bold')


def create_combined_chart(sparc_gym_data, traceback_data, output_path=None):
    """Create the combined figure with two subplots."""
    setup_plot_style(use_latex=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 2.5), sharey=True)
    
    # Left subplot: SPaRC-Gym vs SPaRC
    add_bars_to_subplot(ax1, sparc_gym_data, "(a) SPaRC-Gym vs SPaRC")
    
    # Right subplot: Traceback vs Non-traceback
    add_bars_to_subplot(ax2, traceback_data, "(b) Traceback vs Standard")
    
    # Calculate shared y-limits from both datasets
    all_diffs = [m['difference'] for m in sparc_gym_data] + [m['difference'] for m in traceback_data]
    y_max = max(all_diffs) if max(all_diffs) > 0 else 0
    y_min = min(all_diffs) if min(all_diffs) < 0 else 0
    padding = max(abs(y_max), abs(y_min)) * 0.7
    shared_ylim = (y_min - padding, y_max + padding)
    
    ax1.set_ylim(shared_ylim)
    ax2.set_ylabel('')

    # Set yticks at 2.5 intervals
    from matplotlib.ticker import MultipleLocator
    ax1.yaxis.set_major_locator(MultipleLocator(2.5))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.close(fig)
    return fig, (ax1, ax2)


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "combined_comparison.pdf"
    output_png = Path(__file__).parent / "combined_comparison.png"
    
    print("Loading model statistics...")
    sparc_gym_data = load_sparc_gym_comparison(results_dir)
    traceback_data = load_traceback_comparison(results_dir)
    
    print("\nSPaRC-Gym vs SPaRC:")
    print("-" * 50)
    for m in sparc_gym_data:
        sign = '+' if m['difference'] > 0 else ''
        print(f"{m['display_name']:<20} {sign}{m['difference']:>6.1f}%")
    
    print("\nTraceback vs Standard:")
    print("-" * 50)
    for m in traceback_data:
        sign = '+' if m['difference'] > 0 else ''
        print(f"{m['display_name']:<20} {sign}{m['difference']:>6.1f}%")
    
    print("\nCreating combined chart...")
    create_combined_chart(sparc_gym_data, traceback_data, output_pdf)
    create_combined_chart(sparc_gym_data, traceback_data, output_png)


if __name__ == "__main__":
    main()
