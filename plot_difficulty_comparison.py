"""
Script to create a 3-panel line plot showing accuracy by difficulty level
for each model individually, with subplots for SPaRC, Gym w/o traceback, and Gym w/ traceback.
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
    "Qwen_Qwen3-14B": "Qwen 3 14B",
    "Qwen_Qwen3-4B": "Qwen 3 4B",
    "Qwen_Qwen3-0.6B": "Qwen 3 0.6B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "R1 Distill 32B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "9Tobi_ragen_sparc_qwen3_4B_CW3": "Qwen 3 4B (FT)",
    "mistralai_Magistral-Small-2507": "Magistral Small",
    "Qwen_Qwen3-VL-32B-Thinking": "Qwen 3 VL 32B",
}

# Model family colors (for models not in MODEL_COLORS)
MODEL_FAMILY_COLORS = {
    "openai": "#F79F1F",
    "Qwen": "#A47AFF",
    "deepseek": "#61DB7E",
    "google": "#4285F4",
    "nvidia": "#76B900",
    "allenai": "#FF615C",
    "9Tobi": "#FFD93D",
    "mistralai": "#FF69B4",
}

MODEL_MARKERS = {
    "GPT-OSS 120B": "o",
    "OLMo 3.1 32B": "s",
    "Nemotron 49B": "^",
    "Qwen 3 32B": "D",
    "Qwen 3 14B": "v",
    "Qwen 3 4B": "p",
    "Qwen 3 0.6B": "h",
    "R1 Distill 32B": "P",
    "Gemma 3 27B": "*",
    "Magistral Small": "X",
    "Qwen 3 4B (FT)": "8",
}

# Skip no-reason / visual / ablation variants
SKIP_PATTERNS = ["no-reason", "no_reason", "visual", "ablation", "baseline", "astar", "random"]

# Only include the models from the main accuracy chart
INCLUDED_MODELS = {
    "GPT-OSS 120B",
    "OLMo 3.1 32B",
    "Nemotron 49B",
    "Qwen 3 32B",
    "Qwen 3 0.6B",
    "R1 Distill 32B",
    "Gemma 3 27B",
    "Magistral Small",
}


def get_color_for_model(display_name, internal_name):
    """Get color for a model, trying MODEL_COLORS first, then family colors."""
    if display_name in MODEL_COLORS:
        return MODEL_COLORS[display_name]
    for family, color in MODEL_FAMILY_COLORS.items():
        if family.lower() in internal_name.lower():
            return color
    return "#808080"


def extract_difficulty_accuracies(stats_file):
    """Extract accuracy percentages for each difficulty level from a stats CSV file."""
    df = pd.read_csv(stats_file)

    difficulties = {}
    for i in range(1, 6):
        row = df[df['Metric'] == f'Difficulty {i} Solved']
        if len(row) > 0:
            percentage_str = row['Percentage'].values[0]
            match = re.search(r'([\d.]+)%', str(percentage_str))
            if match:
                difficulties[i] = float(match.group(1))

    return difficulties


def get_internal_name(filename, variant):
    """Extract internal model name from filename based on variant type."""
    if variant == "Gym w/ traceback":
        return filename.replace("_gym_traceback_stats.csv", "")
    elif variant == "Gym w/o traceback":
        return filename.replace("_gym_stats.csv", "")
    else:
        return filename.replace("_stats.csv", "")


def categorize_stats_files(results_dir):
    """Categorize stats files into SPaRC, Gym w/o traceback, and Gym w/ traceback."""
    results_path = Path(results_dir)

    categories = {"SPaRC": [], "Gym w/o traceback": [], "Gym w/ traceback": []}

    for stats_file in sorted(results_path.glob("*_stats.csv")):
        if "archive" in str(stats_file):
            continue

        filename = stats_file.name

        # Skip unwanted variants
        if any(p in filename.lower() for p in SKIP_PATTERNS):
            continue

        if "_gym_traceback_stats.csv" in filename:
            categories["Gym w/ traceback"].append(stats_file)
        elif "_gym_stats.csv" in filename:
            categories["Gym w/o traceback"].append(stats_file)
        elif "_stats.csv" in filename and "_gym_" not in filename:
            categories["SPaRC"].append(stats_file)

    return categories


def create_difficulty_comparison_plot(categorized_files, output_path=None):
    """Create 3-panel line plot with individual model lines."""
    setup_plot_style(use_latex=True)

    variant_names = ["SPaRC", "Gym w/o traceback", "Gym w/ traceback"]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_INCHES, 3.0), sharey=True)

    difficulties = np.array([1, 2, 3, 4, 5])

    # Collect all models across panels for unified legend
    all_models_seen = {}

    for i, (ax, variant) in enumerate(zip(axes, variant_names)):
        files = categorized_files.get(variant, [])
        if not files:
            ax.set_title(f'({chr(97 + i)}) {variant}', fontsize=10)
            continue

        for stats_file in files:
            internal_name = get_internal_name(stats_file.name, variant)
            display_name = MODEL_DISPLAY_NAMES.get(internal_name, internal_name)

            # Only include models from the main accuracy chart
            if display_name not in INCLUDED_MODELS:
                continue

            diff_acc = extract_difficulty_accuracies(stats_file)
            if not diff_acc:
                continue

            values = np.array([diff_acc.get(i, 0) for i in difficulties])
            color = get_color_for_model(display_name, internal_name)
            marker = MODEL_MARKERS.get(display_name, "o")

            ax.plot(difficulties, values,
                    color=color, marker=marker, markersize=4,
                    linewidth=1.2, label=display_name,
                    markeredgecolor='white', markeredgewidth=0.3)

            if display_name not in all_models_seen:
                all_models_seen[display_name] = (color, marker)

        ax.set_title(f'({chr(97 + i)}) {variant}', fontsize=10)
        ax.set_xlabel('Difficulty Level')
        ax.set_xticks(difficulties)
        ax.set_xlim(0.7, 5.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Accuracy (\\%)')

    # Create unified legend below the plot — deduplicate across panels
    seen_labels = set()
    handles = []
    labels = []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen_labels:
                seen_labels.add(l)
                handles.append(h)
                labels.append(l)

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    leg = fig.legend(handles, labels, loc='lower center',
               ncol=4, fontsize=7, framealpha=0,
               bbox_to_anchor=(0.5, -0.02),
               handlelength=2, handletextpad=1.4)

    # Add logos to legend - need to draw first to get positions
    fig.canvas.draw()

    # Add logos to legend handles
    for label, legend_handle in zip(labels, leg.legend_handles):
        imagebox = get_model_imagebox(label, zoom_factor=0.8)
        if not imagebox:
            continue

        ab = AnnotationBbox(imagebox, (0.5, 0.5),
                           xybox=(19, 0),
                           xycoords=legend_handle,
                           boxcoords="offset points",
                           frameon=False,
                           box_alignment=(0.5, 0.5),
                           zorder=10)
        fig.add_artist(ab)

    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")

    plt.close(fig)


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "difficulty_comparison.pdf"
    output_png = Path(__file__).parent / "difficulty_comparison.png"

    print("Categorizing stats files...")
    categorized_files = categorize_stats_files(results_dir)

    print("\nFiles found:")
    for variant, files in categorized_files.items():
        print(f"  {variant}: {len(files)} models")
        for f in files:
            internal = get_internal_name(f.name, variant)
            display = MODEL_DISPLAY_NAMES.get(internal, internal)
            print(f"    - {display}")

    print("\nCreating plot...")
    create_difficulty_comparison_plot(categorized_files, output_pdf)
    create_difficulty_comparison_plot(categorized_files, output_png)


if __name__ == "__main__":
    main()
