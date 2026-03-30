"""
Script to create a 3-panel line plot showing accuracy by difficulty level
for each model individually, with subplots for SPaRC, Gym w/o backtracking, and Gym w/ backtracking.
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
    figure_fraction_anchor_from_display_xy,
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
    "9Tobi":     "#9070F0",  # Fine-tuned Qwen → Qwen purple
    "mistralai": "#D96818",  # Magistral → warm orange (Mistral logo)
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
    if variant == "Gym w/ backtracking":
        return filename.replace("_gym_traceback_stats.csv", "")
    elif variant == "Gym w/o backtracking":
        return filename.replace("_gym_stats.csv", "")
    else:
        return filename.replace("_stats.csv", "")


def categorize_stats_files(results_dir):
    """Categorize stats files into SPaRC, Gym w/o backtracking, and Gym w/ backtracking."""
    results_path = Path(results_dir)

    categories = {"SPaRC": [], "Gym w/o backtracking": [], "Gym w/ backtracking": []}

    for stats_file in sorted(results_path.glob("*_stats.csv")):
        if "archive" in str(stats_file):
            continue

        filename = stats_file.name

        # Skip unwanted variants
        if any(p in filename.lower() for p in SKIP_PATTERNS):
            continue

        if "_gym_traceback_stats.csv" in filename:
            categories["Gym w/ backtracking"].append(stats_file)
        elif "_gym_stats.csv" in filename:
            categories["Gym w/o backtracking"].append(stats_file)
        elif "_stats.csv" in filename and "_gym_" not in filename:
            categories["SPaRC"].append(stats_file)

    return categories


def create_difficulty_comparison_plot(categorized_files, output_path=None):
    """Create 3-panel line plot with individual model lines."""
    setup_plot_style(use_latex=True)

    variant_names = ["SPaRC", "Gym w/o backtracking", "Gym w/ backtracking"]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_INCHES, 2.0), sharey=True)

    difficulties = np.array([1, 2, 3, 4, 5])

    # Collect all models across panels for unified legend
    all_models_seen = {}

    for i, (ax, variant) in enumerate(zip(axes, variant_names)):
        files = categorized_files.get(variant, [])
        if not files:
            ax.set_title(f'({chr(97 + i)}) {variant}')
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

        ax.set_title(f'({chr(97 + i)}) {variant}')
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
               handlelength=2, handletextpad=2)

    # Add logos: draw first so handle bboxes are valid. Do not use xycoords=legend_handle
    # with fig.add_artist — transform chain is wrong; use display bboxes + figure fraction.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for label, legend_handle in zip(labels, leg.legend_handles):
        imagebox = get_model_imagebox(label, zoom_factor=0.8)
        if not imagebox:
            continue

        bbox = legend_handle.get_window_extent(renderer)
        xd = bbox.x0 + 0.25 * bbox.width
        yd = bbox.y0 + 0.5 * bbox.height
        fx, fy = figure_fraction_anchor_from_display_xy(fig, (xd, yd), (-0.0225, 0.0))
        ab = AnnotationBbox(imagebox, (fx, fy),
                           xybox=(19, 0),
                           xycoords='figure fraction',
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
    results_dir = Path(__file__).parent / "results" / "spatial_gym"
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
