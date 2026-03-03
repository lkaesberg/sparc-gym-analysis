"""
Script to compare actual path length vs steps taken for SPaRC-Gym Traceback.
This shows the efficiency of traceback - how many steps are taken to produce a path of a certain length.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

from matplotlib.offsetbox import AnnotationBbox

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
    get_model_imagebox,
)


def load_jsonl_data(filepath):
    """Load all entries from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def clean_path(path):
    """Remove loops/detours from a path.
    
    If the same coordinate is visited twice, remove everything between
    the first and second occurrence (keeping only the second occurrence).
    
    Example: [A, B, C, B, D] -> [A, B, D]
    """
    if not path:
        return []
    
    # Convert path to list of tuples for hashing
    # Handle both dict format {'x': x, 'y': y} and list format [x, y]
    path_tuples = []
    for p in path:
        if isinstance(p, dict):
            path_tuples.append((p.get('x'), p.get('y')))
        else:
            path_tuples.append(tuple(p))
    
    cleaned = []
    for coord in path_tuples:
        # Check if this coordinate is already in the cleaned path
        if coord in cleaned:
            # Remove everything from the first occurrence onwards
            idx = cleaned.index(coord)
            cleaned = cleaned[:idx]
        cleaned.append(coord)
    
    return cleaned


def extract_traceback_steps_vs_path(results_dir):
    """Extract steps_taken and cleaned path length from SPaRC-Gym Traceback files.
    
    Note: extracted_path includes the starting position, so the number of 
    path segments (edges) is len(path) - 1. Each step corresponds to one edge.
    """
    results_path = Path(results_dir)
    
    steps_taken_list = []
    path_edges_list = []  # Number of edges = path_length - 1
    
    for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
        if "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        data = load_jsonl_data(jsonl_file)
        
        for entry in data:
            result = entry.get('result', {})
            steps_taken = result.get('steps_taken')
            extracted_path = result.get('extracted_path', [])
            
            if steps_taken is not None and extracted_path:
                # Clean the path to remove backtracking detours
                cleaned = clean_path(extracted_path)
                # Path edges = number of moves = len(path) - 1 (starting point doesn't count)
                path_edges = max(0, len(cleaned) - 1)
                steps_taken_list.append(steps_taken)
                path_edges_list.append(path_edges)
    
    return np.array(steps_taken_list), np.array(path_edges_list)


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

MODEL_FAMILY_COLORS = {
    "openai": "#F79F1F",
    "Qwen": "#A47AFF",
    "deepseek": "#61DB7E",
    "google": "#4285F4",
    "nvidia": "#76B900",
    "allenai": "#FF615C",
    "mistralai": "#FF9500",
}


def get_model_family_color(model_name):
    for family, color in MODEL_FAMILY_COLORS.items():
        if family.lower() in model_name.lower():
            return color
    return "#808080"


def extract_traceback_steps_vs_path_per_model(results_dir):
    """Extract steps_taken and cleaned path length per model."""
    results_path = Path(results_dir)

    per_model = {}  # model_name -> {'steps': [], 'path_edges': []}

    for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
        if "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue

        model_name = jsonl_file.stem.replace("_gym_traceback", "")
        data = load_jsonl_data(jsonl_file)

        steps_list = []
        path_edges_list = []
        for entry in data:
            result = entry.get('result', {})
            steps_taken = result.get('steps_taken')
            extracted_path = result.get('extracted_path', [])

            if steps_taken is not None and extracted_path:
                cleaned = clean_path(extracted_path)
                path_edges = max(0, len(cleaned) - 1)
                steps_list.append(steps_taken)
                path_edges_list.append(path_edges)

        if steps_list:
            per_model[model_name] = {
                'steps': np.array(steps_list),
                'path_edges': np.array(path_edges_list),
            }

    return per_model


def create_comparison_plot(results_dir, output_path=None, filter_max_steps=True):
    """Create a 2-panel figure: scatter of steps vs path edges + per-model ratio."""
    setup_plot_style(use_latex=True)

    print("Extracting traceback data...")
    steps, path_edges = extract_traceback_steps_vs_path(results_dir)
    per_model = extract_traceback_steps_vs_path_per_model(results_dir)

    # Filter out entries where steps hit the 100 limit
    if filter_max_steps:
        mask = steps < 100
        steps_filtered = steps[mask]
        path_edges_filtered = path_edges[mask]
        print(f"Data points: {len(steps)} total, {len(steps_filtered)} after filtering (removed {len(steps) - len(steps_filtered)} at 100-step limit)")
    else:
        steps_filtered = steps
        path_edges_filtered = path_edges
        print(f"Data points: {len(steps)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 3.0))

    # --- Left: scatter plot ---
    ax1.scatter(path_edges_filtered, steps_filtered, alpha=0.4, s=12,
                color='#7B1FA2', edgecolors='none', rasterized=True)

    max_path = path_edges_filtered.max()
    max_steps = steps_filtered.max()
    diag_max = max(max_path, max_steps) * 1.05
    ax1.plot([0, diag_max], [0, diag_max], color='#666666', linestyle='--',
             linewidth=1.5, label='No backtracking', zorder=1)

    z = np.polyfit(path_edges_filtered, steps_filtered, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max_path, 100)
    ax1.plot(x_line, p(x_line), color='#E65100', linewidth=2.5,
             label=f'Linear fit (slope={z[0]:.2f})', zorder=2)

    valid_mask = steps_filtered > 0
    efficiency = path_edges_filtered[valid_mask] / steps_filtered[valid_mask]
    print(f"\nStatistics (filtered):")
    print(f"  Mean steps: {steps_filtered.mean():.1f}")
    print(f"  Mean path edges: {path_edges_filtered.mean():.1f}")
    print(f"  Mean efficiency (edges/steps): {efficiency.mean():.3f}")
    print(f"  Median efficiency: {np.median(efficiency):.3f}")
    print(f"  Slope interpretation: ~{z[0]:.1f} steps per path edge")

    ax1.set_xlabel('Final Path Edges (moves)')
    ax1.set_ylabel('Total Steps Taken')
    ax1.set_xlim(0, max_path * 1.05)
    ax1.set_ylim(0, max_steps * 1.05)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    ax1.set_title('(a) Steps vs Path Length', fontsize=10, fontweight='bold')

    # --- Right: per-model steps/path_edges ratio ---
    model_ratios = {}
    for model_name, data in per_model.items():
        s = data['steps']
        pe = data['path_edges']
        if filter_max_steps:
            m = s < 100
            s, pe = s[m], pe[m]
        valid = pe > 0
        if valid.sum() > 0:
            ratios = s[valid] / pe[valid]
            model_ratios[model_name] = {
                'mean': ratios.mean(),
                'median': np.median(ratios),
                'std': ratios.std(),
                'ratios': ratios,
            }

    sorted_models = sorted(model_ratios.keys(),
                           key=lambda m: model_ratios[m]['median'])
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in sorted_models]
    medians = [model_ratios[m]['median'] for m in sorted_models]
    colors = [get_model_family_color(m) for m in sorted_models]

    x_pos = np.arange(len(sorted_models))
    box_data = [model_ratios[m]['ratios'] for m in sorted_models]
    bp = ax2.boxplot(box_data, positions=x_pos, widths=0.55, patch_artist=True,
                     showfliers=False, medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_names, fontsize=8, rotation=45, ha='right')
    ax2.set_ylabel('Steps / Path Edges')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_title('(b) Backtracking Ratio by Model', fontsize=10, fontweight='bold')

    print("\nPer-model backtracking ratio (steps / path edges):")
    for m in sorted_models:
        n = MODEL_DISPLAY_NAMES.get(m, m)
        r = model_ratios[m]
        print(f"  {n:<20s} median={r['median']:.2f}  mean={r['mean']:.2f}  std={r['std']:.2f}")

    plt.tight_layout()

    if output_path:
        # First pass: save to finalise layout so text positions are stable
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        renderer = fig.canvas.get_renderer()

        # X-axis logos on ax2 (per-model boxplot)
        for tick_label in ax2.get_xticklabels():
            name = tick_label.get_text()
            imagebox = get_model_imagebox(name, zoom_factor=0.65, rotation=45)
            if not imagebox:
                continue
            bbox = tick_label.get_window_extent(renderer)
            fig_x, fig_y = fig.transFigure.inverted().transform(
                [bbox.x0, bbox.y0]
            )
            ab = AnnotationBbox(imagebox, (fig_x - 0.01, fig_y),
                               xycoords='figure fraction',
                               frameon=False,
                               box_alignment=(1.0, 0.5),
                               pad=0)
            fig.add_artist(ab)

        # Second pass: save with logos in place
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")

    plt.close(fig)
    return fig, (ax1, ax2)


def create_efficiency_histogram(results_dir, output_path=None):
    """Create a histogram of efficiency (path_edges / steps).
    
    Efficiency is now correctly calculated as (path_length - 1) / steps,
    where path_length - 1 is the number of edges (moves) in the path.
    Maximum efficiency is 1.0 (no backtracking needed).
    """
    setup_plot_style(use_latex=True)
    
    steps, path_edges = extract_traceback_steps_vs_path(results_dir)
    
    # Calculate efficiency (avoid division by zero)
    valid_mask = steps > 0
    efficiency = path_edges[valid_mask] / steps[valid_mask]
    
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, 2.5))
    
    ax.hist(efficiency, bins=50, color='#7B1FA2', edgecolor='white', alpha=0.8)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='Perfect efficiency')
    ax.axvline(x=efficiency.mean(), color='#E65100', linestyle='-', linewidth=2, 
               label=f'Mean = {efficiency.mean():.3f}')
    
    ax.set_xlabel('Efficiency (Path Edges / Steps)')
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.close(fig)
    return fig, ax


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    print("=" * 60)
    print("Creating traceback steps vs path length comparison...")
    print("=" * 60)
    
    # Scatter plot
    output_pdf = Path(__file__).parent / "traceback_steps_vs_path.pdf"
    output_png = Path(__file__).parent / "traceback_steps_vs_path.png"
    create_comparison_plot(results_dir, output_pdf)
    create_comparison_plot(results_dir, output_png)
    
    # Efficiency histogram
    print("\n" + "=" * 60)
    print("Creating efficiency histogram...")
    print("=" * 60)
    hist_pdf = Path(__file__).parent / "traceback_efficiency_hist.pdf"
    hist_png = Path(__file__).parent / "traceback_efficiency_hist.png"
    create_efficiency_histogram(results_dir, hist_pdf)
    create_efficiency_histogram(results_dir, hist_png)


if __name__ == "__main__":
    main()
