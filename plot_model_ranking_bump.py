"""
Bump chart showing how model rankings change across difficulty levels.
This reveals if certain models are relatively better at easy vs hard puzzles.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
)

# Display names for models
MODEL_DISPLAY_NAMES = {
    'openai_gpt-oss-120b': 'GPT-OSS-120B',
    'deepseek-ai_DeepSeek-R1-Distill-Qwen-32B': 'R1-Distill-32B',
    'allenai_Olmo-3.1-32B-Think': 'OLMo-3.1-32B',
    'google_gemma-3-27b-it': 'Gemma-3-27B',
    'nvidia_Llama-3_3-Nemotron-Super-49B-v1_5': 'Nemotron-49B',
    'Qwen_Qwen3-32B': 'Qwen3-32B',
    'Qwen_Qwen3-14B': 'Qwen3-14B',
    'Qwen_Qwen3-8B': 'Qwen3-8B',
    'Qwen_Qwen3-4B': 'Qwen3-4B',
    'Qwen_Qwen3-1.7B': 'Qwen3-1.7B',
    'Qwen_Qwen3-0.6B': 'Qwen3-0.6B',
    'mistralai_Magistral-Small-2507': 'Magistral-Small',
    '9Tobi_ragen_sparc_qwen3_4B_CW3': 'RAGEN-Qwen3-4B',
}


def load_jsonl_data(filepath):
    """Load all entries from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def calculate_accuracy_by_difficulty(results_dir, variant='gym'):
    """Calculate accuracy for each model at each difficulty level."""
    results_path = Path(results_dir)
    
    # model -> difficulty_level -> {correct, total}
    model_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    if variant == 'gym':
        pattern = "*_gym.jsonl"
        suffix = "_gym"
    elif variant == 'traceback':
        pattern = "*_gym_traceback.jsonl"
        suffix = "_gym_traceback"
    else:  # sparc
        pattern = "*.jsonl"
        suffix = ""
    
    for jsonl_file in results_path.glob(pattern):
        if variant == 'sparc' and "_gym" in jsonl_file.name:
            continue
        if "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem.replace(suffix, "")
        data = load_jsonl_data(jsonl_file)
        
        for entry in data:
            difficulty_level = entry.get('difficulty_level')
            result = entry.get('result', {})
            solved = result.get('solved', False)
            
            if difficulty_level is not None:
                model_stats[model_name][difficulty_level]['total'] += 1
                if solved:
                    model_stats[model_name][difficulty_level]['correct'] += 1
    
    # Calculate accuracy
    accuracy = {}
    for model in model_stats:
        accuracy[model] = {}
        for diff_level in model_stats[model]:
            stats = model_stats[model][diff_level]
            if stats['total'] > 0:
                accuracy[model][diff_level] = stats['correct'] / stats['total'] * 100
            else:
                accuracy[model][diff_level] = 0
    
    return accuracy


def get_rankings_by_difficulty(accuracy_data):
    """Get model rankings at each difficulty level."""
    # Find all difficulty levels
    all_levels = set()
    for model in accuracy_data:
        all_levels.update(accuracy_data[model].keys())
    
    difficulty_levels = sorted(all_levels)
    
    rankings = {level: [] for level in difficulty_levels}
    
    for level in difficulty_levels:
        # Get accuracy for each model at this level
        model_accs = []
        for model in accuracy_data:
            acc = accuracy_data[model].get(level, 0)
            model_accs.append((model, acc))
        
        # Sort by accuracy (descending) and assign ranks
        model_accs.sort(key=lambda x: -x[1])
        for rank, (model, acc) in enumerate(model_accs, 1):
            rankings[level].append((model, rank, acc))
    
    return rankings, difficulty_levels


def create_bump_chart(results_dir, output_path=None, variant='gym'):
    """Create a bump chart showing model ranking changes across difficulty levels."""
    setup_plot_style(use_latex=True)
    
    print(f"Calculating accuracy by difficulty ({variant})...")
    accuracy = calculate_accuracy_by_difficulty(results_dir, variant)
    
    if not accuracy:
        print("No data found!")
        return None, None
    
    rankings, difficulty_levels = get_rankings_by_difficulty(accuracy)
    
    # Print rankings
    print(f"\nModel rankings by difficulty level ({variant}):")
    for level in difficulty_levels:
        print(f"\nDifficulty {level}:")
        for model, rank, acc in rankings[level]:
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            print(f"  {rank}. {display_name}: {acc:.1f}%")
    
    # Get all models
    models = list(accuracy.keys())
    n_models = len(models)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, 3.5))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Plot lines for each model
    for i, model in enumerate(models):
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        
        x_vals = []
        y_vals = []  # rank (inverted so rank 1 is at top)
        
        for level in difficulty_levels:
            for m, rank, acc in rankings[level]:
                if m == model:
                    x_vals.append(level)
                    y_vals.append(rank)
                    break
        
        ax.plot(x_vals, y_vals, 'o-', color=colors[i], linewidth=2.5, 
                markersize=8, label=display_name, zorder=3)
        
        # Add model name at the end
        if y_vals:
            ax.annotate(display_name, (x_vals[-1] + 0.15, y_vals[-1]), 
                       fontsize=7, va='center', color=colors[i])
    
    # Styling
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Rank')
    ax.set_xticks(difficulty_levels)
    ax.set_yticks(range(1, n_models + 1))
    ax.set_ylim(n_models + 0.5, 0.5)  # Invert so rank 1 is at top
    ax.set_xlim(min(difficulty_levels) - 0.3, max(difficulty_levels) + 1.5)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Title
    variant_name = {'gym': 'Spatial Gym', 'traceback': 'Spatial Gym Traceback', 'sparc': 'SPaRC'}
    ax.set_title(f'Model Ranking by Difficulty ({variant_name.get(variant, variant)})', 
                 fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    return fig, ax


def create_combined_bump_chart(results_dir, output_path=None):
    """Create side-by-side bump charts for Spatial Gym and Traceback."""
    setup_plot_style(use_latex=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 3.5))
    
    variants = ['gym', 'traceback']
    variant_names = ['Spatial Gym', 'Spatial Gym Traceback']
    
    for ax, variant, vname in zip(axes, variants, variant_names):
        print(f"\nCalculating for {vname}...")
        accuracy = calculate_accuracy_by_difficulty(results_dir, variant)
        
        if not accuracy:
            continue
        
        rankings, difficulty_levels = get_rankings_by_difficulty(accuracy)
        models = list(accuracy.keys())
        n_models = len(models)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        for i, model in enumerate(models):
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            
            x_vals = []
            y_vals = []
            
            for level in difficulty_levels:
                for m, rank, acc in rankings[level]:
                    if m == model:
                        x_vals.append(level)
                        y_vals.append(rank)
                        break
            
            ax.plot(x_vals, y_vals, 'o-', color=colors[i], linewidth=2, 
                    markersize=6, label=display_name, zorder=3)
            
            if y_vals:
                ax.annotate(display_name, (x_vals[-1] + 0.1, y_vals[-1]), 
                           fontsize=6, va='center', color=colors[i])
        
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Rank')
        ax.set_xticks(difficulty_levels)
        ax.set_yticks(range(1, n_models + 1))
        ax.set_ylim(n_models + 0.5, 0.5)
        ax.set_xlim(min(difficulty_levels) - 0.2, max(difficulty_levels) + 1.2)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_title(vname, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    return fig, axes


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    print("=" * 60)
    print("Creating bump chart for Spatial Gym...")
    print("=" * 60)
    
    output_pdf = Path(__file__).parent / "model_ranking_bump.pdf"
    create_bump_chart(results_dir, output_pdf, variant='gym')
    
    print("\n" + "=" * 60)
    print("Creating combined bump chart...")
    print("=" * 60)
    
    output_combined = Path(__file__).parent / "model_ranking_bump_combined.pdf"
    create_combined_bump_chart(results_dir, output_combined)


if __name__ == "__main__":
    main()
