"""
Model agreement matrix - Heatmap showing which model pairs solve the same puzzles.
Reveals if models cluster (similar behavior) or are complementary (solve different puzzles).
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
    'openai_gpt-oss-120b': 'GPT-OSS',
    'deepseek-ai_DeepSeek-R1-Distill-Qwen-32B': 'R1-Distill',
    'allenai_Olmo-3.1-32B-Think': 'OLMo-3.1',
    'google_gemma-3-27b-it': 'Gemma-3',
    'nvidia_Llama-3_3-Nemotron-Super-49B-v1_5': 'Nemotron',
    'Qwen_Qwen3-32B': 'Qwen3-32B',
    'Qwen_Qwen3-14B': 'Qwen3-14B',
    'Qwen_Qwen3-8B': 'Qwen3-8B',
    'Qwen_Qwen3-4B': 'Qwen3-4B',
    'Qwen_Qwen3-1.7B': 'Qwen3-1.7B',
    'Qwen_Qwen3-0.6B': 'Qwen3-0.6B',
    'mistralai_Magistral-Small-2507': 'Magistral',
    '9Tobi_ragen_sparc_qwen3_4B_CW3': 'RAGEN',
}

# Models to include (matching accuracy plot)
MODELS_TO_INCLUDE = {
    'openai_gpt-oss-120b',
    'allenai_Olmo-3.1-32B-Think',
    'nvidia_Llama-3_3-Nemotron-Super-49B-v1_5',
    'Qwen_Qwen3-32B',
    'deepseek-ai_DeepSeek-R1-Distill-Qwen-32B',
    'google_gemma-3-27b-it',
    'Qwen_Qwen3-0.6B',
    'mistralai_Magistral-Small-2507',
}


def load_jsonl_data(filepath):
    """Load all entries from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_model_solved_puzzles(results_dir, variant='gym'):
    """Get set of solved puzzle IDs for each model."""
    results_path = Path(results_dir)
    
    model_solved = {}  # model -> set of puzzle_ids
    
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
        
        # Filter to only include models from the accuracy plot
        if model_name not in MODELS_TO_INCLUDE:
            continue
        
        data = load_jsonl_data(jsonl_file)
        
        solved_puzzles = set()
        for entry in data:
            puzzle_id = entry.get('id')
            result = entry.get('result', {})
            solved = result.get('solved', False)
            
            if puzzle_id and solved:
                solved_puzzles.add(puzzle_id)
        
        model_solved[model_name] = solved_puzzles
    
    return model_solved


def calculate_agreement_matrix(model_solved):
    """Calculate agreement (Jaccard similarity) between all model pairs."""
    # Sort models by number of puzzles solved (descending) - proxy for performance
    models = sorted(model_solved.keys(), key=lambda m: len(model_solved[m]), reverse=True)
    n = len(models)
    
    # Jaccard similarity matrix
    jaccard_matrix = np.zeros((n, n))
    # Overlap count matrix
    overlap_matrix = np.zeros((n, n))
    # Conditional probability: P(model_j solves | model_i solves)
    conditional_matrix = np.zeros((n, n))
    
    for i, model_i in enumerate(models):
        solved_i = model_solved[model_i]
        for j, model_j in enumerate(models):
            solved_j = model_solved[model_j]
            
            intersection = len(solved_i & solved_j)
            union = len(solved_i | solved_j)
            
            overlap_matrix[i, j] = intersection
            
            if union > 0:
                jaccard_matrix[i, j] = intersection / union
            
            if len(solved_i) > 0:
                conditional_matrix[i, j] = intersection / len(solved_i)
    
    return models, jaccard_matrix, overlap_matrix, conditional_matrix


def create_agreement_heatmap(results_dir, output_path=None, variant='gym'):
    """Create a heatmap showing model agreement on puzzles."""
    setup_plot_style(use_latex=True)
    
    print(f"Calculating model agreement ({variant})...")
    model_solved = get_model_solved_puzzles(results_dir, variant)
    
    if len(model_solved) < 2:
        print("Not enough models found!")
        return None, None
    
    models, jaccard, overlap, conditional = calculate_agreement_matrix(model_solved)
    
    # Get display names
    display_names = [MODEL_DISPLAY_NAMES.get(m, m[:10]) for m in models]
    
    # Print statistics
    print(f"\nModels analyzed: {len(models)}")
    for m in models:
        print(f"  {MODEL_DISPLAY_NAMES.get(m, m)}: {len(model_solved[m])} puzzles solved")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 3.8))
    
    # Plot 1: Jaccard similarity (symmetric)
    ax1 = axes[0]
    im1 = ax1.imshow(jaccard, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    ax1.set_xticks(np.arange(len(display_names)))
    ax1.set_yticks(np.arange(len(display_names)))
    ax1.set_xticklabels(display_names, fontsize=7, rotation=45, ha='right')
    ax1.set_yticklabels(display_names, fontsize=7)
    
    # Add values
    for i in range(len(models)):
        for j in range(len(models)):
            val = jaccard[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color=color, fontsize=5)
    
    ax1.set_title('Jaccard Similarity', fontsize=9, fontweight='bold')
    
    # Colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=7)
    
    # Plot 2: Conditional probability P(j solves | i solves)
    ax2 = axes[1]
    im2 = ax2.imshow(conditional, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    
    ax2.set_xticks(np.arange(len(display_names)))
    ax2.set_yticks(np.arange(len(display_names)))
    ax2.set_xticklabels(display_names, fontsize=7, rotation=45, ha='right')
    ax2.set_yticklabels(display_names, fontsize=7)
    
    # Add values
    for i in range(len(models)):
        for j in range(len(models)):
            val = conditional[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color=color, fontsize=5)
    
    ax2.set_title('P(col solves $|$ row solves)', fontsize=9, fontweight='bold')
    
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    
    # Print interesting findings
    print("\n" + "=" * 50)
    print("Key findings:")
    print("=" * 50)
    
    # Find most similar pairs (excluding diagonal)
    most_similar = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            most_similar.append((models[i], models[j], jaccard[i, j]))
    most_similar.sort(key=lambda x: -x[2])
    
    print("\nMost similar model pairs (Jaccard):")
    for m1, m2, sim in most_similar[:5]:
        n1 = MODEL_DISPLAY_NAMES.get(m1, m1)
        n2 = MODEL_DISPLAY_NAMES.get(m2, m2)
        print(f"  {n1} <-> {n2}: {sim:.3f}")
    
    # Find most complementary pairs (low overlap)
    print("\nMost complementary pairs (lowest Jaccard):")
    for m1, m2, sim in most_similar[-5:]:
        n1 = MODEL_DISPLAY_NAMES.get(m1, m1)
        n2 = MODEL_DISPLAY_NAMES.get(m2, m2)
        # Calculate unique puzzles each solves
        unique_1 = len(model_solved[m1] - model_solved[m2])
        unique_2 = len(model_solved[m2] - model_solved[m1])
        print(f"  {n1} <-> {n2}: {sim:.3f} (unique: {unique_1}/{unique_2})")
    
    return fig, (jaccard, overlap, conditional)


def create_unique_solves_chart(results_dir, output_path=None, variant='gym'):
    """Create a bar chart showing puzzles each model uniquely solves."""
    setup_plot_style(use_latex=True)
    
    model_solved = get_model_solved_puzzles(results_dir, variant)
    models = sorted(model_solved.keys())
    
    # Calculate unique solves for each model
    all_solved = set()
    for m in models:
        all_solved.update(model_solved[m])
    
    unique_counts = {}
    for model in models:
        # Puzzles solved by this model but no other
        others_solved = set()
        for m in models:
            if m != model:
                others_solved.update(model_solved[m])
        
        unique = model_solved[model] - others_solved
        unique_counts[model] = len(unique)
    
    # Sort by unique count
    sorted_models = sorted(models, key=lambda m: -unique_counts[m])
    
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, 2.5))
    
    display_names = [MODEL_DISPLAY_NAMES.get(m, m[:10]) for m in sorted_models]
    counts = [unique_counts[m] for m in sorted_models]
    total_solved = [len(model_solved[m]) for m in sorted_models]
    
    x = np.arange(len(sorted_models))
    width = 0.4
    
    bars1 = ax.bar(x - width/2, total_solved, width, label='Total Solved', color='#1976D2', alpha=0.8)
    bars2 = ax.bar(x + width/2, counts, width, label='Uniquely Solved', color='#E65100', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Puzzles', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title('Total vs Uniquely Solved Puzzles', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    
    print("\nUniquely solved puzzles:")
    for m in sorted_models:
        n = MODEL_DISPLAY_NAMES.get(m, m)
        print(f"  {n}: {unique_counts[m]} unique / {len(model_solved[m])} total")
    
    return fig, ax


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    print("=" * 60)
    print("Creating model agreement matrix for SPARC-Gym...")
    print("=" * 60)
    
    output_pdf = Path(__file__).parent / "model_agreement.pdf"
    create_agreement_heatmap(results_dir, output_pdf, variant='gym')
    
    print("\n" + "=" * 60)
    print("Creating unique solves chart...")
    print("=" * 60)
    
    output_unique = Path(__file__).parent / "model_unique_solves.pdf"
    create_unique_solves_chart(results_dir, output_unique, variant='gym')


if __name__ == "__main__":
    main()
