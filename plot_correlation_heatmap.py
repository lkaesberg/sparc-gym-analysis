"""
Script to create correlation heatmaps for SPARC, SPARC-Gym, and Traceback variants.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
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
    """Remove loops/detours from a path."""
    if not path:
        return []
    
    path_tuples = []
    for p in path:
        if isinstance(p, dict):
            path_tuples.append((p.get('x'), p.get('y')))
        else:
            path_tuples.append(tuple(p))
    
    cleaned = []
    for coord in path_tuples:
        if coord in cleaned:
            idx = cleaned.index(coord)
            cleaned = cleaned[:idx]
        cleaned.append(coord)
    
    return cleaned


def extract_data_for_variant(results_dir, variant):
    """Extract metrics from files for a specific variant."""
    results_path = Path(results_dir)
    
    records = []
    
    if variant == 'sparc':
        pattern = "*.jsonl"
        suffix_to_remove = ""
    elif variant == 'gym':
        pattern = "*_gym.jsonl"
        suffix_to_remove = "_gym"
    else:  # traceback
        pattern = "*_gym_traceback.jsonl"
        suffix_to_remove = "_gym_traceback"
    
    for jsonl_file in results_path.glob(pattern):
        # Skip files that don't match the exact variant
        if variant == 'sparc' and "_gym" in jsonl_file.name:
            continue
        if variant == 'gym' and "traceback" in jsonl_file.name:
            continue
        if "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem
        if suffix_to_remove:
            model_name = model_name.replace(suffix_to_remove, "")
        
        data = load_jsonl_data(jsonl_file)
        
        for entry in data:
            result = entry.get('result', {})
            
            difficulty = entry.get('difficulty_score')
            extracted_path = result.get('extracted_path', [])
            solved = result.get('solved', False)
            
            # For SPARC (single-turn), steps_taken is 1
            # For gym/traceback, get actual steps_taken
            if variant == 'sparc':
                steps_taken = 1
            else:
                steps_taken = result.get('steps_taken')
            
            if difficulty is not None and steps_taken is not None:
                cleaned = clean_path(extracted_path)
                path_edges = max(0, len(cleaned) - 1) if cleaned else 0
                efficiency = path_edges / steps_taken if steps_taken > 0 else 0
                
                records.append({
                    'model': model_name,
                    'difficulty': difficulty,
                    'steps_taken': steps_taken,
                    'path_length': path_edges,
                    'success': 1 if solved else 0,
                    'efficiency': efficiency,
                })
    
    return pd.DataFrame(records)


def get_models_with_all_variants(results_dir):
    """Get list of models that have all 3 variants."""
    results_path = Path(results_dir)
    
    sparc_models = set()
    gym_models = set()
    traceback_models = set()
    
    for f in results_path.glob("*.jsonl"):
        if "_gym" not in f.name and "visual" not in f.name and "archive" not in str(f):
            sparc_models.add(f.stem)
    
    for f in results_path.glob("*_gym.jsonl"):
        if "traceback" not in f.name and "visual" not in f.name and "archive" not in str(f):
            gym_models.add(f.stem.replace("_gym", ""))
    
    for f in results_path.glob("*_gym_traceback.jsonl"):
        if "visual" not in f.name and "archive" not in str(f):
            traceback_models.add(f.stem.replace("_gym_traceback", ""))
    
    return sparc_models & gym_models & traceback_models


def create_correlation_heatmap(results_dir, output_path=None):
    """Create 3 correlation heatmaps as subplots for each variant."""
    setup_plot_style(use_latex=True)
    
    # Get models with all 3 variants
    complete_models = get_models_with_all_variants(results_dir)
    print(f"Models with all 3 variants: {len(complete_models)}")
    print(f"  {sorted(complete_models)}")
    
    variants = ['gym', 'traceback']
    variant_labels = ['SPARC-Gym', 'Traceback']
    
    cols = ['difficulty', 'steps_taken', 'path_length', 'success', 'efficiency']
    labels = ['Difficulty', 'Steps', 'Path Len', 'Success', 'Efficiency']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES / 2.5 + 0.3))
    
    corr_matrices = {}
    
    for idx, (variant, var_label, ax) in enumerate(zip(variants, variant_labels, axes)):
        print(f"\nExtracting {variant} data...")
        df = extract_data_for_variant(results_dir, variant)
        
        if len(df) == 0:
            print(f"  No data for {variant}")
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var_label, fontsize=10, fontweight='bold')
            continue
        
        # Filter to complete models only
        df = df[df['model'].isin(complete_models)]
        print(f"  Records for {variant}: {len(df)}")
        
        if len(df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var_label, fontsize=10, fontweight='bold')
            continue
        
        # Calculate correlation matrix
        corr_matrix = df[cols].corr()
        corr_matrices[variant] = corr_matrix
        
        # Create heatmap
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.set_yticklabels([])
        
        # Add correlation values as text
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr_matrix.values[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=6, fontweight='bold')
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_title(var_label, fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Correlation', fontsize=9)
    
    plt.subplots_adjust(left=0.1, right=0.88, wspace=0.1)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    # Print correlation insights for each variant
    for variant, var_label in zip(variants, variant_labels):
        if variant in corr_matrices:
            corr = corr_matrices[variant]
            print(f"\n{var_label} key correlations:")
            print(f"  Difficulty vs Success: {corr.loc['difficulty', 'success']:.3f}")
            print(f"  Difficulty vs Steps: {corr.loc['difficulty', 'steps_taken']:.3f}")
            print(f"  Steps vs Success: {corr.loc['steps_taken', 'success']:.3f}")
    
    plt.close(fig)
    return fig, axes, corr_matrices


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    print("=" * 60)
    print("Creating correlation heatmaps for all 3 variants...")
    print("=" * 60)
    
    output_pdf = Path(__file__).parent / "correlation_heatmap.pdf"
    
    create_correlation_heatmap(results_dir, output_pdf)


if __name__ == "__main__":
    main()
