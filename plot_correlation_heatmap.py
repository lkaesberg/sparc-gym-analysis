"""
Script to create a correlation heatmap between difficulty, steps taken, path length, and success.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

from plot_config import (
    setup_plot_style,
    COLUMN_WIDTH_INCHES,
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


def extract_all_data(results_dir):
    """Extract all relevant metrics from SPARC-Gym Traceback files."""
    results_path = Path(results_dir)
    
    records = []
    
    for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
        if "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem.replace("_gym_traceback", "")
        data = load_jsonl_data(jsonl_file)
        
        for entry in data:
            # Data structure: puzzle info at top level, result nested under 'result'
            result = entry.get('result', {})
            
            difficulty = entry.get('difficulty_score')
            steps_taken = result.get('steps_taken')
            extracted_path = result.get('extracted_path', [])
            solved = result.get('solved', False)  # 'solved' not 'correct'
            
            if difficulty is not None and steps_taken is not None:
                # Clean path and get edges (moves)
                cleaned = clean_path(extracted_path)
                path_edges = max(0, len(cleaned) - 1) if cleaned else 0
                
                # Calculate efficiency (avoid div by zero)
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


def create_correlation_heatmap(results_dir, output_path=None):
    """Create a correlation heatmap."""
    setup_plot_style(use_latex=True)
    
    print("Extracting data...")
    df = extract_all_data(results_dir)
    print(f"Total records: {len(df)}")
    
    # Select columns for correlation
    cols = ['difficulty', 'steps_taken', 'path_length', 'success', 'efficiency']
    labels = ['Difficulty', 'Steps Taken', 'Path Length', 'Success', 'Efficiency']
    
    # Calculate correlation matrix
    corr_matrix = df[cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.9))
    
    # Create heatmap
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=9)
    
    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=8)
    
    # Add correlation values as text
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=8, fontweight='bold')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    # Print correlation insights
    print("\nKey correlations:")
    print(f"  Difficulty vs Success: {corr_matrix.loc['difficulty', 'success']:.3f}")
    print(f"  Difficulty vs Steps: {corr_matrix.loc['difficulty', 'steps_taken']:.3f}")
    print(f"  Steps vs Success: {corr_matrix.loc['steps_taken', 'success']:.3f}")
    print(f"  Path Length vs Steps: {corr_matrix.loc['path_length', 'steps_taken']:.3f}")
    print(f"  Efficiency vs Success: {corr_matrix.loc['efficiency', 'success']:.3f}")
    
    plt.close(fig)
    return fig, ax, corr_matrix


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    print("=" * 60)
    print("Creating correlation heatmap...")
    print("=" * 60)
    
    output_pdf = Path(__file__).parent / "correlation_heatmap.pdf"
    output_png = Path(__file__).parent / "correlation_heatmap.png"
    
    create_correlation_heatmap(results_dir, output_pdf)
    create_correlation_heatmap(results_dir, output_png)


if __name__ == "__main__":
    main()
