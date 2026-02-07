"""
Script to compare actual path length vs steps taken for SPARC-Gym Traceback.
This shows the efficiency of traceback - how many steps are taken to produce a path of a certain length.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
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
    """Extract steps_taken and cleaned path length from SPARC-Gym Traceback files.
    
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


def create_comparison_plot(results_dir, output_path=None, filter_max_steps=True):
    """Create a scatter plot comparing steps vs path edges (path_length - 1)."""
    setup_plot_style(use_latex=True)
    
    print("Extracting traceback data...")
    steps, path_edges = extract_traceback_steps_vs_path(results_dir)
    
    # Filter out entries where steps hit the 100 limit (incomplete)
    if filter_max_steps:
        mask = steps < 100
        steps_filtered = steps[mask]
        path_edges_filtered = path_edges[mask]
        print(f"Data points: {len(steps)} total, {len(steps_filtered)} after filtering (removed {len(steps) - len(steps_filtered)} at 100-step limit)")
    else:
        steps_filtered = steps
        path_edges_filtered = path_edges
        print(f"Data points: {len(steps)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, 2.8))
    
    # Scatter plot with better visibility
    ax.scatter(path_edges_filtered, steps_filtered, alpha=0.4, s=12, 
               color='#7B1FA2', edgecolors='none', rasterized=True)
    
    # Add diagonal line (steps = path_edges, i.e., perfect efficiency)
    max_path = path_edges_filtered.max()
    max_steps = steps_filtered.max()
    diag_max = max(max_path, max_steps) * 1.05
    ax.plot([0, diag_max], [0, diag_max], color='#666666', linestyle='--', 
            linewidth=1.5, label='No backtracking', zorder=1)
    
    # Calculate and plot trend line (only within data range)
    z = np.polyfit(path_edges_filtered, steps_filtered, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max_path, 100)
    ax.plot(x_line, p(x_line), color='#E65100', linewidth=2.5, 
            label=f'Linear fit (slope={z[0]:.2f})', zorder=2)
    
    # Statistics
    valid_mask = steps_filtered > 0
    efficiency = path_edges_filtered[valid_mask] / steps_filtered[valid_mask]
    print(f"\nStatistics (filtered):")
    print(f"  Mean steps: {steps_filtered.mean():.1f}")
    print(f"  Mean path edges: {path_edges_filtered.mean():.1f}")
    print(f"  Mean efficiency (edges/steps): {efficiency.mean():.3f}")
    print(f"  Median efficiency: {np.median(efficiency):.3f}")
    print(f"  Slope interpretation: ~{z[0]:.1f} steps per path edge")
    
    ax.set_xlabel('Final Path Edges (moves)')
    ax.set_ylabel('Total Steps Taken')
    ax.set_xlim(0, max_path * 1.05)
    ax.set_ylim(0, max_steps * 1.05)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    
    plt.close(fig)
    return fig, ax


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
