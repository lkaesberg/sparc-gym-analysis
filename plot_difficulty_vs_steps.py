"""
Script to create a 4-subplot figure showing difficulty_score vs steps taken for:
1. True solution (ground truth path length)
2. SPARC (extracted path length)
3. SPARC-Gym (steps_taken)
4. SPARC-Gym Traceback (steps_taken)
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    LOGOBLAU,
    LOGOMITTELBLAU,
    LOGOHELLBLAU,
)

# Colors for each variant - distinct and visually appealing
VARIANT_COLORS = {
    "True Solution": "#2E7D32",      # Forest green
    "SPARC": "#E65100",              # Deep orange
    "SPARC-Gym": "#1565C0",          # Strong blue
    "SPARC-Gym Traceback": "#7B1FA2", # Purple
}


def load_jsonl_data(filepath):
    """Load all entries from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def categorize_jsonl_files(results_dir):
    """Categorize JSONL files into SPARC, SPARC-Gym, and SPARC-Gym Traceback."""
    results_path = Path(results_dir)
    
    sparc_files = []
    gym_files = []
    traceback_files = []
    
    for jsonl_file in results_path.glob("*.jsonl"):
        if "archive" in str(jsonl_file):
            continue
        if "visual" in jsonl_file.name:
            continue
            
        filename = jsonl_file.name
        
        if "_gym_traceback.jsonl" in filename:
            traceback_files.append(jsonl_file)
        elif "_gym.jsonl" in filename:
            gym_files.append(jsonl_file)
        elif ".jsonl" in filename and "_gym" not in filename:
            sparc_files.append(jsonl_file)
    
    return {
        "SPARC": sparc_files,
        "SPARC-Gym": gym_files,
        "SPARC-Gym Traceback": traceback_files,
    }


def extract_true_solution_data(jsonl_files):
    """Extract difficulty_score and true solution path length from any JSONL file.
    
    Includes ALL solutions for each puzzle (not just the first one).
    """
    # Use first available file to get ground truth (all files have the same puzzles)
    difficulty_scores = []
    path_lengths = []
    
    if not jsonl_files:
        return np.array([]), np.array([])
    
    # Use the first file
    data = load_jsonl_data(jsonl_files[0])
    
    for entry in data:
        difficulty_score = entry.get('difficulty_score')
        solutions = entry.get('solutions', [])
        
        if difficulty_score is not None and solutions:
            # Include ALL solutions, not just the first one
            for solution in solutions:
                path_length = solution.get('pathLength')
                if path_length is not None:
                    difficulty_scores.append(difficulty_score)
                    path_lengths.append(path_length)
    
    return np.array(difficulty_scores), np.array(path_lengths)


def extract_sparc_data(jsonl_files):
    """Extract difficulty_score and extracted_path length from SPARC JSONL files."""
    all_difficulty_scores = []
    all_path_lengths = []
    
    for filepath in jsonl_files:
        data = load_jsonl_data(filepath)
        
        for entry in data:
            difficulty_score = entry.get('difficulty_score')
            result = entry.get('result', {})
            extracted_path = result.get('extracted_path', [])
            
            if difficulty_score is not None and extracted_path:
                all_difficulty_scores.append(difficulty_score)
                all_path_lengths.append(len(extracted_path))
    
    return np.array(all_difficulty_scores), np.array(all_path_lengths)


def extract_gym_data(jsonl_files):
    """Extract difficulty_score and steps_taken from SPARC-Gym JSONL files."""
    all_difficulty_scores = []
    all_steps = []
    
    for filepath in jsonl_files:
        data = load_jsonl_data(filepath)
        
        for entry in data:
            difficulty_score = entry.get('difficulty_score')
            result = entry.get('result', {})
            steps_taken = result.get('steps_taken')
            
            if difficulty_score is not None and steps_taken is not None:
                all_difficulty_scores.append(difficulty_score)
                all_steps.append(steps_taken)
    
    return np.array(all_difficulty_scores), np.array(all_steps)


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


def extract_traceback_data(jsonl_files):
    """Extract difficulty_score and cleaned path length from SPARC-Gym Traceback JSONL files."""
    all_difficulty_scores = []
    all_path_lengths = []
    
    for filepath in jsonl_files:
        data = load_jsonl_data(filepath)
        
        for entry in data:
            difficulty_score = entry.get('difficulty_score')
            result = entry.get('result', {})
            extracted_path = result.get('extracted_path', [])
            
            if difficulty_score is not None and extracted_path:
                # Clean the path to remove backtracking detours
                cleaned = clean_path(extracted_path)
                all_difficulty_scores.append(difficulty_score)
                all_path_lengths.append(len(cleaned))
    
    return np.array(all_difficulty_scores), np.array(all_path_lengths)


def bin_data_by_difficulty(difficulty_scores, values, n_bins=20):
    """Bin data by difficulty score and compute mean and std for each bin."""
    if len(difficulty_scores) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Create bins
    bin_edges = np.linspace(difficulty_scores.min(), difficulty_scores.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Bin the data
    bin_means = []
    bin_stds = []
    valid_centers = []
    
    for i in range(n_bins):
        mask = (difficulty_scores >= bin_edges[i]) & (difficulty_scores < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (difficulty_scores >= bin_edges[i]) & (difficulty_scores <= bin_edges[i+1])
        
        if mask.sum() > 0:
            bin_means.append(np.mean(values[mask]))
            bin_stds.append(np.std(values[mask]))
            valid_centers.append(bin_centers[i])
    
    return np.array(valid_centers), np.array(bin_means), np.array(bin_stds)


def create_subplot(ax, difficulty_scores, values, title, color, ylabel=True):
    """Create a single subplot with scatter and trend line."""
    if len(difficulty_scores) == 0:
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Scatter plot with transparency
    ax.scatter(difficulty_scores, values, alpha=0.15, s=8, color=color, edgecolors='none')
    
    # Bin data and plot mean line with std band
    bin_centers, bin_means, bin_stds = bin_data_by_difficulty(difficulty_scores, values, n_bins=15)
    
    if len(bin_centers) > 0:
        ax.plot(bin_centers, bin_means, color=color, linewidth=2, label='Mean')
        ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, 
                        color=color, alpha=0.3, label='±1 Std')
    
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Difficulty Score')
    if ylabel:
        ax.set_ylabel('Path Length')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)


def filter_by_max_steps(difficulty_scores, steps, max_steps=100):
    """Filter out data points where steps >= max_steps."""
    mask = steps < max_steps
    return difficulty_scores[mask], steps[mask]


def create_difficulty_steps_plot(results_dir, output_path=None, max_steps=None):
    """Create the 4-subplot figure.
    
    Args:
        results_dir: Path to results directory
        output_path: Path to save the figure
        max_steps: If set, filter out data points with steps >= max_steps
    """
    setup_plot_style(use_latex=True)
    
    # Categorize files
    categorized_files = categorize_jsonl_files(results_dir)
    
    # True solution data is the same for all variants (same puzzles)
    # Use any available file to extract ground truth
    any_file = (categorized_files["SPARC"] + categorized_files["SPARC-Gym"] + 
                categorized_files["SPARC-Gym Traceback"])[:1]
    
    # Extract data for each variant
    print("Extracting true solution data (same for all variants)...")
    true_diff, true_steps = extract_true_solution_data(any_file)
    
    print("Extracting SPARC data...")
    sparc_diff, sparc_steps = extract_sparc_data(categorized_files["SPARC"])
    
    print("Extracting SPARC-Gym data...")
    gym_diff, gym_steps = extract_gym_data(categorized_files["SPARC-Gym"])
    
    print("Extracting SPARC-Gym Traceback data...")
    traceback_diff, traceback_steps = extract_traceback_data(categorized_files["SPARC-Gym Traceback"])
    
    # Apply filtering if max_steps is set
    if max_steps is not None:
        print(f"\nFiltering out steps >= {max_steps}...")
        true_diff, true_steps = filter_by_max_steps(true_diff, true_steps, max_steps)
        sparc_diff, sparc_steps = filter_by_max_steps(sparc_diff, sparc_steps, max_steps)
        gym_diff, gym_steps = filter_by_max_steps(gym_diff, gym_steps, max_steps)
        traceback_diff, traceback_steps = filter_by_max_steps(traceback_diff, traceback_steps, max_steps)
    
    # Print statistics
    print(f"\nData points: True={len(true_diff)}, SPARC={len(sparc_diff)}, Gym={len(gym_diff)}, Traceback={len(traceback_diff)}")
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_INCHES, 2.5), sharey=True)
    
    # Create each subplot
    create_subplot(axes[0], true_diff, true_steps, "(a) True Solution", 
                   VARIANT_COLORS["True Solution"], ylabel=True)
    create_subplot(axes[1], sparc_diff, sparc_steps, "(b) SPARC", 
                   VARIANT_COLORS["SPARC"], ylabel=False)
    create_subplot(axes[2], gym_diff, gym_steps, "(c) Gym w/o traceback", 
                   VARIANT_COLORS["SPARC-Gym"], ylabel=False)
    create_subplot(axes[3], traceback_diff, traceback_steps, "(d) Gym w/ traceback", 
                   VARIANT_COLORS["SPARC-Gym Traceback"], ylabel=False)
    
    # Compute shared y-limits
    all_steps = np.concatenate([
        true_steps if len(true_steps) > 0 else np.array([0]),
        sparc_steps if len(sparc_steps) > 0 else np.array([0]),
        gym_steps if len(gym_steps) > 0 else np.array([0]),
        traceback_steps if len(traceback_steps) > 0 else np.array([0]),
    ])
    y_max = np.percentile(all_steps, 99)  # Use 99th percentile to avoid outliers
    for ax in axes:
        ax.set_ylim(0, y_max * 1.05)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.close(fig)
    return fig, axes


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    # Version 1: All data
    print("=" * 60)
    print("Creating difficulty vs steps plot (all data)...")
    print("=" * 60)
    output_pdf = Path(__file__).parent / "difficulty_vs_steps.pdf"
    output_png = Path(__file__).parent / "difficulty_vs_steps.png"
    create_difficulty_steps_plot(results_dir, output_pdf)
    create_difficulty_steps_plot(results_dir, output_png)
    
    # Version 2: Filtered (steps < 100)
    print("\n" + "=" * 60)
    print("Creating difficulty vs steps plot (filtered < 100 steps)...")
    print("=" * 60)
    output_pdf_filtered = Path(__file__).parent / "difficulty_vs_steps_filtered.pdf"
    output_png_filtered = Path(__file__).parent / "difficulty_vs_steps_filtered.png"
    create_difficulty_steps_plot(results_dir, output_pdf_filtered, max_steps=100)
    create_difficulty_steps_plot(results_dir, output_png_filtered, max_steps=100)


if __name__ == "__main__":
    main()
