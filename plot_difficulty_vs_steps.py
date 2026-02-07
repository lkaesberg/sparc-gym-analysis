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

# Colors for each variant
VARIANT_COLORS = {
    "True Solution": "#2E7D32",      # Dark green
    "SPARC": "#153268",              # Dark blue (UNIBLAU)
    "SPARC-Gym": "#0091c8",          # Medium blue
    "SPARC-Gym Traceback": "#50a5d2", # Light blue
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
    """Extract difficulty_score and true solution path length from any JSONL file."""
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
            # Take the first solution's path length
            path_length = solutions[0].get('pathLength')
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
        ax.set_ylabel('Steps / Path Length')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)


def create_difficulty_steps_plot(results_dir, output_path=None):
    """Create the 4-subplot figure."""
    setup_plot_style(use_latex=True)
    
    # Categorize files
    categorized_files = categorize_jsonl_files(results_dir)
    
    # Get all available files for true solution extraction
    all_files = categorized_files["SPARC"] + categorized_files["SPARC-Gym"] + categorized_files["SPARC-Gym Traceback"]
    
    # Extract data for each variant
    print("Extracting true solution data...")
    true_diff, true_steps = extract_true_solution_data(all_files)
    
    print("Extracting SPARC data...")
    sparc_diff, sparc_steps = extract_sparc_data(categorized_files["SPARC"])
    
    print("Extracting SPARC-Gym data...")
    gym_diff, gym_steps = extract_gym_data(categorized_files["SPARC-Gym"])
    
    print("Extracting SPARC-Gym Traceback data...")
    traceback_diff, traceback_steps = extract_gym_data(categorized_files["SPARC-Gym Traceback"])
    
    # Print statistics
    print(f"\nData points: True={len(true_diff)}, SPARC={len(sparc_diff)}, Gym={len(gym_diff)}, Traceback={len(traceback_diff)}")
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_INCHES, 2.5), sharey=True)
    
    # Create each subplot
    create_subplot(axes[0], true_diff, true_steps, "(a) True Solution", 
                   VARIANT_COLORS["True Solution"], ylabel=True)
    create_subplot(axes[1], sparc_diff, sparc_steps, "(b) SPARC", 
                   VARIANT_COLORS["SPARC"], ylabel=False)
    create_subplot(axes[2], gym_diff, gym_steps, "(c) SPARC-Gym", 
                   VARIANT_COLORS["SPARC-Gym"], ylabel=False)
    create_subplot(axes[3], traceback_diff, traceback_steps, "(d) SPARC-Gym Traceback", 
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
    output_pdf = Path(__file__).parent / "difficulty_vs_steps.pdf"
    output_png = Path(__file__).parent / "difficulty_vs_steps.png"
    
    print("Creating difficulty vs steps plot...")
    create_difficulty_steps_plot(results_dir, output_pdf)
    create_difficulty_steps_plot(results_dir, output_png)


if __name__ == "__main__":
    main()
