"""
Script to create a line plot showing average accuracy by difficulty level
for SPARC, SPARC-Gym, and SPARC-Gym Traceback variants.
Shows mean with standard deviation bands.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
    LOGOBLAU,
    LOGOMITTELBLAU,
    LOGOHELLBLAU,
)

# Define colors for the three variants
VARIANT_COLORS = {
    "SPARC": "#153268",           # Dark blue (UNIBLAU)
    "SPARC-Gym": "#0091c8",       # Medium blue (LOGOMITTELBLAU)
    "SPARC-Gym Traceback": "#50a5d2",  # Light blue (LOGOHELLBLAU)
}

VARIANT_MARKERS = {
    "SPARC": "o",
    "SPARC-Gym": "s",
    "SPARC-Gym Traceback": "^",
}


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


def categorize_stats_files(results_dir):
    """Categorize stats files into SPARC, SPARC-Gym, and SPARC-Gym Traceback."""
    results_path = Path(results_dir)
    
    sparc_files = []
    gym_files = []
    traceback_files = []
    
    for stats_file in results_path.glob("*_stats.csv"):
        # Skip archive folder
        if "archive" in str(stats_file):
            continue
        # Skip visual files
        if "visual" in stats_file.name:
            continue
            
        filename = stats_file.name
        
        if "_gym_traceback_stats.csv" in filename:
            traceback_files.append(stats_file)
        elif "_gym_stats.csv" in filename:
            gym_files.append(stats_file)
        elif "_stats.csv" in filename and "_gym_" not in filename:
            sparc_files.append(stats_file)
    
    return {
        "SPARC": sparc_files,
        "SPARC-Gym": gym_files,
        "SPARC-Gym Traceback": traceback_files,
    }


def calculate_mean_std_by_difficulty(files):
    """Calculate mean and std accuracy for each difficulty level across all files."""
    all_data = {i: [] for i in range(1, 6)}
    
    for f in files:
        diff_acc = extract_difficulty_accuracies(f)
        for level, acc in diff_acc.items():
            all_data[level].append(acc)
    
    means = {}
    stds = {}
    for level in range(1, 6):
        if all_data[level]:
            means[level] = np.mean(all_data[level])
            stds[level] = np.std(all_data[level])
        else:
            means[level] = 0
            stds[level] = 0
    
    return means, stds


def create_difficulty_comparison_plot(categorized_files, output_path=None):
    """Create the line plot comparing variants by difficulty."""
    setup_plot_style(use_latex=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, 2.8))
    
    difficulties = np.array([1, 2, 3, 4, 5])
    
    for variant_name, files in categorized_files.items():
        if not files:
            continue
            
        means, stds = calculate_mean_std_by_difficulty(files)
        
        mean_values = np.array([means[i] for i in difficulties])
        std_values = np.array([stds[i] for i in difficulties])
        
        color = VARIANT_COLORS[variant_name]
        marker = VARIANT_MARKERS[variant_name]
        
        # Plot line with markers
        ax.plot(difficulties, mean_values, 
                color=color, marker=marker, markersize=5,
                linewidth=1.5, label=variant_name)
        
        # Add shaded band for standard deviation
        ax.fill_between(difficulties, 
                        mean_values - std_values, 
                        mean_values + std_values,
                        color=color, alpha=0.2)
    
    # Customize axes
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Accuracy (\\%)')
    ax.set_xticks(difficulties)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, None)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.close(fig)
    
    return fig, ax


def main():
    # Define paths
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "difficulty_comparison.pdf"
    output_png = Path(__file__).parent / "difficulty_comparison.png"
    
    # Categorize files
    print("Categorizing stats files...")
    categorized_files = categorize_stats_files(results_dir)
    
    # Print summary
    print("\nFiles found:")
    for variant, files in categorized_files.items():
        print(f"  {variant}: {len(files)} models")
        for f in files:
            print(f"    - {f.name}")
    
    # Calculate and print statistics
    print("\nMean accuracy by difficulty level:")
    print("-" * 70)
    print(f"{'Variant':<25} {'D1':>8} {'D2':>8} {'D3':>8} {'D4':>8} {'D5':>8}")
    print("-" * 70)
    
    for variant_name, files in categorized_files.items():
        if files:
            means, stds = calculate_mean_std_by_difficulty(files)
            print(f"{variant_name:<25} {means[1]:>7.1f}% {means[2]:>7.1f}% {means[3]:>7.1f}% {means[4]:>7.1f}% {means[5]:>7.1f}%")
    print("-" * 70)
    
    # Create chart
    print("\nCreating line plot...")
    create_difficulty_comparison_plot(categorized_files, output_pdf)
    create_difficulty_comparison_plot(categorized_files, output_png)


if __name__ == "__main__":
    main()
