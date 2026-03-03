"""
Radar plot showing solve rates by puzzle rule type for SPaRC, SPaRC-Gym, and SPaRC-Gym Traceback.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

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


def extract_puzzle_rules(text_viz):
    """Extract rule types from text_visualization."""
    rules = set()
    
    for line in text_viz.split('\n'):
        line_lower = line.lower().strip()
        if 'dot:' in line_lower:
            rules.add('Dot')
        elif 'gap:' in line_lower:
            rules.add('Gap')
        elif 'type: "poly"' in line_lower:
            rules.add('Polyomino')
        elif 'type: "ylop"' in line_lower:
            rules.add('Ylop')
        elif 'type: "star"' in line_lower:
            rules.add('Star')
        elif 'type: "square"' in line_lower:
            rules.add('Square')
        elif 'type: "triangle"' in line_lower:
            rules.add('Triangle')
    
    return rules


def calculate_solve_rates_by_rule(results_dir):
    """Calculate solve rates for each rule type across all variants."""
    results_path = Path(results_dir)
    
    # Track: rule -> variant -> (solved, total)
    rule_stats = defaultdict(lambda: defaultdict(lambda: {'solved': 0, 'total': 0}))
    
    # Process SPaRC (normal) - from stats files
    for stats_file in results_path.glob("*_stats.csv"):
        # Skip gym variants
        if "_gym" in stats_file.name:
            continue
        if "archive" in str(stats_file):
            continue
        
        # Need to match with jsonl to get rules
        model_name = stats_file.stem.replace("_stats", "")
        jsonl_file = results_path / f"{model_name}.jsonl"
        
        if jsonl_file.exists():
            data = load_jsonl_data(jsonl_file)
            for entry in data:
                text_viz = entry.get('text_visualization', '')
                rules = extract_puzzle_rules(text_viz)
                result = entry.get('result', {})
                solved = result.get('solved', False)  # 'solved' not 'correct'
                
                for rule in rules:
                    rule_stats[rule]['SPaRC']['total'] += 1
                    if solved:
                        rule_stats[rule]['SPaRC']['solved'] += 1
    
    # Process SPaRC-Gym
    for jsonl_file in results_path.glob("*_gym.jsonl"):
        if "traceback" in jsonl_file.name or "archive" in str(jsonl_file):
            continue
        
        data = load_jsonl_data(jsonl_file)
        for entry in data:
            text_viz = entry.get('text_visualization', '')
            rules = extract_puzzle_rules(text_viz)
            result = entry.get('result', {})
            solved = result.get('solved', False)
            
            for rule in rules:
                rule_stats[rule]['SPaRC-Gym']['total'] += 1
                if solved:
                    rule_stats[rule]['SPaRC-Gym']['solved'] += 1
    
    # Process SPaRC-Gym Traceback
    for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
        if "archive" in str(jsonl_file):
            continue
        
        data = load_jsonl_data(jsonl_file)
        for entry in data:
            text_viz = entry.get('text_visualization', '')
            rules = extract_puzzle_rules(text_viz)
            result = entry.get('result', {})
            solved = result.get('solved', False)
            
            for rule in rules:
                rule_stats[rule]['Traceback']['total'] += 1
                if solved:
                    rule_stats[rule]['Traceback']['solved'] += 1
    
    # Calculate rates
    solve_rates = {}
    for rule in rule_stats:
        solve_rates[rule] = {}
        for variant in ['SPaRC', 'SPaRC-Gym', 'Traceback']:
            stats = rule_stats[rule][variant]
            if stats['total'] > 0:
                solve_rates[rule][variant] = stats['solved'] / stats['total'] * 100
            else:
                solve_rates[rule][variant] = 0
    
    return solve_rates, rule_stats


def create_radar_plot(results_dir, output_path=None):
    """Create a radar plot of solve rates by rule type."""
    setup_plot_style(use_latex=True)
    
    print("Calculating solve rates by rule type...")
    solve_rates, rule_stats = calculate_solve_rates_by_rule(results_dir)
    
    # Print statistics
    print("\nSolve rates by rule type:")
    for rule in sorted(solve_rates.keys()):
        print(f"\n{rule}:")
        for variant in ['SPaRC', 'SPaRC-Gym', 'Traceback']:
            stats = rule_stats[rule][variant]
            rate = solve_rates[rule].get(variant, 0)
            print(f"  {variant}: {rate:.1f}% ({stats['solved']}/{stats['total']})")
    
    # Prepare data for radar plot
    # Filter rules with enough data
    rules = [r for r in solve_rates.keys() if rule_stats[r]['SPaRC-Gym']['total'] >= 100]
    rules = sorted(rules)
    
    if not rules:
        print("Not enough data for radar plot")
        return None, None
    
    # Number of variables
    N = len(rules)
    
    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES), 
                           subplot_kw=dict(polar=True))
    
    # Colors and variants
    variants = ['SPaRC', 'SPaRC-Gym', 'Traceback']
    colors = ['#1976D2', '#7B1FA2', '#E65100']
    
    for variant, color in zip(variants, colors):
        values = [solve_rates[rule].get(variant, 0) for rule in rules]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=variant, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rules, fontsize=8)
    
    # Set y-axis - adjusted for actual data range
    max_val = max([max(solve_rates[r].values()) for r in rules])
    ax.set_ylim(0, max(12, max_val * 1.15))
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2\\%', '4\\%', '6\\%', '8\\%', '10\\%'], fontsize=7)
    
    # Add legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        # Also save as PNG if saving PDF
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    return fig, ax, solve_rates, rule_stats


def calculate_solve_rates_by_rule_single_model(results_dir, model_filter):
    """Calculate solve rates for a single model across all variants."""
    results_path = Path(results_dir)
    
    rule_stats = {}
    for rule in ['Dot', 'Gap', 'Polyomino', 'Square', 'Star', 'Triangle', 'Ylop']:
        rule_stats[rule] = {
            'SPaRC': {'solved': 0, 'total': 0},
            'SPaRC-Gym': {'solved': 0, 'total': 0},
            'Traceback': {'solved': 0, 'total': 0},
        }
    
    # Process SPaRC (normal)
    for jsonl_file in results_path.glob("*.jsonl"):
        if "_gym" in jsonl_file.name or "archive" in str(jsonl_file):
            continue
        
        model_name = jsonl_file.stem
        if model_filter not in model_name:
            continue
        
        data = load_jsonl_data(jsonl_file)
        for entry in data:
            text_viz = entry.get('text_visualization', '')
            rules = extract_puzzle_rules(text_viz)
            result = entry.get('result', {})
            solved = result.get('solved', False)
            
            for rule in rules:
                if rule in rule_stats:
                    rule_stats[rule]['SPaRC']['total'] += 1
                    if solved:
                        rule_stats[rule]['SPaRC']['solved'] += 1
    
    # Process SPaRC-Gym
    for jsonl_file in results_path.glob("*_gym.jsonl"):
        if "traceback" in jsonl_file.name or "archive" in str(jsonl_file):
            continue
        
        model_name = jsonl_file.stem.replace("_gym", "")
        if model_filter not in model_name:
            continue
        
        data = load_jsonl_data(jsonl_file)
        for entry in data:
            text_viz = entry.get('text_visualization', '')
            rules = extract_puzzle_rules(text_viz)
            result = entry.get('result', {})
            solved = result.get('solved', False)
            
            for rule in rules:
                if rule in rule_stats:
                    rule_stats[rule]['SPaRC-Gym']['total'] += 1
                    if solved:
                        rule_stats[rule]['SPaRC-Gym']['solved'] += 1
    
    # Process SPaRC-Gym Traceback
    for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
        if "archive" in str(jsonl_file):
            continue
        
        model_name = jsonl_file.stem.replace("_gym_traceback", "")
        if model_filter not in model_name:
            continue
        
        data = load_jsonl_data(jsonl_file)
        for entry in data:
            text_viz = entry.get('text_visualization', '')
            rules = extract_puzzle_rules(text_viz)
            result = entry.get('result', {})
            solved = result.get('solved', False)
            
            for rule in rules:
                if rule in rule_stats:
                    rule_stats[rule]['Traceback']['total'] += 1
                    if solved:
                        rule_stats[rule]['Traceback']['solved'] += 1
    
    # Calculate rates
    solve_rates = {}
    for rule in rule_stats:
        solve_rates[rule] = {}
        for variant in ['SPaRC', 'SPaRC-Gym', 'Traceback']:
            stats = rule_stats[rule][variant]
            if stats['total'] > 0:
                solve_rates[rule][variant] = stats['solved'] / stats['total'] * 100
            else:
                solve_rates[rule][variant] = 0
    
    return solve_rates, rule_stats


def create_radar_plot_single_model(results_dir, model_filter, model_display_name, output_path=None):
    """Create a radar plot for a single model."""
    setup_plot_style(use_latex=True)
    
    print(f"Calculating solve rates for {model_display_name}...")
    solve_rates, rule_stats = calculate_solve_rates_by_rule_single_model(results_dir, model_filter)
    
    # Print statistics
    print(f"\nSolve rates by rule type for {model_display_name}:")
    for rule in sorted(solve_rates.keys()):
        print(f"\n{rule}:")
        for variant in ['SPaRC', 'SPaRC-Gym', 'Traceback']:
            stats = rule_stats[rule][variant]
            rate = solve_rates[rule].get(variant, 0)
            print(f"  {variant}: {rate:.1f}% ({stats['solved']}/{stats['total']})")
    
    # Filter rules with data
    rules = [r for r in solve_rates.keys() if rule_stats[r]['SPaRC-Gym']['total'] > 0]
    rules = sorted(rules)
    
    if not rules:
        print(f"Not enough data for {model_display_name}")
        return None, None
    
    N = len(rules)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES), 
                           subplot_kw=dict(polar=True))
    
    variants = ['SPaRC', 'SPaRC-Gym', 'Traceback']
    colors = ['#1976D2', '#7B1FA2', '#E65100']
    
    for variant, color in zip(variants, colors):
        values = [solve_rates[rule].get(variant, 0) for rule in rules]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=variant, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rules, fontsize=8)
    
    max_val = max([max(solve_rates[r].values()) for r in rules if solve_rates[r].values()])
    ax.set_ylim(0, max(12, max_val * 1.15))
    ax.set_yticks([5, 10, 15, 20])
    ax.set_yticklabels(['5\\%', '10\\%', '15\\%', '20\\%'], fontsize=7)
    
    ax.set_title(model_display_name, fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=True, framealpha=0.9)
    
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


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    print("=" * 60)
    print("Creating radar plot by puzzle rule type (all models)...")
    print("=" * 60)
    
    output_pdf = Path(__file__).parent / "solve_rate_by_rule.pdf"
    create_radar_plot(results_dir, output_pdf)
    
    print("\n" + "=" * 60)
    print("Creating radar plot for GPT-OSS-120B...")
    print("=" * 60)
    
    output_pdf_gpt = Path(__file__).parent / "solve_rate_by_rule_gpt_oss.pdf"
    create_radar_plot_single_model(results_dir, "gpt-oss-120b", "GPT-OSS-120B", output_pdf_gpt)


if __name__ == "__main__":
    main()
