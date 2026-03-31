"""
Radar plot showing solve rates by puzzle rule type for Baseline, Gym w/o backtracking, and Gym w/ backtracking.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from plot_config import (
    setup_plot_style,
    COLUMN_WIDTH_INCHES,
    TEXT_WIDTH_INCHES,
)


def style_polar_grid(ax, yticks):
    """Draw visible concentric rings and style the polar grid."""
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linewidth=0.3, color='#bbbbbb')
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    for r in yticks:
        ax.plot(theta_ring, [r] * len(theta_ring), color='#999999', linewidth=0.5, zorder=1.5)
    # Draw one extra outer ring at the ylim boundary (no ytick label)
    outer = ax.get_ylim()[1]
    if outer > yticks[-1]:
        ax.plot(theta_ring, [outer] * len(theta_ring), color='#999999', linewidth=0.5, zorder=1.5)
    ax.spines['polar'].set_visible(False)


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
    
    # Process Baseline (normal) - from stats files
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
                    rule_stats[rule]['Baseline']['total'] += 1
                    if solved:
                        rule_stats[rule]['Baseline']['solved'] += 1
    
    # Process Gym w/o backtracking
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
                rule_stats[rule]['Gym w/o backtracking']['total'] += 1
                if solved:
                    rule_stats[rule]['Gym w/o backtracking']['solved'] += 1
    
    # Process Gym w/ backtracking
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
                rule_stats[rule]['Gym w/ backtracking']['total'] += 1
                if solved:
                    rule_stats[rule]['Gym w/ backtracking']['solved'] += 1
    
    # Calculate rates
    solve_rates = {}
    for rule in rule_stats:
        solve_rates[rule] = {}
        for variant in ['Baseline', 'Gym w/o backtracking', 'Gym w/ backtracking']:
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
        for variant in ['Baseline', 'Gym w/o backtracking', 'Gym w/ backtracking']:
            stats = rule_stats[rule][variant]
            rate = solve_rates[rule].get(variant, 0)
            print(f"  {variant}: {rate:.1f}% ({stats['solved']}/{stats['total']})")
    
    # Prepare data for radar plot
    # Filter rules with enough data
    rules = [r for r in solve_rates.keys() if rule_stats[r]['Gym w/o backtracking']['total'] >= 100]
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
    variants = ['Baseline', 'Gym w/o backtracking', 'Gym w/ backtracking']
    colors = ['#1976D2', '#7B1FA2', '#E65100']
    
    # Rotate so that Polyomino faces up
    if 'Polyomino' in rules:
        poly_idx = rules.index('Polyomino')
        ax.set_theta_offset(np.pi / 2 - angles[poly_idx])

    for variant, color in zip(variants, colors):
        values = [solve_rates[rule].get(variant, 0) for rule in rules]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=1.5,
                label=variant, color=color, markersize=4, zorder=3)
        ax.fill(angles, values, alpha=0.1, color=color, zorder=2)

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rules)

    # Set y-axis - adjusted for actual data range
    max_val = max([max(solve_rates[r].values()) for r in rules])
    y_max = max_val * 1.15
    ax.set_ylim(0, y_max)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2\\%', '4\\%', '6\\%', '8\\%', '10\\%'])
    ax.yaxis.set_tick_params(labelsize=7)
    for label in ax.yaxis.get_ticklabels():
        label.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])
        label.set_zorder(10)

    style_polar_grid(ax, [2, 4, 6, 8, 10])

    # Add legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)
    
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
            'Baseline': {'solved': 0, 'total': 0},
            'Gym w/o backtracking': {'solved': 0, 'total': 0},
            'Gym w/ backtracking': {'solved': 0, 'total': 0},
        }
    
    # Process Baseline (normal)
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
                    rule_stats[rule]['Baseline']['total'] += 1
                    if solved:
                        rule_stats[rule]['Baseline']['solved'] += 1
    
    # Process Gym w/o backtracking
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
                    rule_stats[rule]['Gym w/o backtracking']['total'] += 1
                    if solved:
                        rule_stats[rule]['Gym w/o backtracking']['solved'] += 1
    
    # Process Gym w/ backtracking
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
                    rule_stats[rule]['Gym w/ backtracking']['total'] += 1
                    if solved:
                        rule_stats[rule]['Gym w/ backtracking']['solved'] += 1
    
    # Calculate rates
    solve_rates = {}
    for rule in rule_stats:
        solve_rates[rule] = {}
        for variant in ['Baseline', 'Gym w/o backtracking', 'Gym w/ backtracking']:
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
        for variant in ['Baseline', 'Gym w/o backtracking', 'Gym w/ backtracking']:
            stats = rule_stats[rule][variant]
            rate = solve_rates[rule].get(variant, 0)
            print(f"  {variant}: {rate:.1f}% ({stats['solved']}/{stats['total']})")
    
    # Filter rules with data
    rules = [r for r in solve_rates.keys() if rule_stats[r]['Gym w/o backtracking']['total'] > 0]
    rules = sorted(rules)
    
    if not rules:
        print(f"Not enough data for {model_display_name}")
        return None, None
    
    N = len(rules)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES), 
                           subplot_kw=dict(polar=True))
    
    variants = ['Baseline', 'Gym w/o backtracking', 'Gym w/ backtracking']
    colors = ['#1976D2', '#7B1FA2', '#E65100']
    
    # Rotate so that Polyomino faces up
    if 'Polyomino' in rules:
        poly_idx = rules.index('Polyomino')
        ax.set_theta_offset(np.pi / 2 - angles[poly_idx])

    for variant, color in zip(variants, colors):
        values = [solve_rates[rule].get(variant, 0) for rule in rules]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5,
                label=variant, color=color, markersize=4, zorder=3)
        ax.fill(angles, values, alpha=0.1, color=color, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rules)

    max_val = max([max(solve_rates[r].values()) for r in rules if solve_rates[r].values()])
    ax.set_ylim(0, max_val * 1.15)
    ax.set_yticks([5, 10, 15, 20])
    ax.set_yticklabels(['5\\%', '10\\%', '15\\%', '20\\%'])
    for label in ax.yaxis.get_ticklabels():
        label.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])
        label.set_zorder(10)

    style_polar_grid(ax, [5, 10, 15, 20])

    ax.set_title(model_display_name, fontweight='bold', pad=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)
    
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


def create_combined_radar_plot(results_dir, output_path=None):
    """Create a side-by-side radar plot (all models + GPT-OSS-120B) at text width with shared legend."""
    setup_plot_style(use_latex=True)

    solve_rates_all, rule_stats_all = calculate_solve_rates_by_rule(results_dir)
    rules_all = [r for r in solve_rates_all.keys() if rule_stats_all[r]['Gym w/o backtracking']['total'] >= 100]
    rules_all = sorted(rules_all)

    solve_rates_gpt, rule_stats_gpt = calculate_solve_rates_by_rule_single_model(results_dir, "gpt-oss-120b")
    rules_gpt = [r for r in solve_rates_gpt.keys() if rule_stats_gpt[r]['Gym w/o backtracking']['total'] > 0]
    rules_gpt = sorted(rules_gpt)

    variants = ['Baseline', 'Gym w/o backtracking', 'Gym w/ backtracking']
    colors = ['#1976D2', '#7B1FA2', '#E65100']

    panel_h = TEXT_WIDTH_INCHES / 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, panel_h + 0.4),
                                    subplot_kw=dict(polar=True))

    # Panel (a): All models
    N1 = len(rules_all)
    angles1 = np.linspace(0, 2 * np.pi, N1, endpoint=False).tolist()
    angles1 += angles1[:1]

    # Rotate panel (a) so Polyomino faces up
    if 'Polyomino' in rules_all:
        poly_idx = rules_all.index('Polyomino')
        ax1.set_theta_offset(np.pi / 2 - angles1[poly_idx])

    legend_handles = []
    for variant, color in zip(variants, colors):
        values = [solve_rates_all[rule].get(variant, 0) for rule in rules_all]
        values += values[:1]
        line, = ax1.plot(angles1, values, 'o-', linewidth=1.5,
                         label=variant, color=color, markersize=4, zorder=3)
        ax1.fill(angles1, values, alpha=0.1, color=color, zorder=2)
        legend_handles.append(line)

    ax1.set_xticks(angles1[:-1])
    ax1.set_xticklabels(rules_all)
    max_val1 = max([max(solve_rates_all[r].values()) for r in rules_all])
    ax1.set_ylim(0, max_val1 * 1.15)
    ax1.set_yticks([2, 4, 6, 8, 10])
    ax1.set_yticklabels(['2\\%', '4\\%', '6\\%', '8\\%', '10\\%'])
    for label in ax1.yaxis.get_ticklabels():
        label.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])
        label.set_zorder(10)
    style_polar_grid(ax1, [2, 4, 6, 8, 10])
    ax1.set_title('(a) All Models')

    # Panel (b): GPT-OSS-120B
    N2 = len(rules_gpt)
    angles2 = np.linspace(0, 2 * np.pi, N2, endpoint=False).tolist()
    angles2 += angles2[:1]

    # Rotate panel (b) so Polyomino faces up
    if 'Polyomino' in rules_gpt:
        poly_idx = rules_gpt.index('Polyomino')
        ax2.set_theta_offset(np.pi / 2 - angles2[poly_idx])

    for variant, color in zip(variants, colors):
        values = [solve_rates_gpt[rule].get(variant, 0) for rule in rules_gpt]
        values += values[:1]
        ax2.plot(angles2, values, 'o-', linewidth=1.5,
                 label=variant, color=color, markersize=4, zorder=3)
        ax2.fill(angles2, values, alpha=0.1, color=color, zorder=2)

    ax2.set_xticks(angles2[:-1])
    ax2.set_xticklabels(rules_gpt)
    max_val2 = max([max(solve_rates_gpt[r].values()) for r in rules_gpt if solve_rates_gpt[r].values()])
    ax2.set_ylim(0, max(12, max_val2 * 1.15))
    ax2.set_yticks([5, 10, 15, 20])
    ax2.set_yticklabels(['5\\%', '10\\%', '15\\%', '20\\%'])
    for label in ax2.yaxis.get_ticklabels():
        label.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])
        label.set_zorder(10)
    style_polar_grid(ax2, [5, 10, 15, 20])
    ax2.set_title('(b) GPT-OSS-120B')

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    fig.legend(legend_handles, variants, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")

    plt.close(fig)


def main():
    results_dir = Path(__file__).parent / "results" / "spatial_gym"
    
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

    print("\n" + "=" * 60)
    print("Creating combined radar plot (all models + GPT-OSS-120B)...")
    print("=" * 60)

    output_pdf_combined = Path(__file__).parent / "solve_rate_by_rule_combined.pdf"
    create_combined_radar_plot(results_dir, output_pdf_combined)


if __name__ == "__main__":
    main()
