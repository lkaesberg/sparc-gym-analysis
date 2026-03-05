"""
Improvement ceiling analysis: Compare puzzles where standard fails but traceback succeeds
(and vice versa) to understand characteristics that benefit from traceback.
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


def load_jsonl_data(filepath):
    """Load all entries from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_puzzle_results(results_dir):
    """Extract puzzle results from both standard and traceback variants."""
    results_path = Path(results_dir)
    
    # Store results by model and puzzle_id
    standard_results = defaultdict(dict)  # model -> puzzle_id -> result
    traceback_results = defaultdict(dict)
    puzzle_info = {}  # puzzle_id -> info (difficulty, etc.)
    
    # Load standard gym results
    for jsonl_file in results_path.glob("*_gym.jsonl"):
        if "traceback" in jsonl_file.name or "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem.replace("_gym", "")
        data = load_jsonl_data(jsonl_file)
        
        for entry in data:
            puzzle_id = entry.get('id')
            result = entry.get('result', {})
            
            if puzzle_id:
                standard_results[model_name][puzzle_id] = {
                    'solved': result.get('solved', False),
                    'steps_taken': result.get('steps_taken'),
                }
                
                # Store puzzle info
                if puzzle_id not in puzzle_info:
                    puzzle_info[puzzle_id] = {
                        'difficulty_score': entry.get('difficulty_score'),
                        'difficulty_level': entry.get('difficulty_level'),
                        'grid_size': entry.get('grid_size', {}),
                        'solution_count': entry.get('solution_count'),
                    }
    
    # Load traceback results
    for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
        if "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem.replace("_gym_traceback", "")
        data = load_jsonl_data(jsonl_file)
        
        for entry in data:
            puzzle_id = entry.get('id')
            result = entry.get('result', {})
            
            if puzzle_id:
                traceback_results[model_name][puzzle_id] = {
                    'solved': result.get('solved', False),
                    'steps_taken': result.get('steps_taken'),
                }
    
    return standard_results, traceback_results, puzzle_info


def categorize_puzzles(standard_results, traceback_results, puzzle_info):
    """Categorize puzzles into improvement categories."""
    categories = {
        'both_succeed': [],
        'both_fail': [],
        'traceback_helps': [],  # standard fails, traceback succeeds
        'traceback_hurts': [],  # standard succeeds, traceback fails
    }
    
    # Find models that have both standard and traceback
    common_models = set(standard_results.keys()) & set(traceback_results.keys())
    
    for model in common_models:
        std_puzzles = standard_results[model]
        tb_puzzles = traceback_results[model]
        
        common_puzzles = set(std_puzzles.keys()) & set(tb_puzzles.keys())
        
        for puzzle_id in common_puzzles:
            std_solved = std_puzzles[puzzle_id]['solved']
            tb_solved = tb_puzzles[puzzle_id]['solved']
            
            info = puzzle_info.get(puzzle_id, {})
            record = {
                'model': model,
                'puzzle_id': puzzle_id,
                'difficulty_score': info.get('difficulty_score'),
                'difficulty_level': info.get('difficulty_level'),
                'solution_count': info.get('solution_count'),
                'grid_size': info.get('grid_size'),
                'std_steps': std_puzzles[puzzle_id].get('steps_taken'),
                'tb_steps': tb_puzzles[puzzle_id].get('steps_taken'),
            }
            
            if std_solved and tb_solved:
                categories['both_succeed'].append(record)
            elif not std_solved and not tb_solved:
                categories['both_fail'].append(record)
            elif not std_solved and tb_solved:
                categories['traceback_helps'].append(record)
            else:  # std_solved and not tb_solved
                categories['traceback_hurts'].append(record)
    
    return categories


def create_improvement_analysis(results_dir, output_path=None):
    """Create visualization of improvement ceiling analysis."""
    setup_plot_style(use_latex=True)
    
    print("Extracting puzzle results...")
    standard_results, traceback_results, puzzle_info = extract_puzzle_results(results_dir)
    
    print(f"Models with standard results: {len(standard_results)}")
    print(f"Models with traceback results: {len(traceback_results)}")
    
    categories = categorize_puzzles(standard_results, traceback_results, puzzle_info)
    
    # Print summary
    total = sum(len(v) for v in categories.values())
    print(f"\nCategory breakdown (total {total} puzzle-model pairs):")
    for cat, records in categories.items():
        print(f"  {cat}: {len(records)} ({100*len(records)/total:.1f}%)")
    
    # Create DataFrames for analysis
    dfs = {cat: pd.DataFrame(records) for cat, records in categories.items() if records}
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_INCHES, 4.5))
    
    # Colors for categories
    colors = {
        'both_succeed': '#4CAF50',
        'both_fail': '#9E9E9E',
        'traceback_helps': '#2196F3',
        'traceback_hurts': '#F44336',
    }
    labels = {
        'both_succeed': 'Both Succeed',
        'both_fail': 'Both Fail',
        'traceback_helps': 'Traceback Helps',
        'traceback_hurts': 'Traceback Hurts',
    }
    
    # Plot 1: Category distribution (pie chart)
    ax1 = axes[0, 0]
    sizes = [len(categories[cat]) for cat in ['both_succeed', 'both_fail', 'traceback_helps', 'traceback_hurts']]
    pie_colors = [colors[cat] for cat in ['both_succeed', 'both_fail', 'traceback_helps', 'traceback_hurts']]
    pie_labels = [labels[cat] for cat in ['both_succeed', 'both_fail', 'traceback_helps', 'traceback_hurts']]
    
    wedges, texts, autotexts = ax1.pie(sizes, colors=pie_colors, autopct='%1.1f%%',
                                        startangle=90, pctdistance=0.75)
    for autotext in autotexts:
        autotext.set_fontsize(7)
    ax1.set_title('Outcome Distribution', fontweight='bold')
    
    # Plot 2: Difficulty distribution by category
    ax2 = axes[0, 1]
    cat_order = ['traceback_helps', 'traceback_hurts', 'both_succeed', 'both_fail']
    positions = []
    box_data = []
    box_colors = []
    
    for i, cat in enumerate(cat_order):
        if cat in dfs and 'difficulty_score' in dfs[cat].columns:
            data = dfs[cat]['difficulty_score'].dropna()
            if len(data) > 0:
                positions.append(i)
                box_data.append(data.values)
                box_colors.append(colors[cat])
    
    if box_data:
        bp = ax2.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
        
        ax2.set_xticks(range(len(cat_order)))
        ax2.set_xticklabels(['TB Helps', 'TB Hurts', 'Both OK', 'Both Fail'], rotation=15)
        ax2.set_ylabel('Difficulty Score')
        ax2.set_title('Difficulty by Category', fontweight='bold')
    
    # Plot 3: Solution count distribution
    ax3 = axes[1, 0]
    for cat in ['traceback_helps', 'traceback_hurts']:
        if cat in dfs and 'solution_count' in dfs[cat].columns:
            data = dfs[cat]['solution_count'].dropna()
            if len(data) > 0:
                # Bin solution counts
                bins = [0, 1, 2, 5, 10, 50, 100, 1000]
                hist, _ = np.histogram(data, bins=bins)
                hist_pct = hist / len(data) * 100
                x = np.arange(len(bins)-1)
                width = 0.35
                offset = -width/2 if cat == 'traceback_helps' else width/2
                ax3.bar(x + offset, hist_pct, width, label=labels[cat], color=colors[cat], alpha=0.8)
    
    ax3.set_xticks(np.arange(len(bins)-1))
    ax3.set_xticklabels(['1', '2', '3-5', '6-10', '11-50', '51-100', '>100'], fontsize=7)
    ax3.set_xlabel('Solution Count')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Solution Count Distribution', fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    
    # Plot 4: Per-model breakdown
    ax4 = axes[1, 1]
    model_stats = defaultdict(lambda: {'helps': 0, 'hurts': 0, 'total': 0})
    
    for cat in ['traceback_helps', 'traceback_hurts']:
        if cat in dfs:
            for _, row in dfs[cat].iterrows():
                model = row['model']
                model_stats[model]['total'] += 1
                if cat == 'traceback_helps':
                    model_stats[model]['helps'] += 1
                else:
                    model_stats[model]['hurts'] += 1
    
    # Also count total puzzles per model
    for cat in ['both_succeed', 'both_fail']:
        if cat in dfs:
            for _, row in dfs[cat].iterrows():
                model_stats[row['model']]['total'] += 1
    
    # Calculate net benefit and sort
    models = list(model_stats.keys())
    net_benefits = []
    for m in models:
        helps = model_stats[m]['helps']
        hurts = model_stats[m]['hurts']
        total = model_stats[m]['total']
        net_benefit = (helps - hurts) / total * 100 if total > 0 else 0
        net_benefits.append(net_benefit)
    
    # Sort by net benefit
    sorted_idx = np.argsort(net_benefits)[::-1]
    models_sorted = [models[i] for i in sorted_idx]
    net_benefits_sorted = [net_benefits[i] for i in sorted_idx]
    
    # Shorten model names
    short_names = []
    for m in models_sorted:
        name = m.replace('_', ' ').replace('-', ' ')
        # Take key identifier
        if 'Qwen3' in m:
            short = 'Qwen3-' + m.split('Qwen3')[1].split('_')[0]
        elif 'DeepSeek' in m:
            short = 'R1-Distill'
        elif 'Olmo' in m:
            short = 'OLMo-3.1'
        elif 'gemma' in m:
            short = 'Gemma-3'
        elif 'Nemotron' in m:
            short = 'Nemotron'
        elif 'gpt-oss' in m:
            short = 'GPT-OSS'
        elif 'Magistral' in m:
            short = 'Magistral'
        else:
            short = m[:12]
        short_names.append(short)
    
    y_pos = np.arange(len(models_sorted))
    bar_colors = ['#2196F3' if nb >= 0 else '#F44336' for nb in net_benefits_sorted]
    ax4.barh(y_pos, net_benefits_sorted, color=bar_colors, alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(short_names, fontsize=7)
    ax4.set_xlabel('Net Benefit (\\%)')
    ax4.set_title('Traceback Net Benefit by Model', fontweight='bold')
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.invert_yaxis()
    
    # Style all axes
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    
    plt.close(fig)
    
    # Print detailed statistics
    print("\n" + "=" * 60)
    print("Detailed Statistics:")
    print("=" * 60)
    
    for cat in ['traceback_helps', 'traceback_hurts']:
        if cat in dfs and len(dfs[cat]) > 0:
            df = dfs[cat]
            print(f"\n{labels[cat]} ({len(df)} cases):")
            print(f"  Mean difficulty: {df['difficulty_score'].mean():.2f}")
            print(f"  Median difficulty: {df['difficulty_score'].median():.2f}")
            print(f"  Mean solution count: {df['solution_count'].mean():.1f}")
            print(f"  Difficulty range: [{df['difficulty_score'].min():.2f}, {df['difficulty_score'].max():.2f}]")
    
    return fig, categories


def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    
    print("=" * 60)
    print("Improvement Ceiling Analysis")
    print("=" * 60)
    
    output_pdf = Path(__file__).parent / "improvement_ceiling.pdf"
    output_png = Path(__file__).parent / "improvement_ceiling.png"
    
    create_improvement_analysis(results_dir, output_pdf)
    create_improvement_analysis(results_dir, output_png)


if __name__ == "__main__":
    main()
