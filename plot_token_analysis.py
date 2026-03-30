"""
Token usage visualizations - comparing token counts across models and variants.
Includes SPaRC, Spatial Gym, and Spatial Gym Traceback.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import tiktoken
from pathlib import Path

from matplotlib.offsetbox import AnnotationBbox

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
    get_model_color,
    get_model_imagebox,
    MODEL_COLORS,
    figure_fraction_anchor_from_display_xy,
)

# Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
ENCODING = tiktoken.get_encoding("cl100k_base")

# Display names for models
MODEL_DISPLAY_NAMES = {
    'openai/gpt-oss-120b': 'GPT-OSS-120B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'R1-Distill-32B',
    'allenai/Olmo-3.1-32B-Think': 'OLMo-3.1-32B',
    'google/gemma-3-27b-it': 'Gemma-3-27B',
    'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5': 'Nemotron-49B',
    'Qwen/Qwen3-32B': 'Qwen3-32B',
    'Qwen/Qwen3-14B': 'Qwen3-14B',
    'Qwen/Qwen3-8B': 'Qwen3-8B',
    'Qwen/Qwen3-4B': 'Qwen3-4B',
    'Qwen/Qwen3-1.7B': 'Qwen3-1.7B',
    'Qwen/Qwen3-0.6B': 'Qwen3-0.6B',
    'mistralai/Magistral-Small-2507': 'Magistral-Small',
    'Qwen/Qwen3-VL-32B-Thinking': 'Qwen3-VL-32B',
}


def count_tokens(text):
    """Count tokens using tiktoken."""
    if not text:
        return 0
    return len(ENCODING.encode(text))


def calculate_token_stats_from_jsonl(jsonl_path):
    """Calculate token statistics from a JSONL file."""
    tokens_list = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                result = record.get('result', {})
                messages = result.get('message', [])
                
                puzzle_tokens = 0
                if messages:
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, str) and msg:
                                puzzle_tokens += count_tokens(msg)
                    elif isinstance(messages, str):
                        puzzle_tokens = count_tokens(messages)
                
                tokens_list.append(puzzle_tokens)
            except json.JSONDecodeError:
                continue
    
    if tokens_list:
        return {
            'num_puzzles': len(tokens_list),
            'avg_tokens': np.mean(tokens_list),
            'total_tokens': sum(tokens_list),
        }
    return None


def load_all_token_data(results_dir):
    """Load token data for all variants: SPaRC, Spatial Gym, Traceback.
    Uses existing token_analysis.csv for gym/traceback data.
    """
    results_path = Path(results_dir)
    
    # Load existing token_analysis.csv which has gym (standard) and traceback
    csv_path = results_path / "token_analysis.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Rename 'standard' to 'gym' for consistency
        df['File Type'] = df['File Type'].replace('standard', 'gym')
        # Filter out visual variant
        df = df[df['File Type'] != 'visual']
        return df
    
    # Fallback: compute from scratch (slow)
    print("No token_analysis.csv found. Computing from scratch...")
    cache_path = results_path / "token_cache.csv"
    
    # Check if cache exists and is newer than all jsonl files
    if cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        jsonl_files = list(results_path.glob("*.jsonl"))
        if all(f.stat().st_mtime < cache_mtime for f in jsonl_files):
            print("Loading token data from cache...")
            return pd.read_csv(cache_path)
    
    print("Computing token counts (this may take a while)...")
    records = []
    
    # SPaRC (non-gym)
    for jsonl_file in results_path.glob("*.jsonl"):
        if "_gym" in jsonl_file.name or "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem.replace("_", "/", 1)
        print(f"  Processing {jsonl_file.name}...")
        stats = calculate_token_stats_from_jsonl(jsonl_file)
        
        if stats:
            records.append({
                'Model': model_name,
                'File Type': 'sparc',
                'Num Puzzles': stats['num_puzzles'],
                'Avg Tokens per Puzzle': stats['avg_tokens'],
            })
    
    # Spatial Gym
    for jsonl_file in results_path.glob("*_gym.jsonl"):
        if "traceback" in jsonl_file.name or "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem.replace("_gym", "").replace("_", "/", 1)
        print(f"  Processing {jsonl_file.name}...")
        stats = calculate_token_stats_from_jsonl(jsonl_file)
        
        if stats:
            records.append({
                'Model': model_name,
                'File Type': 'gym',
                'Num Puzzles': stats['num_puzzles'],
                'Avg Tokens per Puzzle': stats['avg_tokens'],
            })
    
    # Traceback
    for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
        if "archive" in str(jsonl_file) or "visual" in jsonl_file.name:
            continue
        
        model_name = jsonl_file.stem.replace("_gym_traceback", "").replace("_", "/", 1)
        print(f"  Processing {jsonl_file.name}...")
        stats = calculate_token_stats_from_jsonl(jsonl_file)
        
        if stats:
            records.append({
                'Model': model_name,
                'File Type': 'traceback',
                'Num Puzzles': stats['num_puzzles'],
                'Avg Tokens per Puzzle': stats['avg_tokens'],
            })
    
    df = pd.DataFrame(records)
    
    # Save cache
    df.to_csv(cache_path, index=False)
    print(f"Token data cached to {cache_path}")
    
    return df


def load_accuracy_data(results_dir):
    """Load accuracy from stats files."""
    results_path = Path(results_dir)
    
    accuracy = {}
    
    # SPaRC (non-gym)
    for stats_file in results_path.glob("*_stats.csv"):
        if "_gym" in stats_file.name:
            continue
        model_name = stats_file.stem.replace("_stats", "").replace("_", "/", 1)
        df = pd.read_csv(stats_file)
        # Find row with "Correctly Solved"
        solved_row = df[df['Metric'] == 'Correctly Solved']
        if len(solved_row) > 0:
            pct = solved_row['Percentage'].iloc[0]
            if isinstance(pct, str):
                pct = float(pct.replace('%', ''))
            accuracy[(model_name, 'sparc')] = pct
    
    # Standard gym
    for stats_file in results_path.glob("*_gym_stats.csv"):
        if "traceback" in stats_file.name:
            continue
        model_name = stats_file.stem.replace("_gym_stats", "").replace("_", "/", 1)
        df = pd.read_csv(stats_file)
        # Find row with "Correctly Solved"
        solved_row = df[df['Metric'] == 'Correctly Solved']
        if len(solved_row) > 0:
            pct = solved_row['Percentage'].iloc[0]
            if isinstance(pct, str):
                pct = float(pct.replace('%', ''))
            accuracy[(model_name, 'gym')] = pct
    
    # Traceback
    for stats_file in results_path.glob("*_gym_traceback_stats.csv"):
        model_name = stats_file.stem.replace("_gym_traceback_stats", "").replace("_", "/", 1)
        df = pd.read_csv(stats_file)
        solved_row = df[df['Metric'] == 'Correctly Solved']
        if len(solved_row) > 0:
            pct = solved_row['Percentage'].iloc[0]
            if isinstance(pct, str):
                pct = float(pct.replace('%', ''))
            accuracy[(model_name, 'traceback')] = pct
    
    return accuracy


def create_tokens_vs_accuracy(results_dir, output_path=None):
    """Create scatter plot of tokens vs accuracy."""
    setup_plot_style(use_latex=True)
    
    token_df = load_all_token_data(results_dir)
    accuracy = load_accuracy_data(results_dir)
    
    if token_df is None or len(token_df) == 0:
        print("No token data found!")
        return None, None
    
    # Filter to models that have all 3 variants
    model_variants = token_df.groupby('Model')['File Type'].apply(set)
    complete_models = model_variants[model_variants.apply(lambda x: {'sparc', 'gym', 'traceback'}.issubset(x))].index
    token_df = token_df[token_df['Model'].isin(complete_models)]
    
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, 2.5))
    
    # Shapes for variants
    variant_markers = {'sparc': '^', 'gym': 'o', 'traceback': 's'}
    variant_labels = {'sparc': 'Baseline', 'gym': 'Gym w/o backtracking', 'traceback': 'Gym w/ backtracking'}
    
    # Collect data by model
    plotted_models = set()
    plotted_variants = set()
    
    for _, row in token_df.iterrows():
        model = row['Model']
        file_type = row['File Type']
        avg_tokens = row['Avg Tokens per Puzzle']
        
        if avg_tokens == 0 or row['Num Puzzles'] == 0:
            continue
        
        acc = accuracy.get((model, file_type), None)
        if acc is None:
            continue
        
        display_name = MODEL_DISPLAY_NAMES.get(model, model.split('/')[-1])
        color = MODEL_COLORS.get(display_name, get_model_color(display_name, warn_on_missing=False))
        marker = variant_markers.get(file_type, 'o')
        
        x = avg_tokens / 1000
        y = acc
        
        # Plot point
        ax.scatter(x, y, s=36, c=color, alpha=0.85, marker=marker,
                  edgecolors='white', linewidth=0.5, zorder=3)
        
        plotted_models.add(display_name)
        plotted_variants.add(file_type)
    
    ax.set_xlabel('Avg Tokens per Puzzle (K)')
    ax.set_ylabel('Accuracy (\\%)')
    
    ax.set_xscale('log')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Create two legends: one for models (colors), one for variants (shapes)
    from matplotlib.lines import Line2D
    
    # Model legend (colors)
    model_labels = sorted(plotted_models, key=lambda m: m if m in MODEL_COLORS else '')
    model_labels = [m for m in model_labels if m in MODEL_COLORS]
    model_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=MODEL_COLORS.get(m, '#666'),
                           markersize=6, label=m) for m in model_labels]
    
    # Variant legend (shapes)  
    variant_handles = [Line2D([0], [0], marker=variant_markers[v], color='w', markerfacecolor='#444',
                             markersize=6, label=variant_labels[v]) for v in ['sparc', 'gym', 'traceback'] if v in plotted_variants]
    
    # Place legends inside plot
    leg1 = ax.legend(handles=model_handles, loc='upper left', fontsize=7, 
                    frameon=True, framealpha=0.95, title='Model', title_fontsize=7,
                    handletextpad=2.0)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=variant_handles, loc='upper right', fontsize=7,
                    frameon=True, framealpha=0.95, title='Variant', title_fontsize=7)
    
    plt.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, legend_handle in zip(model_labels, leg1.legend_handles):
        imagebox = get_model_imagebox(label, zoom_factor=0.8)
        if not imagebox:
            continue
        bbox = legend_handle.get_window_extent(renderer)
        xd = bbox.x0 + 0.5 * bbox.width
        yd = bbox.y0 + 0.5 * bbox.height
        fx, fy = figure_fraction_anchor_from_display_xy(fig, (xd, yd), (-0.025, -0.05))
        ab = AnnotationBbox(imagebox, (fx, fy),
                            xybox=(15, 0),
                            xycoords='figure fraction',
                            boxcoords="offset points",
                            frameon=False,
                            box_alignment=(0.5, 0.5),
                            zorder=10)
        fig.add_artist(ab)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    return fig, ax


def create_token_comparison_bar(results_dir, output_path=None):
    """Create bar chart comparing token usage across models."""
    setup_plot_style(use_latex=True)
    
    token_df = load_all_token_data(results_dir)
    
    if token_df is None or len(token_df) == 0:
        print("No token data found!")
        return None, None
    
    # Filter to models that have all 3 variants
    model_variants = token_df.groupby('Model')['File Type'].apply(set)
    complete_models = model_variants[model_variants.apply(lambda x: {'sparc', 'gym', 'traceback'}.issubset(x))].index
    token_df = token_df[token_df['Model'].isin(complete_models)]
    
    # Pivot to get models as rows and file types as columns
    pivot_df = token_df.pivot_table(
        index='Model', 
        columns='File Type', 
        values='Avg Tokens per Puzzle',
        aggfunc='first'
    ).reset_index()
    
    merged = pivot_df.copy()
    
    if len(merged) == 0:
        print("No models with all three variants!")
        return None, None
    
    # Sort by gym tokens
    merged = merged.sort_values('gym', ascending=True)
    
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, 2.5))
    
    n_models = len(merged)
    y_pos = np.arange(n_models)
    
    display_names = [MODEL_DISPLAY_NAMES.get(m, m.split('/')[-1]) for m in merged['Model']]
    
    height = 0.25
    sparc_tokens = merged['sparc'].values / 1000
    gym_tokens = merged['gym'].values / 1000
    tb_tokens = merged['traceback'].values / 1000
    
    bars1 = ax.barh(y_pos - height, sparc_tokens, height, label='Baseline', color='#2E7D32', alpha=0.8)
    bars2 = ax.barh(y_pos, gym_tokens, height, label='Spatial Gym', color='#1976D2', alpha=0.8)
    bars3 = ax.barh(y_pos + height, tb_tokens, height, label='Traceback', color='#E65100', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Avg Tokens per Puzzle (K)')
    ax.set_xscale('log')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    return fig, ax


def create_efficiency_chart(results_dir, output_path=None):
    """Create chart showing accuracy per 1000 tokens (efficiency)."""
    setup_plot_style(use_latex=True)
    
    token_df = load_all_token_data(results_dir)
    accuracy = load_accuracy_data(results_dir)
    
    if token_df is None or len(token_df) == 0:
        print("No token data found!")
        return None, None
    
    # Filter to models that have all 3 variants
    model_variants = token_df.groupby('Model')['File Type'].apply(set)
    complete_models = model_variants[model_variants.apply(lambda x: {'sparc', 'gym', 'traceback'}.issubset(x))].index
    token_df = token_df[token_df['Model'].isin(complete_models)]
    
    # Calculate efficiency
    records = []
    for _, row in token_df.iterrows():
        model = row['Model']
        file_type = row['File Type']
        avg_tokens = row['Avg Tokens per Puzzle']
        
        if avg_tokens == 0 or row['Num Puzzles'] == 0:
            continue
        
        acc = accuracy.get((model, file_type), None)
        if acc is None:
            continue
        
        # Efficiency: accuracy points per 1K tokens
        efficiency = acc / (avg_tokens / 1000)
        
        display_name = MODEL_DISPLAY_NAMES.get(model, model.split('/')[-1])
        records.append({
            'model': display_name,
            'file_type': file_type,
            'accuracy': acc,
            'tokens_k': avg_tokens / 1000,
            'efficiency': efficiency,
        })
    
    if not records:
        print("No efficiency data!")
        return None, None
    
    df = pd.DataFrame(records)
    
    # Group by model and create side-by-side bars
    models = df['model'].unique()
    file_types = ['sparc', 'gym', 'traceback']
    color_map = {'sparc': '#2E7D32', 'gym': '#1976D2', 'traceback': '#E65100'}
    
    # Pivot data for grouped bar chart
    pivot = df.pivot_table(index='model', columns='file_type', values='efficiency', aggfunc='first')
    
    # Sort by max efficiency across variants
    pivot['max_eff'] = pivot.max(axis=1)
    pivot = pivot.sort_values('max_eff', ascending=True)
    pivot = pivot.drop('max_eff', axis=1)
    
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, 2.8))
    
    y_pos = np.arange(len(pivot))
    height = 0.25
    
    # Plot bars for each file type
    for i, ft in enumerate(file_types):
        if ft in pivot.columns:
            values = pivot[ft].fillna(0).values
            offset = (i - 1) * height
            ax.barh(y_pos + offset, values, height, label=ft.upper() if ft == 'sparc' else ('Spatial Gym' if ft == 'gym' else 'Traceback'), 
                   color=color_map[ft], alpha=0.8)
    
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Efficiency (Accuracy \\% per 1K tokens)')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        if str(output_path).endswith('.pdf'):
            png_path = str(output_path).replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")
    
    plt.close(fig)
    
    # Print stats
    print("\nToken efficiency (accuracy % per 1K tokens):")
    for _, row in df.sort_values('efficiency', ascending=False).iterrows():
        print(f"  {row['model']} ({row['file_type']}): {row['efficiency']:.4f} ({row['accuracy']:.1f}% / {row['tokens_k']:.1f}K)")
    
    return fig, ax


def main():
    results_dir = Path(__file__).parent / "results" / "spatial_gym"
    
    print("=" * 60)
    print("Creating tokens vs accuracy scatter plot...")
    print("=" * 60)
    
    output1 = Path(__file__).parent / "tokens_vs_accuracy.pdf"
    create_tokens_vs_accuracy(results_dir, output1)
    
    print("\n" + "=" * 60)
    print("Creating token comparison bar chart...")
    print("=" * 60)
    
    output2 = Path(__file__).parent / "token_comparison.pdf"
    create_token_comparison_bar(results_dir, output2)
    
    print("\n" + "=" * 60)
    print("Creating token efficiency chart...")
    print("=" * 60)
    
    output3 = Path(__file__).parent / "token_efficiency.pdf"
    create_efficiency_chart(results_dir, output3)


if __name__ == "__main__":
    main()
