"""
Script to create a 3-panel line plot showing average tokens per puzzle by difficulty level
for each model individually, with subplots for SPaRC, Gym w/o traceback, and Gym w/ traceback.
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

import tiktoken
from matplotlib.offsetbox import AnnotationBbox

from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    get_model_imagebox,
    MODEL_COLORS,
    figure_fraction_anchor_from_display_xy,
)

# ---------------------------------------------------------------------------
# Config (mirrors plot_difficulty_comparison.py)
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "allenai_Olmo-3.1-32B-Think": "OLMo 3.1 32B",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron 49B",
    "Qwen_Qwen3-32B": "Qwen 3 32B",
    "Qwen_Qwen3-14B": "Qwen 3 14B",
    "Qwen_Qwen3-4B": "Qwen 3 4B",
    "Qwen_Qwen3-0.6B": "Qwen 3 0.6B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "R1 Distill 32B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "9Tobi_ragen_sparc_qwen3_4B_CW3": "Qwen 3 4B (FT)",
    "mistralai_Magistral-Small-2507": "Magistral Small",
    "Qwen_Qwen3-VL-32B-Thinking": "Qwen 3 VL 32B",
}

# Logo-inspired fallback colors keyed on tokens in the internal model path/name
MODEL_FAMILY_COLORS = {
    "openai":    "#10A37F",  # GPT / OpenAI → ChatGPT teal-green
    "gpt":       "#10A37F",
    "google":    "#4E84C4",  # Gemma → blue (Gemma logo)
    "gemma":     "#4E84C4",
    "Qwen":      "#6040E0",  # Qwen → purple-indigo (Qwen logo)
    "qwen":      "#6040E0",
    "deepseek":  "#4A6EA8",  # R1 / DeepSeek → cobalt blue
    "nvidia":    "#76B900",  # Nemotron → NVIDIA lime green
    "allenai":   "#D43870",  # OLMo → hot pink (OLMo logo)
    "9Tobi":     "#9070F0",  # Fine-tuned Qwen → Qwen purple
    "mistralai": "#D96818",  # Magistral → warm orange (Mistral logo)
}

MODEL_MARKERS = {
    "GPT-OSS 120B": "o",
    "OLMo 3.1 32B": "s",
    "Nemotron 49B": "^",
    "Qwen 3 32B": "D",
    "Qwen 3 14B": "v",
    "Qwen 3 4B": "p",
    "Qwen 3 0.6B": "h",
    "R1 Distill 32B": "P",
    "Gemma 3 27B": "*",
    "Magistral Small": "X",
    "Qwen 3 4B (FT)": "8",
}

# Skip no-reason / visual / ablation variants
SKIP_PATTERNS = ["no-reason", "no_reason", "visual", "ablation", "baseline", "astar", "random"]

# Only include these models (same as accuracy chart)
INCLUDED_MODELS = {
    "GPT-OSS 120B",
    "OLMo 3.1 32B",
    "Nemotron 49B",
    "Qwen 3 32B",
    "Qwen 3 0.6B",
    "R1 Distill 32B",
    "Gemma 3 27B",
    "Magistral Small",
}

FILE_TYPE_TO_VARIANT = {
    "sparc": "Baseline",
    "gym": "Gym w/o traceback",
    "traceback": "Gym w/ traceback",
}

# ---------------------------------------------------------------------------
# Token counting (reused from plot_token_by_difficulty.py)
# ---------------------------------------------------------------------------

ENCODING = tiktoken.get_encoding("cl100k_base")


def _tokens_for_record(record):
    result = record.get("result", {})
    messages = result.get("message", [])
    total = 0
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, str):
                total += len(ENCODING.encode(msg))
    elif isinstance(messages, str):
        total = len(ENCODING.encode(messages))
    return total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_token_by_difficulty_per_model(results_dir):
    """
    Load average tokens per puzzle by difficulty level for each model.

    Returns a DataFrame with columns:
        Internal Name | Display Name | File Type | Difficulty | Avg Tokens
    """
    results_path = Path(results_dir)
    cache_path = results_path.parent / "token_by_difficulty_per_model_cache.csv"

    # Use cache if up-to-date
    if cache_path.exists():
        jsonl_files = list(results_path.glob("*.jsonl"))
        cache_mtime = cache_path.stat().st_mtime
        if all(f.stat().st_mtime < cache_mtime for f in jsonl_files):
            print("Loading per-model token-by-difficulty data from cache...")
            return pd.read_csv(cache_path)

    print("Computing per-model token-by-difficulty data (this may take a while)…")
    records = []

    for jsonl_file in sorted(results_path.glob("*.jsonl")):
        fname = jsonl_file.name

        if any(p in fname.lower() for p in SKIP_PATTERNS):
            continue
        if "archive" in str(jsonl_file):
            continue

        # Classify and extract internal name
        if "_gym_traceback.jsonl" in fname:
            file_type = "traceback"
            internal = jsonl_file.stem.replace("_gym_traceback", "")
        elif "_gym.jsonl" in fname:
            file_type = "gym"
            internal = jsonl_file.stem.replace("_gym", "")
        elif "_gym_" not in fname:
            file_type = "sparc"
            internal = jsonl_file.stem
        else:
            continue  # Unknown gym suffix

        display_name = MODEL_DISPLAY_NAMES.get(internal, internal)
        if display_name not in INCLUDED_MODELS:
            continue

        print(f"  {fname} …")

        tokens_by_diff: dict[int, list[int]] = {}

        with open(jsonl_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                diff = rec.get("difficulty_level")
                if diff is None:
                    continue
                diff = int(diff)

                tokens = _tokens_for_record(rec)
                tokens_by_diff.setdefault(diff, []).append(tokens)

        for diff, tok_list in tokens_by_diff.items():
            records.append({
                "Internal Name": internal,
                "Display Name": display_name,
                "File Type": file_type,
                "Difficulty": diff,
                "Avg Tokens": float(np.mean(tok_list)),
                "Num Puzzles": len(tok_list),
            })

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"Cached to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def get_color_for_model(display_name, internal_name):
    if display_name in MODEL_COLORS:
        return MODEL_COLORS[display_name]
    for family, color in MODEL_FAMILY_COLORS.items():
        if family.lower() in internal_name.lower():
            return color
    return "#808080"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def create_token_by_difficulty_per_model_plot(results_dir, output_path=None):
    """Create 3-panel line plot of avg tokens by difficulty (one line per model)."""
    setup_plot_style(use_latex=True)

    df = load_token_by_difficulty_per_model(results_dir)
    if df.empty:
        print("No data available.")
        return None, None

    variant_names = ["Baseline", "Gym w/o traceback", "Gym w/ traceback"]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_INCHES, 2.0))

    difficulties = np.array([1, 2, 3, 4, 5])

    all_models_seen = {}  # display_name -> (color, marker, internal_name)

    for i, (ax, variant) in enumerate(zip(axes, variant_names)):
        file_type = {v: k for k, v in FILE_TYPE_TO_VARIANT.items()}[variant]
        vdf = df[df["File Type"] == file_type]

        if vdf.empty:
            ax.set_title(f'({chr(97 + i)}) {variant}')
            continue

        for internal_name in sorted(vdf["Internal Name"].unique()):
            display_name = MODEL_DISPLAY_NAMES.get(internal_name, internal_name)
            if display_name not in INCLUDED_MODELS:
                continue

            mdf = vdf[vdf["Internal Name"] == internal_name].sort_values("Difficulty")
            if mdf.empty:
                continue

            # Fill missing difficulties with NaN so the line has gaps
            x = np.array([d for d in difficulties if d in mdf["Difficulty"].values])
            y = np.array([
                mdf.loc[mdf["Difficulty"] == d, "Avg Tokens"].values[0] / 1000
                for d in difficulties
                if d in mdf["Difficulty"].values
            ])

            color = get_color_for_model(display_name, internal_name)
            marker = MODEL_MARKERS.get(display_name, "o")

            ax.plot(x, y,
                    color=color, marker=marker, markersize=4,
                    linewidth=1.2, label=display_name,
                    markeredgecolor='white', markeredgewidth=0.3)

            if display_name not in all_models_seen:
                all_models_seen[display_name] = (color, marker, internal_name)

        ax.set_title(f'({chr(97 + i)}) {variant}')
        ax.set_xlabel('Difficulty Level')
        ax.set_xticks(difficulties)
        ax.set_xlim(0.7, 5.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    for ax in axes:
        ax.set_ylabel('Avg Tokens per Puzzle (K)')

    # Unified deduplicated legend
    seen_labels = set()
    handles = []
    labels = []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen_labels:
                seen_labels.add(l)
                handles.append(h)
                labels.append(l)

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    leg = fig.legend(handles, labels, loc='lower center',
                     ncol=4, fontsize=7, frameon=False,
                     bbox_to_anchor=(0.5, -0.02),
                     handlelength=2, handletextpad=2.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for label, legend_handle in zip(labels, leg.legend_handles):
        imagebox = get_model_imagebox(label, zoom_factor=0.8)
        if not imagebox:
            continue

        bbox = legend_handle.get_window_extent(renderer)
        xd = bbox.x0 + 0.5 * bbox.width
        yd = bbox.y0 + 0.5 * bbox.height
        fx, fy = figure_fraction_anchor_from_display_xy(fig, (xd, yd), (-0.0225, -0.005))
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
        if str(output_path).endswith(".pdf"):
            png_path = str(output_path).replace(".pdf", ".png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {png_path}")

    plt.close(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results_dir = Path(__file__).parent / "results" / "spatial_gym"
    output_pdf = Path(__file__).parent / "token_by_difficulty_per_model.pdf"

    print("=" * 60)
    print("Creating per-model token-by-difficulty line plot…")
    print("=" * 60)
    create_token_by_difficulty_per_model_plot(results_dir, output_pdf)


if __name__ == "__main__":
    main()
