"""
Line plot showing average token usage per difficulty level (1–5) for
SPaRC, SPaRC-Gym, and Traceback variants, with ±1 std error bands
computed across models.

Uses the existing per-puzzle JSONL data; caches per-difficulty averages
to token_by_difficulty_cache.csv in the results directory.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
from pathlib import Path

from plot_config import (
    setup_plot_style,
    COLUMN_WIDTH_INCHES,
    TEXT_WIDTH_INCHES,
)

ENCODING = tiktoken.get_encoding("cl100k_base")

# Filename stem → display name (same mapping as plot_difficulty_comparison.py)
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
    "mistralai_Magistral-Small-2507": "Magistral Small",
}

# Subset used in the main accuracy / difficulty-comparison figures
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

# Filename substrings that mark variants we do NOT want
SKIP_PATTERNS = [
    "no-reason", "no_reason", "visual", "ablation",
    "baseline", "astar", "random",
]

VARIANT_CONFIG = {
    "sparc":     {"label": "SPaRC",     "color": "#2E7D32"},
    "gym":       {"label": "SPaRC-Gym", "color": "#1976D2"},
    "traceback": {"label": "Traceback", "color": "#E65100"},
}


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(ENCODING.encode(text))


def _tokens_for_record(record: dict) -> int:
    result = record.get("result", {})
    messages = result.get("message", [])
    total = 0
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, str):
                total += _count_tokens(msg)
    elif isinstance(messages, str):
        total = _count_tokens(messages)
    return total


# ---------------------------------------------------------------------------
# Cache loader / builder
# ---------------------------------------------------------------------------

def load_token_by_difficulty_data(results_dir):
    """Return a DataFrame with per-model, per-variant, per-difficulty token averages.

    Columns: Internal Name, File Type, Difficulty, Avg Tokens, Num Puzzles

    Results are cached in <results_dir>/token_by_difficulty_cache.csv and
    regenerated only when any JSONL file is newer than the cache.
    """
    results_path = Path(results_dir)
    cache_path = results_path / "token_by_difficulty_cache.csv"

    # Use cache when it is up-to-date
    if cache_path.exists():
        jsonl_files = list(results_path.glob("*.jsonl"))
        cache_mtime = cache_path.stat().st_mtime
        if all(f.stat().st_mtime < cache_mtime for f in jsonl_files):
            print("Loading token-by-difficulty data from cache...")
            return pd.read_csv(cache_path)

    print("Computing token-by-difficulty data (this may take a while)…")
    records = []

    for jsonl_file in sorted(results_path.glob("*.jsonl")):
        fname = jsonl_file.name

        # Skip unwanted variants
        if any(p in fname.lower() for p in SKIP_PATTERNS):
            continue
        if "archive" in str(jsonl_file):
            continue

        # Classify the file
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
            continue  # gym_visual or other unknown suffix

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
# Plot
# ---------------------------------------------------------------------------

def create_token_by_difficulty_plot(results_dir, output_path=None):
    """Single-panel line plot: avg tokens/puzzle vs. difficulty for GPT-OSS 120B.
    One line per variant (SPaRC / SPaRC-Gym / Traceback).
    """
    setup_plot_style(use_latex=True)

    df = load_token_by_difficulty_data(results_dir)
    if df.empty:
        print("No data available.")
        return None, None

    # Filter to Qwen3-32B only
    df = df[df["Internal Name"] == "Qwen_Qwen3-32B"].copy()
    if df.empty:
        print("No data for Qwen3-32B.")
        return None, None

    difficulties = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, 2.6))

    for file_type, cfg in VARIANT_CONFIG.items():
        vdf = df[df["File Type"] == file_type].sort_values("Difficulty")
        if vdf.empty:
            continue

        x = vdf["Difficulty"].values
        y = vdf["Avg Tokens"].values / 1000

        ax.plot(
            x, y,
            color=cfg["color"], linewidth=1.5, marker="o", markersize=4,
            label=cfg["label"], markeredgecolor="white", markeredgewidth=0.4,
            zorder=3,
        )

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Avg Tokens per Puzzle (K)")
    ax.set_xticks(difficulties)
    ax.set_xlim(0.7, 5.3)
    # linear scale

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")
        if str(output_path).endswith(".pdf"):
            png_path = str(output_path).replace(".pdf", ".png")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {png_path}")

    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results_dir = Path(__file__).parent / "results" / "sparc"
    output_pdf = Path(__file__).parent / "token_by_difficulty.pdf"

    print("=" * 60)
    print("Creating token-by-difficulty line plot…")
    print("=" * 60)
    create_token_by_difficulty_plot(results_dir, output_pdf)


if __name__ == "__main__":
    main()
