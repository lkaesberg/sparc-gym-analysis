"""
Generate a LaTeX table comparing model results across SPaRC, Spatial Gym, and
Spatial Gym Traceback variants.

Only models that have results for *all three* configurations are included.
Logos are referenced via images/logos/<name>.png for inclusion in the paper.
"""

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results" / "sparc"

# Display names for each model base name
MODEL_DISPLAY_NAMES = {
    "Qwen_Qwen3-0.6B": "Qwen 3 0.6B",
    "Qwen_Qwen3-32B": "Qwen 3 32B",
    "allenai_Olmo-3.1-32B-Think": "OLMo 3.1 32B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "R1 Distill 32B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "mistralai_Magistral-Small-2507": "Magistral Small",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron 49B",
    "openai_gpt-oss-120b": "GPT-OSS 120B",
}

# Map from model base name → logo file (relative to images/logos/)
MODEL_LOGOS = {
    "Qwen_Qwen3-0.6B": "qwen.png",
    "Qwen_Qwen3-32B": "qwen.png",
    "allenai_Olmo-3.1-32B-Think": "olmo.png",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "deepseek.png",
    "google_gemma-3-27b-it": "gemma.png",
    "mistralai_Magistral-Small-2507": "mistral.png",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "nvidia.png",
    "openai_gpt-oss-120b": "openai.png",
}

# Desired row order (sorted by parameter count / importance)
MODEL_ORDER = [
    "openai_gpt-oss-120b",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5",
    "Qwen_Qwen3-32B",
    "allenai_Olmo-3.1-32B-Think",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
    "google_gemma-3-27b-it",
    "mistralai_Magistral-Small-2507",
    "Qwen_Qwen3-0.6B",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_accuracy(stats_file: Path) -> float:
    """Return the 'Correctly Solved' percentage from a stats CSV."""
    df = pd.read_csv(stats_file)
    row = df[df["Metric"] == "Correctly Solved"]
    if len(row) == 0:
        return 0.0
    pct_str = str(row["Percentage"].values[0])
    m = re.search(r"([\d.]+)%", pct_str)
    return float(m.group(1)) if m else 0.0


def extract_difficulty_accuracies(stats_file: Path) -> dict:
    """Return per-difficulty accuracy from a stats CSV.

    Returns dict like {1: 19.8, 2: 4.2, 3: 2.5, 4: 1.2, 5: 0.0}
    """
    df = pd.read_csv(stats_file)
    difficulties = {}
    for _, row in df.iterrows():
        m = re.match(r"Difficulty (\d+) Solved", str(row["Metric"]))
        if m:
            diff_level = int(m.group(1))
            pct_str = str(row["Percentage"])
            pct_match = re.search(r"([\d.]+)%", pct_str)
            if pct_match:
                difficulties[diff_level] = float(pct_match.group(1))
            else:
                difficulties[diff_level] = 0.0
    return difficulties


def extract_avg_path_length(stats_file: Path) -> float:
    """Extract average path / step length from a stats CSV.

    SPaRC files use 'Avg Path Length'; gym files use 'Avg Steps Taken'.
    """
    df = pd.read_csv(stats_file)
    for _, row in df.iterrows():
        metric = str(row["Metric"]).strip()
        if metric in ("Avg Path Length", "Avg Steps Taken"):
            val = str(row["Value"])
            m = re.search(r"([\d.]+)", val)
            return float(m.group(1)) if m else 0.0
    return 0.0


def extract_gym_metrics(stats_file: Path) -> dict:
    """Extract gym-specific metrics: avg steps, reached-end rate."""
    df = pd.read_csv(stats_file)
    metrics = {}
    for _, row in df.iterrows():
        metric = str(row["Metric"]).strip()
        if metric == "Avg Steps Taken":
            val = str(row["Value"])
            m = re.search(r"([\d.]+)", val)
            metrics["avg_steps"] = float(m.group(1)) if m else 0.0
        elif metric == "Reached End":
            pct_str = str(row["Percentage"])
            m = re.search(r"([\d.]+)%", pct_str)
            metrics["reached_end"] = float(m.group(1)) if m else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_model_data() -> list[dict]:
    """Load data for every model that has all three variants."""
    models = []

    for base in MODEL_ORDER:
        sparc_stats = RESULTS_DIR / f"{base}_stats.csv"
        gym_stats = RESULTS_DIR / f"{base}_gym_stats.csv"
        tb_stats = RESULTS_DIR / f"{base}_gym_traceback_stats.csv"

        if not (sparc_stats.exists() and gym_stats.exists() and tb_stats.exists()):
            continue

        sparc_acc = extract_accuracy(sparc_stats)
        gym_acc = extract_accuracy(gym_stats)
        tb_acc = extract_accuracy(tb_stats)

        sparc_diff = extract_difficulty_accuracies(sparc_stats)
        gym_diff = extract_difficulty_accuracies(gym_stats)
        tb_diff = extract_difficulty_accuracies(tb_stats)

        sparc_path = extract_avg_path_length(sparc_stats)
        gym_path = extract_avg_path_length(gym_stats)
        tb_path = extract_avg_path_length(tb_stats)

        gym_metrics = extract_gym_metrics(gym_stats)
        tb_metrics = extract_gym_metrics(tb_stats)

        display = MODEL_DISPLAY_NAMES.get(base, base)
        logo = MODEL_LOGOS.get(base, "")

        models.append({
            "base": base,
            "display": display,
            "logo": logo,
            "sparc_acc": sparc_acc,
            "gym_acc": gym_acc,
            "tb_acc": tb_acc,
            "sparc_diff": sparc_diff,
            "gym_diff": gym_diff,
            "tb_diff": tb_diff,
            "sparc_path": sparc_path,
            "gym_path": gym_path,
            "tb_path": tb_path,
            "gym_steps": gym_metrics.get("avg_steps", 0),
            "gym_reached": gym_metrics.get("reached_end", 0),
            "tb_steps": tb_metrics.get("avg_steps", 0),
            "tb_reached": tb_metrics.get("reached_end", 0),
        })

    return models


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def bold_best_column(values: list[float], fmt: str = ".1f") -> list[str]:
    """Format a list of floats; boldface the maximum value(s) in the column."""
    best = max(values)
    formatted = []
    for v in values:
        s = f"{v:{fmt}}"
        if v == best and best > 0:
            formatted.append(f"\\textbf{{{s}}}")
        else:
            formatted.append(s)
    return formatted


def bold_best_column_min(values: list[float], fmt: str = ".1f") -> list[str]:
    """Format a list of floats; boldface the minimum (non-zero) value(s)."""
    nonzero = [v for v in values if v > 0]
    best = min(nonzero) if nonzero else 0
    formatted = []
    for v in values:
        s = f"{v:{fmt}}"
        if v == best and best > 0:
            formatted.append(f"\\textbf{{{s}}}")
        else:
            formatted.append(s)
    return formatted


def generate_single_variant_table(
    models: list[dict],
    variant: str,
    caption: str,
    label: str,
) -> str:
    """Generate a single LaTeX table for one variant (sparc / gym / tb).

    Each table has columns: Logo, Model, Acc, D1, D2, D3, D4, D5
    and boldfaces the best value in each *column* across models.
    """

    # Keys into the model dicts
    acc_key = {"sparc": "sparc_acc", "gym": "gym_acc", "tb": "tb_acc"}[variant]
    diff_key = {"sparc": "sparc_diff", "gym": "gym_diff", "tb": "tb_diff"}[variant]
    path_key = {"sparc": "sparc_path", "gym": "gym_path", "tb": "tb_path"}[variant]

    # Collect per-column values for bolding the best
    acc_vals = [m[acc_key] for m in models]
    diff_vals = {d: [m[diff_key].get(d, 0) for m in models] for d in range(1, 6)}
    path_vals = [m[path_key] for m in models]

    acc_fmt = bold_best_column(acc_vals)
    d_fmt = {d: bold_best_column(diff_vals[d]) for d in range(1, 6)}
    # No bolding for avg steps – it's informational, not a "best" metric
    path_fmt = [f"{v:.1f}" for v in path_vals]

    lines = [
        r"\begin{table*}[b]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{cl | c | ccccc | c}",
        r"    \toprule",
        r"    & \textbf{Model}"
        r" & \textbf{Accuracy}"
        r" & \textbf{D1} & \textbf{D2} & \textbf{D3} & \textbf{D4} & \textbf{D5}"
        r" & \textbf{Avg.\ Steps} \\",
        r"    \midrule",
    ]

    for i, m in enumerate(models):
        logo_cmd = ""
        if m["logo"]:
            logo_cmd = (
                r"\raisebox{-0.3\height}{"
                r"\includegraphics[width=1.2em]{images/logos/" + m["logo"] + r"}}"
            )

        row = (
            f"    {logo_cmd} & {m['display']}"
            f" & {acc_fmt[i]}"
            f" & {d_fmt[1][i]} & {d_fmt[2][i]} & {d_fmt[3][i]}"
            f" & {d_fmt[4][i]} & {d_fmt[5][i]}"
            f" & {path_fmt[i]} \\\\"
        )
        lines.append(row)

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table*}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models = load_all_model_data()

    if not models:
        print("No models found with all three variants!")
        return

    print(f"Found {len(models)} models with all three variants.\n")

    variant_configs = [
        {
            "variant": "sparc",
            "caption": r"SPaRC results. Overall accuracy (\%) and per-difficulty solve rate (D1--D5).",
            "label": "tab:sparc-results",
            "filename": "sparc_results_table.tex",
        },
        {
            "variant": "gym",
            "caption": r"Spatial Gym results. Overall accuracy (\%) and per-difficulty solve rate (D1--D5).",
            "label": "tab:sparc-gym-results",
            "filename": "sparc_gym_results_table.tex",
        },
        {
            "variant": "tb",
            "caption": r"Spatial Gym with Traceback results. Overall accuracy (\%) and per-difficulty solve rate (D1--D5).",
            "label": "tab:sparc-gym-tb-results",
            "filename": "sparc_gym_tb_results_table.tex",
        },
    ]

    out_dir = Path(__file__).parent / "results"
    for cfg in variant_configs:
        tex = generate_single_variant_table(
            models,
            variant=cfg["variant"],
            caption=cfg["caption"],
            label=cfg["label"],
        )
        out_path = out_dir / cfg["filename"]
        out_path.write_text(tex)
        print(f"Written: {out_path}")
        print(tex)
        print()


if __name__ == "__main__":
    main()
