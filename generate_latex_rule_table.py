"""
Generate three LaTeX tables (SPaRC, Spatial Gym, Spatial Gym + Traceback) showing
per-model solve rates broken down by puzzle rule type.

Each table has models on the y-axis and rule types on the x-axis.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results" / "sparc"

# All models we want to include, with their display names and logos.
# Separate orderings per variant since they have different model sets.
MODEL_DISPLAY_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron 49B",
    "Qwen_Qwen3-32B": "Qwen 3 32B",
    "allenai_Olmo-3.1-32B-Think": "OLMo 3.1 32B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "R1 Distill 32B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "mistralai_Magistral-Small-2507": "Magistral Small",
    "Qwen_Qwen3-14B": "Qwen 3 14B",
    "Qwen_Qwen3-4B": "Qwen 3 4B",
    "Qwen_Qwen3-0.6B": "Qwen 3 0.6B",
    "Qwen_Qwen3-VL-32B-Thinking": "Qwen 3 VL 32B",
    "9Tobi_ragen_sparc_qwen3_4B_CW3": "Qwen 3 4B (FT)",
}

MODEL_LOGOS = {
    "openai_gpt-oss-120b": "openai.png",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5": "nvidia.png",
    "Qwen_Qwen3-32B": "qwen.png",
    "allenai_Olmo-3.1-32B-Think": "olmo.png",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "deepseek.png",
    "google_gemma-3-27b-it": "gemma.png",
    "mistralai_Magistral-Small-2507": "mistral.png",
    "Qwen_Qwen3-14B": "qwen.png",
    "Qwen_Qwen3-4B": "qwen.png",
    "Qwen_Qwen3-0.6B": "qwen.png",
    "Qwen_Qwen3-VL-32B-Thinking": "qwen.png",
    "9Tobi_ragen_sparc_qwen3_4B_CW3": "qwen.png",
}

# Preferred row order (models appearing first will be listed first)
MODEL_ORDER = [
    "openai_gpt-oss-120b",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5",
    "Qwen_Qwen3-32B",
    "allenai_Olmo-3.1-32B-Think",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
    "google_gemma-3-27b-it",
    "mistralai_Magistral-Small-2507",
    "Qwen_Qwen3-14B",
    "Qwen_Qwen3-4B",
    "Qwen_Qwen3-VL-32B-Thinking",
    "9Tobi_ragen_sparc_qwen3_4B_CW3",
    "Qwen_Qwen3-0.6B",
]

# Canonical rule order for columns
RULE_ORDER = ["Dot", "Gap", "Polyomino", "Ylop", "Star", "Square", "Triangle"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_jsonl(filepath: Path) -> list[dict]:
    """Load all entries from a JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_puzzle_rules(text_viz: str) -> set[str]:
    """Extract rule types from text_visualization."""
    rules = set()
    for line in text_viz.split("\n"):
        ll = line.lower().strip()
        if "dot:" in ll:
            rules.add("Dot")
        elif "gap:" in ll:
            rules.add("Gap")
        elif 'type: "poly"' in ll:
            rules.add("Polyomino")
        elif 'type: "ylop"' in ll:
            rules.add("Ylop")
        elif 'type: "star"' in ll:
            rules.add("Star")
        elif 'type: "square"' in ll:
            rules.add("Square")
        elif 'type: "triangle"' in ll:
            rules.add("Triangle")
    return rules


def compute_per_model_rule_rates(jsonl_file: Path) -> dict[str, dict]:
    """Compute solve rate per rule for a single model's JSONL file.

    Returns: {rule: {"solved": int, "total": int, "rate": float}}
    """
    data = load_jsonl(jsonl_file)
    stats: dict[str, dict] = defaultdict(lambda: {"solved": 0, "total": 0})

    for entry in data:
        text_viz = entry.get("text_visualization", "")
        rules = extract_puzzle_rules(text_viz)
        solved = entry.get("result", {}).get("solved", False)

        for rule in rules:
            stats[rule]["total"] += 1
            if solved:
                stats[rule]["solved"] += 1

    result = {}
    for rule in RULE_ORDER:
        s = stats.get(rule, {"solved": 0, "total": 0})
        total = s["total"]
        rate = (s["solved"] / total * 100) if total > 0 else 0.0
        result[rule] = {"solved": s["solved"], "total": total, "rate": rate}

    return result


# ---------------------------------------------------------------------------
# Discover models per variant
# ---------------------------------------------------------------------------


def discover_models(variant: str) -> list[tuple[str, Path]]:
    """Return (model_base_name, jsonl_path) for each model in the variant,
    but only if the model has JSONL files for ALL three variants."""
    # First, find models that have all three variants
    sparc_bases = set()
    for f in RESULTS_DIR.glob("*.jsonl"):
        if "_gym" in f.name or "ablation" in f.name or "archive" in str(f):
            continue
        if "_no-reason" in f.name or "_visual" in f.name:
            continue
        sparc_bases.add(f.stem)

    gym_bases = set()
    for f in RESULTS_DIR.glob("*_gym.jsonl"):
        if "traceback" in f.name or "archive" in str(f):
            continue
        if "_no-reason" in f.name or "_visual" in f.name:
            continue
        gym_bases.add(f.stem.replace("_gym", ""))

    tb_bases = set()
    for f in RESULTS_DIR.glob("*_gym_traceback.jsonl"):
        if "archive" in str(f):
            continue
        tb_bases.add(f.stem.replace("_gym_traceback", ""))

    common_bases = sparc_bases & gym_bases & tb_bases

    # Now pick the right JSONL file for the requested variant
    found = {}
    if variant == "sparc":
        for base in common_bases:
            f = RESULTS_DIR / f"{base}.jsonl"
            if f.exists():
                found[base] = f
    elif variant == "gym":
        for base in common_bases:
            f = RESULTS_DIR / f"{base}_gym.jsonl"
            if f.exists():
                found[base] = f
    elif variant == "tb":
        for base in common_bases:
            f = RESULTS_DIR / f"{base}_gym_traceback.jsonl"
            if f.exists():
                found[base] = f

    # Sort according to MODEL_ORDER
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    sorted_bases = sorted(
        found.keys(),
        key=lambda b: order_map.get(b, 999),
    )

    return [(b, found[b]) for b in sorted_bases]


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------


def bold_best_column(values: list[float], fmt: str = ".1f") -> list[str]:
    """Boldface the maximum value(s) in a column."""
    best = max(values) if values else 0
    formatted = []
    for v in values:
        s = f"{v:{fmt}}"
        if v == best and best > 0:
            formatted.append(f"\\textbf{{{s}}}")
        else:
            formatted.append(s)
    return formatted


def generate_rule_table(
    variant: str,
    caption: str,
    label: str,
) -> str:
    """Generate a LaTeX table for one variant with per-rule solve rates."""

    models = discover_models(variant)
    if not models:
        return f"% No models found for variant '{variant}'"

    # Compute data
    model_data = []
    for base, jsonl_path in models:
        rule_rates = compute_per_model_rule_rates(jsonl_path)
        display = MODEL_DISPLAY_NAMES.get(base, base)
        logo = MODEL_LOGOS.get(base, "")
        model_data.append({
            "base": base,
            "display": display,
            "logo": logo,
            "rules": rule_rates,
        })

    # Determine which rules actually have data (total > 0 for any model)
    active_rules = [
        r for r in RULE_ORDER
        if any(m["rules"][r]["total"] > 0 for m in model_data)
    ]
    n_rules = len(active_rules)

    # Build per-column values for bolding
    col_vals = {
        rule: [m["rules"][rule]["rate"] for m in model_data]
        for rule in active_rules
    }
    col_fmt = {rule: bold_best_column(col_vals[rule]) for rule in active_rules}

    # Also bold best overall accuracy
    overall_rates = []
    for m in model_data:
        total_solved = sum(m["rules"][r]["solved"] for r in active_rules)
        total_puzzles = sum(m["rules"][r]["total"] for r in active_rules)
        # Use total puzzles from any single rule to avoid double-counting
        # Actually, since puzzles can have multiple rules, we need the JSONL count
        # For simplicity, use the per-model JSONL for overall accuracy
        overall_rates.append(None)  # placeholder

    # Table construction
    rule_cols = " c" * n_rules
    lines = [
        r"\begin{table*}[b]",
        r"  \centering",
        r"  \small",
        f"  \\begin{{tabular}}{{cl | {rule_cols}}}",
        r"    \toprule",
    ]

    # Header row
    rule_headers = " & ".join(f"\\textbf{{{r}}}" for r in active_rules)
    lines.append(f"    & \\textbf{{Model}} & {rule_headers} \\\\")
    lines.append(r"    \midrule")

    # Data rows
    for i, m in enumerate(model_data):
        logo_cmd = ""
        if m["logo"]:
            logo_cmd = (
                r"\raisebox{-0.3\height}{"
                r"\includegraphics[width=1.2em]{images/logos/" + m["logo"] + r"}}"
            )

        vals = " & ".join(col_fmt[rule][i] for rule in active_rules)
        lines.append(f"    {logo_cmd} & {m['display']} & {vals} \\\\")

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
    variant_configs = [
        {
            "variant": "sparc",
            "caption": r"SPaRC accuracy (\%) by rule type.",
            "label": "tab:sparc-rule-rates",
            "filename": "sparc_rule_table.tex",
        },
        {
            "variant": "gym",
            "caption": r"Spatial Gym accuracy (\%) by rule type.",
            "label": "tab:sparc-gym-rule-rates",
            "filename": "sparc_gym_rule_table.tex",
        },
        {
            "variant": "tb",
            "caption": r"Spatial Gym with Traceback accuracy (\%) by rule type.",
            "label": "tab:sparc-gym-tb-rule-rates",
            "filename": "sparc_gym_tb_rule_table.tex",
        },
    ]

    out_dir = Path(__file__).parent / "results"
    for cfg in variant_configs:
        tex = generate_rule_table(
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
