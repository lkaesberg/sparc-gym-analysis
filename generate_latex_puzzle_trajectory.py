#!/usr/bin/env python3
"""
Replay a Spatial Gym episode from JSONL using the Spatial Gym environment package, export renderer PNGs,
and emit a LaTeX fragment for paper inclusion.

Dependencies (install in the active environment)::
    pip install spatial-gym gymnasium pygame

The Hugging Face dataset ``lkaesberg/SPaRC`` is downloaded on first use.

Headless servers: set ``SDL_VIDEODRIVER=dummy`` before running so pygame does not
need a display (Linux; on macOS you may need a real video driver / X11).

Environment id: ``Spatial-Gym`` (see ``spatial_gym.register_env``).

Example (defaults write to ``examples/puzzle1/``)::
    python generate_latex_puzzle_trajectory.py

Override paths::
    python generate_latex_puzzle_trajectory.py \\
        --jsonl results/spatial_gym/openai_gpt-oss-120b_gym.jsonl \\
        --puzzle-id c2f1726c32030b96 \\
        --out-dir examples/puzzle1/figures \\
        --tex-out examples/puzzle1/trajectory_fragment.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_JSONL = Path(__file__).resolve().parent / "results" / "spatial_gym" / "openai_gpt-oss-120b_gym.jsonl"
# Solved GPT-OSS gym run with >=5 actions; fewest steps (5) among those in the current JSONL.
DEFAULT_PUZZLE_ID = "c2f1726c32030b96"
ENV_ID = "Spatial-Gym"

# Default LaTeX example bundle (see examples/puzzle1/puzzle1.tex)
EXAMPLE_PUZZLE1_DIR = Path(__file__).resolve().parent / "examples" / "puzzle1"

# Spatial Gym discrete action meanings (see upstream README)
ACTION_NAMES = ("Right", "Up", "Left", "Down")


def load_episode(jsonl_path: Path, puzzle_id: str) -> dict:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") == puzzle_id:
                return row
    raise FileNotFoundError(
        f"No row with id={puzzle_id!r} in {jsonl_path}. "
        "Check the file path and puzzle id (must exist in the dataset split)."
    )


# GPT-OSS / Harmony-style reasoning segments in gym logs
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/" + "think" + chr(62)


def extract_thoughts(msg: str) -> str:
    """Strip ``think`` XML wrappers; keep inner reasoning only."""
    if not msg:
        return ""
    s = msg.strip()
    if s.startswith(_THINK_OPEN):
        s = s[len(_THINK_OPEN) :].lstrip()
    if _THINK_CLOSE in s:
        s = s.split(_THINK_CLOSE, 1)[0].strip()
    return s.strip()


_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}


def latex_escape(text: str) -> str:
    """Escape special LaTeX characters for use in \texttt{...}."""
    out = []
    for ch in text:
        out.append(_LATEX_ESCAPE_MAP.get(ch, ch))
    return "".join(out)


def latex_format_paragraph(text: str) -> str:
    """Escape text and turn newlines into LaTeX breaks inside a table cell."""
    e = latex_escape(text)
    chunks = e.split("\n\n")
    inner = r"\par ".join(
        c.replace("\n", r"\newline ").strip() for c in chunks if c.strip()
    )
    return inner


def _get_human_screen(env) -> "object":
    e = env.unwrapped
    hr = getattr(e, "human_renderer", None)
    if hr is None or getattr(hr, "screen", None) is None:
        raise RuntimeError(
            "Could not access human_renderer.screen after render(). "
            "Ensure render_mode='human' and Spatial Gym exposes HumanRenderer."
        )
    return hr.screen


def replay_and_save_frames(
    puzzle_id: str,
    actions: list[int],
    out_dir: Path,
) -> list[Path]:
    """Reset env, replay actions, save step_00.png .. step_N.png (N = len(actions))."""
    import pygame  # noqa: F401 — imported after env may init SDL
    import gymnasium as gym
    env_id = ENV_ID
    try:
        import spatial_gym  # noqa: F401 — register env
    except ImportError:
        import SPaRC_Gym  # noqa: F401 — legacy package name
        env_id = "SPaRC-Gym"

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    env = gym.make(
        env_id,
        df_name="lkaesberg/SPaRC",
        df_split="all",
        df_set="test",
        render_mode="human",
        observation="new",
        traceback=False,
        max_steps=1000,
    )
    try:
        env.reset(options={"puzzle_id": puzzle_id})
        env.render()
        screen = _get_human_screen(env)
        p0 = out_dir / "step_00.png"
        pygame.image.save(screen, str(p0))
        paths.append(p0)

        for i, action in enumerate(actions, start=1):
            env.step(int(action))
            env.render()
            screen = _get_human_screen(env)
            pi = out_dir / f"step_{i:02d}.png"
            pygame.image.save(screen, str(pi))
            paths.append(pi)
    finally:
        env.close()

    return paths


def _action_label(code: int) -> str:
    if 0 <= code < len(ACTION_NAMES):
        return ACTION_NAMES[code]
    return f"unknown({code})"


def write_latex_fragment(
    tex_path: Path,
    episode_row: dict,
    image_paths: list[Path],
    messages: list[str],
    actions: list[int],
    wrap_width: str,
    wrap_side: str,
    final_img_width: str,
    rel_to_tex: bool = True,
) -> None:
    """
    Metadata header + per-step: step index, chosen action, frame name, wrapfigure + reasoning.
    Optional centered final frame when there is one more image than messages.
    """
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    puzzle_id = str(episode_row.get("id", ""))
    result = episode_row.get("result") or {}
    gs = episode_row.get("grid_size") or {}
    gh, gw = gs.get("height"), gs.get("width")
    diff = episode_row.get("difficulty_level")
    diff_score = episode_row.get("difficulty_score")
    solved = result.get("solved")
    steps_taken = result.get("steps_taken")

    def rel(p: Path) -> str:
        rp = p.resolve()
        if rel_to_tex:
            try:
                return str(rp.relative_to(tex_path.parent.resolve()))
            except ValueError:
                pass
        return str(rp)

    n_msg = len(messages)
    n_img = len(image_paths)
    n_act = len(actions)

    em = r"\textemdash{}"
    grid_txt = em
    if gh is not None and gw is not None:
        grid_txt = f"{gh}$\\times${gw}"

    diff_txt = em
    if diff is not None and diff_score is not None:
        diff_txt = f"D{diff} (score {diff_score:.2f})"
    elif diff is not None:
        diff_txt = f"D{diff}"
    elif diff_score is not None:
        diff_txt = f"score {diff_score:.2f}"

    outcome = em
    if solved is True:
        outcome = r"Solved"
    elif solved is False:
        outcome = r"Not solved"

    lines: list[str] = [
        r"% Auto-generated by generate_latex_puzzle_trajectory.py",
        r"% Puzzle id: " + latex_escape(puzzle_id),
        r"% Preamble: \usepackage{graphicx,wrapfig}",
        r"",
        r"\begin{center}",
        r"{\large\bfseries Spatial Gym trajectory}",
        r"\\[0.35em]",
        r"{\small GPT-OSS 120B (\texttt{openai\_gpt-oss-120b\_gym.jsonl})}",
        r"\\[0.25em]",
        r"{\ttfamily " + latex_escape(puzzle_id) + r"}",
        r"\end{center}",
        r"\vspace{0.6em}",
        r"\noindent\small",
        r"\begin{tabular}{@{}p{0.22\linewidth}p{0.72\linewidth}@{}}",
        r"\textbf{Difficulty} & " + diff_txt + r" \\",
        r"\textbf{Grid (H$\times$W)} & " + grid_txt + r" \\",
        r"\textbf{Outcome} & " + outcome + r" \\",
    ]
    if steps_taken is not None:
        lines.append(
            r"\textbf{Logged steps} & "
            + latex_escape(str(steps_taken))
            + r" (environment) \\"
        )
    lines.append(
        r"\textbf{Trajectory} & "
        + str(n_act)
        + r" actions, "
        + str(n_img)
        + r" frames (first frame = state before action 1). \\"
    )
    lines.append(r"\end{tabular}")
    lines.append(r"\par\vspace{1em}")

    for i in range(n_msg):
        ip = image_paths[i] if i < n_img else image_paths[-1]
        thought = latex_format_paragraph(extract_thoughts(messages[i]))
        ac = int(actions[i]) if i < len(actions) else -1
        an = _action_label(ac)
        if i > 0:
            lines.append(r"\par\bigskip")
        lines.append(r"\noindent\rule{\linewidth}{0.4pt}\par\medskip")
        lines.append(
            r"\noindent\textbf{Step "
            + str(i + 1)
            + r" of "
            + str(n_act)
            + r"} \quad {\small\textit{(reasoning below $\rightarrow$ then action "
            + str(ac)
            + r")}}"
        )
        lines.append(r"\par\smallskip")
        lines.append(
            r"\noindent\textbf{Chosen action:} \textbf{"
            + an
            + r"} (code \texttt{"
            + str(ac)
            + r"})"
        )
        lines.append(r"\par\medskip")
        lines.append(r"\begin{wrapfigure}{" + wrap_side + "}{" + wrap_width + "}")
        lines.append(r"\centering")
        lines.append(r"\vspace{0pt}")
        lines.append(
            r"\includegraphics[width=\linewidth]{"
            + rel(ip).replace("\\", "/")
            + "}"
        )
        lines.append(r"\end{wrapfigure}")
        lines.append(r"\noindent\textbf{Model reasoning}\par\smallskip")
        lines.append(
            r"\noindent{\raggedright\footnotesize\ttfamily "
            + thought
            + r"\par}"
        )

    if n_img > n_msg and n_msg > 0:
        ip = image_paths[-1]
        last_ac = int(actions[-1]) if actions else -1
        lines.append(r"\par\bigskip")
        lines.append(r"\noindent\rule{\linewidth}{0.4pt}\par\medskip")
        lines.append(
            r"\noindent\textbf{Final frame} \quad {\small\textit{(after last move: \textbf{"
            + _action_label(last_ac)
            + r"}, code "
            + str(last_ac)
            + r")}}"
        )
        lines.append(r"\par\medskip")
        lines.append(r"\begin{center}")
        lines.append(
            r"\includegraphics[width="
            + final_img_width
            + r"]{"
            + rel(ip).replace("\\", "/")
            + "}"
        )
        lines.append(r"\\[0.5em]")
        lines.append(r"{\small\itshape terminal state}")
        lines.append(r"\end{center}")

    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Path to gym JSONL (default: GPT-OSS 120B gym)",
    )
    parser.add_argument(
        "--puzzle-id",
        default=DEFAULT_PUZZLE_ID,
        help="Puzzle id to replay (default: shortest-path candidate)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: examples/puzzle1/figures)",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=None,
        help="Output .tex fragment path (default: examples/puzzle1/trajectory_fragment.tex)",
    )
    parser.add_argument(
        "--wrap-width",
        default=r"0.26\textwidth",
        help=r"Width of each wrapfigure box; image uses \linewidth inside it (default: 0.26\textwidth)",
    )
    parser.add_argument(
        "--wrap-side",
        choices=("l", "r"),
        default="l",
        help="Wrapfigure placement: l = image on left (text wraps right), r = image on right",
    )
    parser.add_argument(
        "--final-img-width",
        default=r"0.34\textwidth",
        help=r"Width of the optional centered final-state image (default: 0.34\textwidth)",
    )
    parser.add_argument(
        "--img-width",
        default=None,
        help="Deprecated: same as --wrap-width",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    out_dir = args.out_dir or (EXAMPLE_PUZZLE1_DIR / "figures")
    tex_out = args.tex_out or (EXAMPLE_PUZZLE1_DIR / "trajectory_fragment.tex")

    row = load_episode(args.jsonl, args.puzzle_id)
    result = row.get("result") or {}
    actions = result.get("actions") or []
    messages = result.get("message") or []

    if not actions:
        print("Error: no actions in result; cannot replay.", file=sys.stderr)
        return 1
    if len(messages) != len(actions):
        print(
            f"Warning: len(message)={len(messages)} != len(actions)={len(actions)}; "
            "truncating/padding may be wrong.",
            file=sys.stderr,
        )

    print(f"Replaying puzzle {args.puzzle_id} ({len(actions)} actions)…")
    image_paths = replay_and_save_frames(args.puzzle_id, actions, out_dir)
    print(f"Wrote {len(image_paths)} frames to {out_dir}")

    wrap_w = args.img_width if args.img_width else args.wrap_width
    write_latex_fragment(
        tex_out,
        row,
        image_paths,
        messages,
        actions,
        wrap_width=wrap_w,
        wrap_side=args.wrap_side,
        final_img_width=args.final_img_width,
        rel_to_tex=True,
    )
    print(f"Wrote LaTeX fragment: {tex_out}")
    print()
    print("In your paper preamble, use: \\usepackage{graphicx,wrapfig}")
    print("Input the fragment where appropriate (\\input{...}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
