#!/usr/bin/env python3
"""
Script to regenerate all plots for the SPaRC-Gym analysis.
"""
import subprocess
import sys
from pathlib import Path
import time

# List of all plot scripts (excluding plot_config.py which is just configuration)
PLOT_SCRIPTS = [
    "plot_accuracy.py",
    "plot_baseline_comparison.py",
    "plot_traceback_diff.py",
    "plot_difficulty_comparison.py",
    "plot_sparc_gym_comparison.py",
    "plot_combined_comparison.py",
    "plot_difficulty_vs_steps.py",
    "plot_traceback_steps_vs_path.py",
    "plot_correlation_heatmap.py",
    "plot_improvement_ceiling.py",
    "plot_navigation_outcome.py",
    "plot_solve_rate_by_rule.py",
    "plot_model_ranking_bump.py",
    "plot_model_agreement.py",
    "plot_qwen_scaling.py",
    "plot_reasoning_comparison.py",
    "plot_token_analysis.py",
    "plot_token_by_difficulty.py",
    "plot_vision_comparison.py",
    "plot_token_by_difficulty_per_model.py",
]


def run_script(script_name):
    """Run a single plot script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully ({elapsed:.1f}s)")
            return True
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ {script_name} failed with error: {e}")
        return False


def main():
    print("=" * 60)
    print("REGENERATING ALL PLOTS")
    print("=" * 60)
    
    total_start = time.time()
    
    successful = []
    failed = []
    
    for script in PLOT_SCRIPTS:
        script_path = Path(__file__).parent / script
        if not script_path.exists():
            print(f"\n⚠ Skipping {script} (file not found)")
            continue
        
        if run_script(script):
            successful.append(script)
        else:
            failed.append(script)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Successful: {len(successful)}/{len(successful) + len(failed)}")
    
    if successful:
        print(f"\n✓ Completed scripts:")
        for s in successful:
            print(f"    {s}")
    
    if failed:
        print(f"\n✗ Failed scripts:")
        for s in failed:
            print(f"    {s}")
        sys.exit(1)
    
    print("\n✓ All plots regenerated successfully!")


if __name__ == "__main__":
    main()
