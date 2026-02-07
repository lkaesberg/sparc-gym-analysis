#!/bin/bash
set -euo pipefail

# Base port for models; will increment for each job
BASE_PORT=8000

# List of models to run
models=(
  "Qwen/Qwen3-32B"
  "allenai/Olmo-3.1-32B-Think"
  "openai/gpt-oss-120b"
  "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "mistralai/Magistral-Small-2507"
  "google/gemma-3-27b-it"
)

# Path to your SLURM script template
SBATCH_SCRIPT="sparc_run.sbatch"

for i in "${!models[@]}"; do
  model="${models[$i]}"
  port=$((BASE_PORT + i))

  # convert slash and special chars for SLURM job name
  jobname=$(echo "$model" | tr '/_.' '-' | tr '[:upper:]' '[:lower:]')

  echo "Submitting $model on port $port as job: $jobname"

  sbatch --export=MODEL="$model",LLM_PORT="$port" \
         --job-name="$jobname" \
         "$SBATCH_SCRIPT"
done

