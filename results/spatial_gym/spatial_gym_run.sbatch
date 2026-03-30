#!/bin/bash
#SBATCH --job-name=sparc
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --output=logs/%x-%j.out
#SBATCH -C inet&80gb

set -euo pipefail

echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"

###############################################################################
# Config
###############################################################################
MODEL="${MODEL:-mistralai/Magistral-Small-2507}"
PORT="${LLM_PORT:-8000}"
TP_SIZE="${SLURM_GPUS_ON_NODE}"
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"
HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
SIF="/scratch-scc/projects/ag_gipp/vllm.sif"
#SIF="/user/kaesberg1/u16096/vllm-gptoss.sif"
mkdir -p "$HF_HOME" logs

###############################################################################
# Modules
###############################################################################
module load gcc/13.2.0 cuda/12.6.2 apptainer

###############################################################################
# Launch vLLM (background)
###############################################################################
apptainer exec \
  --nv \
  --cleanenv \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
    > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

###############################################################################
# Wait for server
###############################################################################
echo -n "⌛ Waiting for vLLM"
for _ in {1..3600}; do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo " ready."
    break
  fi
  sleep 1
  echo -n "."
done

###############################################################################
# Run sparc
###############################################################################
source activate vllm

export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-LOCALHOST_ONLY}"

sparc \
  --api-key "$OPENAI_API_KEY" \
  --base-url "$OPENAI_API_BASE" \
  --model "$MODEL" \
  --batch-size 20 \
  --run-name "" \
  --gym \
  --timeout 1200 \
  "$@"

echo "[$(date +%F\ %T)] Job complete."

