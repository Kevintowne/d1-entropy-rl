#!/usr/bin/env bash
set -euo pipefail
EXTRA=()
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# number of processes (GPUs) to use on this single node (labmate will set NUM_PROC=4)
NUM_PROC=${NUM_PROC:-1}
export WORLD_SIZE=${WORLD_SIZE:-$NUM_PROC}

# recommended env tweaks
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# dataset (offline-friendly)
DATASET_PATH=${DATASET_PATH:-$ROOT/cache/datasets/s1K}
EXTRA=()
[ -d "$DATASET_PATH" ] && EXTRA+=(--dataset_path "$DATASET_PATH") || \
  echo "[WARN] No local dataset at $DATASET_PATH; will try Hub (needs internet)."

# accelerate config (default points to multi-gpu config)
ACC=${ACC:-$ROOT/accelerate/ddp_4gpu.yaml}
if [ ! -f "$ACC" ]; then
  echo "[WARN] accelerate config not found at $ACC. Make sure to create it or pass --config_file to accelerate."
fi

# MODEL: prefer local cache folder; else HF id
MODEL=${MODEL:-$ROOT/cache/models/LLaDA-8B-Instruct}
[ -d "$MODEL" ] || MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}

# REF: SFT final if exists; else MODEL
REF=${REF:-$ROOT/outputs/sft/lora/final}
[ -e "$REF" ] || REF="$MODEL"

PPO_PY=${PPO_PY:-$ROOT/rl/ppo_train.py}
[ -f "$PPO_PY" ] || { echo "[ERR] PPO trainer not found at $PPO_PY"; exit 1; }

# knobs (override via env)
PER_DEV_BS=${PER_DEV_BS:-1}
GA=${GA:-1}
MAX_NEW=${MAX_NEW:-128}
TOTAL_UPDATES=${TOTAL_UPDATES:-3}
ENTROPY_PENALTY_COEF=${ENTROPY_PENALTY_COEF:-0.1}
ENTROPY_PENALTY_SCHEDULE=${ENTROPY_PENALTY_SCHEDULE:-linear}
ENTROPY_WARMUP_STEPS=${ENTROPY_WARMUP_STEPS:-1}
KL_COEF=${KL_COEF:-0.02}

# optional force 8-bit
FORCE_8BIT=${FORCE_8BIT:-0}
export FORCE_8BIT


# launch via accelerate (multi-GPU aware)
python -m accelerate.commands.launch \
  --config_file "$ACC" \
  --num_processes "$NUM_PROC" \
  --num_machines 1 \
  "$PPO_PY" \
  --model_name_or_path "$MODEL" \
  --ref_model "$REF" \
  --per_device_batch_size "$PER_DEV_BS" \
  --grad_accum_steps "$GA" \
  --kl_coef "$KL_COEF" \
  --entropy_penalty_coef "$ENTROPY_PENALTY_COEF" \
  --entropy_penalty_schedule "$ENTROPY_PENALTY_SCHEDULE" \
  --entropy_warmup_steps "$ENTROPY_WARMUP_STEPS" \
  --gen_max_new_tokens "$MAX_NEW" \
  --total_updates "$TOTAL_UPDATES" \
  "${EXTRA[@]:-}"

