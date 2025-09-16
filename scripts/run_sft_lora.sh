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

# accelerate config (point to multi-gpu config but OK for single-node)
ACC=${ACC:-$ROOT/accelerate/ddp_4gpu.yaml}
if [ ! -f "$ACC" ]; then
  echo "[WARN] accelerate config not found at $ACC. Make sure to create it or pass --config_file to accelerate."
fi

# MODEL: prefer local cache folder; else HF id
MODEL=${MODEL:-$ROOT/cache/models/LLaDA-8B-Instruct}
[ -d "$MODEL" ] || MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}

# LoRA output dir
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/sft/lora}
mkdir -p "$OUTPUT_DIR"

# knobs (override via env)
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GA=${GA:-1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-4}
MAX_LEN=${MAX_LEN:-2048}

# optional force 8-bit (1 to force)
FORCE_8BIT=${FORCE_8BIT:-0}
export FORCE_8BIT

# launch (use accelerate)
python -m accelerate.commands.launch \
  --config_file "$ACC" \
  --num_processes "$NUM_PROC" \
  vendor/d1/SFT/sft_train_lora.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$PER_DEVICE_BS" \
  --gradient_accumulation_steps "$GA" \
  --num_train_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --max_length "$MAX_LEN" \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --bf16 true \
  "${EXTRA[@]:-}"
