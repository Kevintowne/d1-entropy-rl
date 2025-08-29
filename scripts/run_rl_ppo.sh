#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ACC=${ACC:-$ROOT/accelerate/ddp_single.yaml}

# MODEL: prefer local cache folder; else fall back to HF id
MODEL=${MODEL:-$ROOT/cache/models/LLaDA-8B-Instruct}
[ -d "$MODEL" ] || MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}

# REF: default to SFT final symlink; if missing, fall back to MODEL
REF=${REF:-$ROOT/outputs/sft/final}
[ -e "$REF" ] || REF="$MODEL"

# bundled PPO trainer by default
PPO_PY=${PPO_PY:-$ROOT/rl/ppo_train.py}
[ -f "$PPO_PY" ] || { echo "[ERR] PPO trainer not found at $PPO_PY"; exit 1; }

# entropy reduction knobs
ENTROPY_PENALTY_COEF=${ENTROPY_PENALTY_COEF:-0.1}
ENTROPY_PENALTY_SCHEDULE=${ENTROPY_PENALTY_SCHEDULE:-linear}
ENTROPY_WARMUP_STEPS=${ENTROPY_WARMUP_STEPS:-500}

python -m accelerate.commands.launch \
  --config_file "$ACC" \
  --num_processes 1 \
  "$PPO_PY" \
  --model_name_or_path "$MODEL" \
  --ref_model "$REF" \
  --per_device_batch_size 1 --grad_accum_steps 8 \
  --kl_coef 0.02 \
  --entropy_penalty_coef "$ENTROPY_PENALTY_COEF" \
  --entropy_penalty_schedule "$ENTROPY_PENALTY_SCHEDULE" \
  --entropy_warmup_steps "$ENTROPY_WARMUP_STEPS" \
  --gen_max_new_tokens 512 --total_updates 1000

