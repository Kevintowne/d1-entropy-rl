#!/usr/bin/env bash
set -euo pipefail

# accelerate 配置：单卡可用 accelerate/ddp_single.yaml；多卡可传 ACC=accelerate/ddp_multi2.yaml
ACC=${ACC:-accelerate/ddp_single.yaml}

# 模型与参考（可用本地路径或 HF ID）
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
REF=${REF:-outputs/sft_ckpt}

# d1 路径与 GRPO 训练脚本路径（按你的实际仓库结构调整）
D1_DIR=${D1_DIR:-vendor/d1}
GRPO_PY=${GRPO_PY:-$PWD/rl/grpo_train.py}

# 熵惩罚（policy entropy reduction）
ENTROPY_PENALTY_COEF=${ENTROPY_PENALTY_COEF:-0.1}
ENTROPY_PENALTY_SCHEDULE=${ENTROPY_PENALTY_SCHEDULE:-linear}
ENTROPY_WARMUP_STEPS=${ENTROPY_WARMUP_STEPS:-500}

# 基本健壮性检查
if [ ! -f "$GRPO_PY" ]; then
  echo "[ERR] GRPO trainer not found at: $GRPO_PY"
  echo "      请把 GRPO_PY 指到你的 d1 中 grpo 训练脚本，例如："
  echo "      export GRPO_PY=\$D1_DIR/RL/grpo_train.py"
  exit 1
fi

python -m accelerate.commands.launch \
  --config_file "$ACC" \
  --num_processes 1 \
  "$GRPO_PY" \
  --model_name_or_path "$MODEL" \
  --ref_model "$REF" \
  --per_device_batch_size 1 \
  --grad_accum_steps 8 \
  --entropy_penalty_coef "$ENTROPY_PENALTY_COEF" \
  --entropy_penalty_schedule "$ENTROPY_PENALTY_SCHEDULE" \
  --entropy_warmup_steps "$ENTROPY_WARMUP_STEPS" \
  --gen_max_new_tokens 512 \
  --total_updates 1000
