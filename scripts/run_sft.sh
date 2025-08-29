#!/usr/bin/env bash
set -euo pipefail

# repo root
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# accelerate config
ACC=${ACC:-$ROOT/accelerate/ddp_single.yaml}

# base model: prefer local cache; else HF id
MODEL=${MODEL:-${OUT:-$ROOT/cache/models/LLaDA-8B-Instruct}}
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}

# vendored d1 path & SFT entry
D1_DIR=${D1_DIR:-$ROOT/vendor/d1}
SFT_PY="$D1_DIR/SFT/sft_train.py"
[ -f "$SFT_PY" ] || { echo "[ERR] $SFT_PY not found"; exit 1; }

# SFT outputs live here
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/sft}
mkdir -p "$OUTPUT_DIR"

SAVE_ARGS="--save_strategy epoch --save_total_limit 1 --save_safetensors true"

# launch SFT
python -m accelerate.commands.launch \
  --config_file "$ACC" \
  --num_processes 1 \
  "$SFT_PY" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 1 --grad_accum_steps 8 --num_epochs 20 \
  $SAVE_ARGS

# symlink latest checkpoint as ./outputs/sft/final  (so REF can point here)
latest_ckpt="$(ls -dt "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -n1 || true)"
if [ -n "$latest_ckpt" ]; then
  ln -sfn "$(basename "$latest_ckpt")" "$OUTPUT_DIR/final"
  echo "[INFO] REF -> $OUTPUT_DIR/final"
else
  echo "[WARN] No checkpoint-* found under $OUTPUT_DIR"
fi
