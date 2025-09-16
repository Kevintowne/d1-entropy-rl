## 1 Create venv & install deps
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r environment.txt
#### if bitsandbytes fails with pip, follow bitsandbytes install docs for your CUDA / driver

## 2 Pre-cache models and dataset(or run locally):
#### this script should pull model to ./cache/models/<model> so runs are offline-friendly
python scripts/cache_model.py --model_id GSAI-ML/LLaDA-8B-Instruct --target_dir ./cache/models/LLaDA-8B-Instruct

# from repo root, this should pull dataset to ./cache/datasets/s1K
source .venv/bin/activate
OUT=./cache/datasets/s1K python scripts/cache_s1k.py

## 3 Smoke-test LoRA SFT (single GPU)
#### single-GPU smoke run
NUM_PROC=1 SMOKE_TEST=1 PER_DEV_BS=1 GRAD_ACCUM=1 bash scripts/run_sft_lora.sh
# run SFT on 8 GPUs (single node)
NUM_PROC=8 SMOKE_TEST=1 PER_DEV_BS=1 GRAD_ACCUM=1 bash scripts/run_sft_lora.sh
#### or if permission issues:
NUM_PROC=1 SMOKE_TEST=1 PER_DEV_BS=1 GRAD_ACCUM=1 bash scripts/run_sft_lora.sh


## 4 Smoke-test PPO (single GPU):
#### force 8-bit to reduce mem; short generation
FORCE_8BIT=1 NUM_PROC=1 PER_DEV_BS=1 GA=1 TOTAL_UPDATES=3 MAX_NEW=32 bash scripts/run_rl_ppo.sh

## 5 If single-GPU smoke passes, scale to 4×A100:
#### make sure accelerate config ddp_4gpu.yaml points to 4 processes per node and uses `mixed_precision: bf16` ideally
NUM_PROC=4 PER_DEV_BS=1 GA=1 TOTAL_UPDATES=100 bash scripts/run_rl_ppo.sh
#### or if you need forced 8-bit:
FORCE_8BIT=1 NUM_PROC=4 PER_DEV_BS=1 GA=1 TOTAL_UPDATES=100 bash scripts/run_rl_ppo.sh
