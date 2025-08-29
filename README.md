# d1-entropy-rl

SFT → PPO/GRPO on top of d1, with **policy entropy reduction** (sample-based entropy penalty + schedule).
This repo includes minimal scripts & configs to reproduce SFT and RL with entropy control.

## Environment
```bash
conda env create -f environment.yml
conda activate d1
bash scripts/verify_env.sh


# choose local cache dir (no need to edit scripts later)
export OUT=$PWD/cache/models/LLaDA-8B-Instruct
export MODEL_ID=GSAI-ML/LLaDA-8B-Instruct

# (optional, for speed / access)
# export HF_HUB_ENABLE_HF_TRANSFER=1
# export HUGGINGFACE_HUB_TOKEN=...         # if gated
# export HF_ENDPOINT=https://hf-mirror.com # if needed

python scripts/cache_model.py   # downloads to $OUT


# uses vendored d1 at vendor/d1, outputs to ./outputs/sft/
bash scripts/run_sft.sh

# defaults: MODEL=./cache/models/LLaDA-8B-Instruct, REF=./outputs/sft/final (if exists)
bash scripts/run_rl_ppo.sh
# bash scripts/run_rl_grpo.sh

# python eval/quick_eval.py --model_path outputs/ppo_entropy/final \
#   --input "Explain PPO vs GRPO in one paragraph." --use_chat_template


