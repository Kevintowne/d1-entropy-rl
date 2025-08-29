#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# where to save & which model to fetch
model_id = os.environ.get("MODEL_ID", "GSAI-ML/LLaDA-8B-Instruct")
out = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "cache", "models", "LLaDA-8B-Instruct"))
out = os.path.abspath(out)

os.makedirs(out, exist_ok=True)
print(f"[CACHE] downloading {model_id} -> {out}")
snapshot_download(
    model_id,
    local_dir=out,
    local_dir_use_symlinks=False,
    repo_type="model",
)
print("[CACHE] done:", out)

