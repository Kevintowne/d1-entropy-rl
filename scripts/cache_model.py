#!/usr/bin/env python3
"""
scripts/cache_model.py

Download a Hugging Face model repo snapshot into a local folder for offline use.

Example:
  python scripts/cache_model.py \
    --repo_id GSAI-ML/LLaDA-8B-Instruct \
    --out_dir /mnt/cache/models/LLaDA-8B-Instruct \
    --revision main \
    --use_auth_token YOUR_HF_TOKEN
"""
import argparse
import os
import shutil
from huggingface_hub import snapshot_download

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True, help="HF repo id (e.g. GSAI-ML/LLaDA-8B-Instruct)")
    p.add_argument("--out_dir", required=False, default=None,
                   help="Target folder to place the snapshot. Defaults to ./cache/models/<repo_id>")
    p.add_argument("--cache_parent", required=False, default=None,
                   help="Parent cache dir to pass to snapshot_download. If omitted, HF cache is used.")
    p.add_argument("--revision", required=False, default=None, help="Repo revision/branch/commit")
    p.add_argument("--use_auth_token", required=False, default=None, help="Hugging Face token (if private)")
    p.add_argument("--allow_patterns", nargs="*", default=None,
                   help="Optional allow_patterns for snapshot_download (e.g. '*.json', '*.safetensors')")
    p.add_argument("--force", action="store_true", help="Overwrite out_dir if it exists")
    return p.parse_args()

def main():
    args = parse_args()
    repo_id = args.repo_id
    if args.out_dir is None:
        # default relative to current repo
        repo_name = repo_id.split("/")[-1]
        default_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache", "models"))
        os.makedirs(default_parent, exist_ok=True)
        out_dir = os.path.join(default_parent, repo_name)
    else:
        out_dir = os.path.abspath(args.out_dir)

    if os.path.exists(out_dir) and not args.force:
        print(f"[INFO] target out_dir already exists: {out_dir}")
        print("Use --force to overwrite or remove the directory manually if you want a fresh copy.")
        return

    print(f"[INFO] Downloading snapshot for '{repo_id}' -> temporary cache (may take time)...")
    # snapshot_download returns the path to the downloaded snapshot dir containing model files
    try:
        snapshot_path = snapshot_download(
            repo_id,
            cache_dir=args.cache_parent,
            revision=args.revision,
            allow_patterns=args.allow_patterns,
            token=args.use_auth_token,
            repo_type="model",
        )
    except Exception as e:
        print("[ERR] snapshot_download failed:", e)
        raise

    print(f"[INFO] snapshot downloaded to: {snapshot_path}")

    # If out_dir exists and force is set, remove it first
    if os.path.exists(out_dir):
        if args.force:
            shutil.rmtree(out_dir)
        else:
            print(f"[ERR] out_dir {out_dir} exists. Use --force to overwrite.")
            return

    # Move (or copy) the snapshot to out_dir
    print(f"[INFO] Moving snapshot to final out_dir: {out_dir} (this may take a while)")
    shutil.move(snapshot_path, out_dir)
    print(f"[OK] Model files are now at: {out_dir}")

    # quick sanity checks
    expected_files = ["config.json"]
    has_any = any(os.path.exists(os.path.join(out_dir, fn)) for fn in ["pytorch_model.bin", "pytorch_model.bin.index.json", "safetensors", "tf_model.h5"])
    if not os.path.exists(os.path.join(out_dir, "config.json")):
        print("[WARN] config.json not found in out_dir - the snapshot may be incomplete.")
    if not has_any:
        print("[WARN] no obvious weight files found (pytorch_model.bin / .index.json / safetensors); check contents manually.")

    print("\nNext steps / tips:")
    print(f" - Point your training script to --model_name_or_path '{out_dir}'")
    print(" - To share this cached model with others, tar it (tar -czf LLaDA-8B-Instruct.tar.gz -C <parent> <dir>) and copy via scp/rsync.")
    print(" - If the repo is private, ensure other users have HF tokens with repo access or share the tarball.")
    print(" - If disk space is a concern, consider quantized versions or using load_in_8bit at runtime (requires bitsandbytes).")

if __name__ == '__main__':
    main()
