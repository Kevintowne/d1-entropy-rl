# rl/ppo_train.py
# Minimal PPO with sample-based entropy penalty using TRL.
# - Uses token-level reward shaping: r_shaped = r_base - lambda(t) * (-logpi)
# - Works with single GPU; multi-GPU via accelerate configs.
#
# Usage:
#   python rl/ppo_train.py --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
#     --ref_model GSAI-ML/LLaDA-8B-Instruct --total_updates 1000 \
#     --entropy_penalty_coef 0.1 --entropy_penalty_schedule linear --entropy_warmup_steps 500
#
# Dataset:
#   - If --dataset_path is provided, loads from disk (datasets.save_to_disk).
#   - Else tries load_dataset("simplescaling/s1K") and uses its "prompt" field.
#
# Notes:
#   - Base reward is set to 0.0 as a placeholder; plug in your reward model at TODO below.

import os
import math
import argparse
from typing import List

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# TRL
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ---------- entropy schedule helpers ----------
def schedule_coef(step: int, max_coef: float, *, total_steps: int | None = None,
                  kind: str = "linear", warmup: int = 0) -> float:
    if max_coef <= 0:
        return 0.0
    if kind == "none":
        return max_coef
    if kind == "linear":
        if warmup <= 0:
            return max_coef
        return max_coef * min(1.0, step / float(warmup))
    if kind == "cosine" and total_steps and total_steps > 0:
        x = min(step, total_steps) / total_steps
        return max_coef * (1.0 - math.cos(math.pi * x)) / 2.0
    return max_coef

# compute logprobs for chosen tokens (responses only)
def gather_logprobs(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                    queries: List[torch.Tensor], responses: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Returns per-token logprobs for the response tokens only (list of 1D tensors).
    """
    device = next(model.parameters()).device
    out_logprobs: List[torch.Tensor] = []

    for q, r in zip(queries, responses):
        # Concatenate prompt + response, then gather logprobs on response tokens
        # q: [Lq], r: [Lr]
        input_ids = torch.cat([q, r], dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[:, :-1, :]                    # [1, Lq+Lr-1, V]
            next_ids = input_ids[:, 1:]                           # [1, Lq+Lr-1]
            logp_all = torch.log_softmax(logits, dim=-1)          # [1, Lq+Lr-1, V]
            logp_chosen = torch.gather(logp_all, -1, next_ids.unsqueeze(-1)).squeeze(-1)  # [1, Lq+Lr-1]
        # Response positions are the last len(r) tokens of the sequence (excluding the first shift)
        Lq, Lr = q.numel(), r.numel()
        # positions aligned to next_ids -> response indices start at (Lq-1) and span Lr tokens
        start = max(Lq - 1, 0)
        resp_logprobs = logp_chosen[0, start:start + Lr]          # [Lr]
        out_logprobs.append(resp_logprobs.detach().cpu())
    return out_logprobs

# ---------- dataset helpers ----------
def load_prompts(dataset_path: str | None):
    if dataset_path and os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        ds = load_dataset("simplescaling/s1K")

    # Try to pick a prompt column; adjust here if your schema differs
    cand_cols = ["prompt", "question", "input", "query"]
    def pick_prompt(ex):
        for k in cand_cols:
            if k in ex and isinstance(ex[k], str) and len(ex[k]) > 0:
                return {"prompt": ex[k]}
        # fallback: stringify whole example
        return {"prompt": str(ex)}

    ds = ds["train"].map(pick_prompt, remove_columns=[c for c in ds["train"].column_names if c != "prompt"])
    prompts = [ex["prompt"] for ex in ds]
    return prompts

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--ref_model", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None, help="load_from_disk path; else uses simplescaling/s1K")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)
    parser.add_argument("--kl_coef", type=float, default=0.02)

    # entropy penalty
    parser.add_argument("--entropy_penalty_coef", type=float, default=0.0)
    parser.add_argument("--entropy_penalty_schedule", type=str, default="linear", choices=["linear", "cosine", "none"])
    parser.add_argument("--entropy_warmup_steps", type=int, default=0)

    parser.add_argument("--save_every", type=int, default=200, help="save model every N updates (0=disable)")
    parser.add_argument("--output_dir", type=str, default="outputs/ppo_entropy")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # better for generation with lm_head

    # models
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    ref_id = args.ref_model or args.model_name_or_path
    ref = AutoModelForCausalLM.from_pretrained(
        ref_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )

    # PPO config
    ppo_config = PPOConfig(
        batch_size=args.per_device_batch_size,
        mini_batch_size=args.per_device_batch_size,     # minimal micro-batch
        gradient_accumulation_steps=args.grad_accum_steps,
        ppo_epochs=1,                                   # small by default
        learning_rate=5e-6,
        init_kl_coef=args.kl_coef,
        target_kl=None,
        adap_kl_ctrl=False,
        seed=args.seed,
        log_with=None,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref,
        tokenizer=tok,
        dataset=None,       # we drive the loop manually
    )

    # dataset -> prompts
    prompts = load_prompts(args.dataset_path)

    # generation config
    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # training loop
    global_step = 0
    from tqdm import trange
    for update in trange(args.total_updates, desc="PPO updates"):
        # 1) sample a batch of prompts
        batch_prompts = prompts[(update * args.per_device_batch_size) % len(prompts) :
                                (update + 1) * args.per_device_batch_size % len(prompts) or None]
        if len(batch_prompts) < args.per_device_batch_size:
            batch_prompts += prompts[: args.per_device_batch_size - len(batch_prompts)]

        # 2) tokenize queries
        batch_queries = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_prompt_len).to(policy.device)
        query_tensors = [t for t in batch_queries["input_ids"]]

        # 3) generate responses
        with torch.no_grad():
            response_tensors = trainer.generate(query_tensors, **gen_cfg.to_dict())
        # 4) compute per-token logprobs on responses (policy)
        resp_logprobs_list = gather_logprobs(policy.pretrained_model, tok, query_tensors, response_tensors)

        # 5) base rewards (TODO: replace with your reward model / preference score)
        base_rewards = [torch.zeros_like(lp) for lp in resp_logprobs_list]

        # 6) entropy penalty (token-level): - lambda(t) * (-logpi)
        coef = schedule_coef(global_step, args.entropy_penalty_coef,
                             total_steps=args.total_updates,
                             kind=args.entropy_penalty_schedule,
                             warmup=args.entropy_warmup_steps)
        shaped_rewards = [br - coef * (-lp) for br, lp in zip(base_rewards, resp_logprobs_list)]

        # 7) PPO step (TRL expects lists of tensors)
        stats = trainer.step(query_tensors, response_tensors, shaped_rewards)

        # 8) logging (simple prints)
        with torch.no_grad():
            # sample-based entropy estimate per sequence = mean(-logpi)
            ent_vals = [(-lp).mean().item() for lp in resp_logprobs_list]
            print(f"[upd {update}] coef={coef:.4f} | entropy_est={sum(ent_vals)/len(ent_vals):.3f} | "
                  f"kl={float(stats.get('objective/kl', 0.0)):.4f}")

        global_step += 1

        # 9) save
        if args.save_every > 0 and (update + 1) % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{update+1}")
            trainer.model.save_pretrained(save_dir, safe_serialization=True)
            tok.save_pretrained(save_dir)

    # final save
    final_dir = os.path.join(args.output_dir, "final")
    trainer.model.save_pretrained(final_dir, safe_serialization=True)
    tok.save_pretrained(final_dir)
    print("Saved:", final_dir)

if __name__ == "__main__":
    main()
