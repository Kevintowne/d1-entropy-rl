# rl/grpo_train.py
# Minimal GRPO-style training with group sampling, reward-weighted sequence NLL,
# KL to a reference model, and sample-based entropy penalty.
#
# Usage (example):
#   python rl/grpo_train.py \
#     --model_name_or_path /path/to/model \
#     --ref_model /path/to/ref_or_same \
#     --total_updates 1000 --group_size 4 --per_device_batch_size 1 \
#     --entropy_penalty_coef 0.1 --entropy_penalty_schedule linear --entropy_warmup_steps 500
#
# Dataset:
#   - If --dataset_path is provided, uses datasets.load_from_disk(path).
#   - Else uses "simplescaling/s1K" train split; pick "prompt"/"question"/"input" as prompt.
#
# Notes:
#   - Replace compute_rewards(...) to plug your own reward model/scorer.

import os
import math
import argparse
from typing import List, Tuple

import torch
from torch.optim import AdamW
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

def device_dtype():
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return ("cuda" if torch.cuda.is_available() else "cpu",
            torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32))

# ------- helpers to gather per-token logprobs on response tokens -------
@torch.no_grad()
def gather_resp_logprobs(model: AutoModelForCausalLM,
                         tok: AutoTokenizer,
                         query: torch.Tensor,
                         resp: torch.Tensor) -> torch.Tensor:
    """
    Return per-token logprob of RESP under model, shape [L_resp].
    """
    device = next(model.parameters()).device
    ids = torch.cat([query, resp], dim=0).unsqueeze(0).to(device)  # [1, Lq+Lr]
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[:, :-1, :]                 # [1, Lq+Lr-1, V]
    next_ids = ids[:, 1:]                          # [1, Lq+Lr-1]
    logp_all = torch.log_softmax(logits, dim=-1)   # [1, Lq+Lr-1, V]
    logp_chosen = torch.gather(logp_all, -1, next_ids.unsqueeze(-1)).squeeze(-1)  # [1, Lq+Lr-1]
    Lq, Lr = query.numel(), resp.numel()
    start = max(Lq - 1, 0)
    return logp_chosen[0, start:start + Lr].detach()  # [Lr]

def seq_nll_from_logprobs(resp_logprobs: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of the whole response (sum over tokens)."""
    return -resp_logprobs.sum()

def seq_len_norm(logprob_sum: torch.Tensor, length: int) -> torch.Tensor:
    return logprob_sum / max(1, length)

# ------- dataset -------
def load_prompts(dataset_path: str | None) -> List[str]:
    if dataset_path and os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
        split = "train" if "train" in ds else list(ds.keys())[0]
        ds = ds[split]
    else:
        ds = load_dataset("simplescaling/s1K")["train"]

    cand_cols = ["prompt", "question", "input", "query"]
    def to_prompt(ex):
        for k in cand_cols:
            if k in ex and isinstance(ex[k], str) and len(ex[k]) > 0:
                return {"prompt": ex[k]}
        return {"prompt": str(ex)}

    cols = [c for c in ds.column_names if c != "prompt"]
    ds = ds.map(to_prompt, remove_columns=cols)
    return [ex["prompt"] for ex in ds]

# ------- rewards (replace with your scorer if available) -------
def compute_rewards(reward_type: str,
                    ref_seq_logprobs_norm: List[float],
                    lengths: List[int]) -> List[float]:
    """
    Returns a list of scalar rewards for a group of candidates.
    reward_type:
      - "ref_logp": use normalized ref logprob as reward (default)
      - "zero": all zeros (placeholder)
      - "len": shorter is better: reward = -length (demo)
    """
    if reward_type == "zero":
        return [0.0 for _ in ref_seq_logprobs_norm]
    if reward_type == "len":
        return [-float(L) for L in lengths]
    # default: ref_logp
    return [float(x) for x in ref_seq_logprobs_norm]

def entropy_coef(step: int, max_coef: float, *, total_steps: int | None,
                 kind: str, warmup: int) -> float:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--ref_model", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-6)

    # Regularizers
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--entropy_penalty_coef", type=float, default=0.0)
    parser.add_argument("--entropy_penalty_schedule", type=str, default="linear", choices=["linear","cosine","none"])
    parser.add_argument("--entropy_warmup_steps", type=int, default=0)

    # Rewards
    parser.add_argument("--reward_type", type=str, default="ref_logp", choices=["ref_logp","zero","len"])
    parser.add_argument("--reward_temperature", type=float, default=0.1, help="softmax temperature for group weighting")

    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="outputs/grpo_entropy")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device, dtype = device_dtype()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    ref_id = args.ref_model or args.model_name_or_path
    ref = AutoModelForCausalLM.from_pretrained(
        ref_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    policy.train()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = AdamW(policy.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype in (torch.float16,)))

    prompts = load_prompts(args.dataset_path)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        num_beams=1,
    )

    global_step = 0
    from tqdm import trange
    for update in trange(args.total_updates, desc="GRPO updates"):
        # ---- sample a batch of prompts ----
        b = args.per_device_batch_size
        start = (update * b) % len(prompts)
        end = start + b
        batch_prompts = prompts[start:end]
        if len(batch_prompts) < b:
            batch_prompts += prompts[: b - len(batch_prompts)]

        # tokenize queries (left-pad)
        q = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_prompt_len).to(policy.device)
        query_tensors = [t for t in q["input_ids"]]

        total_loss = 0.0
        total_tokens = 0
        total_entropy = 0.0
        total_kl = 0.0

        # gradient accumulation over this batch
        for bi, query in enumerate(query_tensors):
            # ---- generate group_size candidates for this query ----
            responses: List[torch.Tensor] = []
            with torch.no_grad():
                for _ in range(args.group_size):
                    out = policy.generate(
                        input_ids=query.unsqueeze(0).to(policy.device),
                        **gen_cfg.to_dict(),
                    )[0]
                    # take only the newly generated tail as "response" (after the query)
                    resp = out[query.numel():]
                    responses.append(resp.detach().cpu())

            # ---- compute per-response logprobs (policy & ref) ----
            resp_logps_pol: List[torch.Tensor] = []
            resp_logps_ref: List[torch.Tensor] = []
            lengths: List[int] = []
            for resp in responses:
                lp_pol = gather_resp_logprobs(policy, tok, query, resp.to(policy.device))
                lp_ref = gather_resp_logprobs(ref, tok, query, resp.to(policy.device))
                resp_logps_pol.append(lp_pol)   # [L]
                resp_logps_ref.append(lp_ref)   # [L]
                lengths.append(lp_pol.numel())

            # sequence NLL (policy) and normalized seq logprob (ref)
            seq_nlls = [seq_nll_from_logprobs(lp) for lp in resp_logps_pol]                  # list of scalars
            ref_seq_logp_norm = [seq_len_norm(lp.sum(), L) for lp, L in zip(resp_logps_ref, lengths)]

            # ---- compute group rewards & weights ----
            rewards = compute_rewards(args.reward_type, ref_seq_logp_norm, lengths)          # list[float]
            # softmax over group with temperature
            r = torch.tensor(rewards, dtype=torch.float32, device=policy.device)
            w = torch.softmax(r / max(args.reward_temperature, 1e-3), dim=0)                 # [G]

            # ---- loss: weighted seq NLL (length-normalized) ----
            seq_loss = 0.0
            for w_i, nll_i, L_i in zip(w, seq_nlls, lengths):
                seq_loss = seq_loss + w_i * (nll_i / max(1, L_i))

            # ---- KL regularization (sample estimate on chosen tokens) ----
            # E_pi[logpi - logpref] over response tokens
            kl_terms = []
            for lp_pol, lp_ref in zip(resp_logps_pol, resp_logps_ref):
                # mean over tokens
                kl_terms.append((lp_pol - lp_ref).mean())
            kl_est = torch.stack(kl_terms).mean() if len(kl_terms) > 0 else torch.tensor(0.0, device=policy.device)
            kl_loss = args.kl_coef * kl_est

            # ---- Entropy penalty (sample-based, over response tokens) ----
            # entropy_est = E[-logpi] over tokens and group
            ent_terms = [(-lp).mean() for lp in resp_logps_pol]
            entropy_est = torch.stack(ent_terms).mean() if len(ent_terms) > 0 else torch.tensor(0.0, device=policy.device)
            coef = entropy_coef(global_step, args.entropy_penalty_coef,
                                total_steps=args.total_updates,
                                kind=args.entropy_penalty_schedule,
                                warmup=args.entropy_warmup_steps)
            ent_loss = coef * entropy_est

            loss = seq_loss + kl_loss + ent_loss

            # ---- backward (amp for fp16 only; bf16 runs in native) ----
            (loss / args.grad_accum_steps).backward()
            total_loss += float(loss.detach().cpu())
            total_tokens += int(sum(lengths))
            total_entropy += float(entropy_est.detach().cpu())
            total_kl += float(kl_est.detach().cpu())

            if (bi + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                global_step += 1

        # ---- simple logging ----
        denom = max(1, len(query_tensors))
        print(f"[upd {update}] "
              f"loss={total_loss/denom:.4f} | "
              f"entropy={total_entropy/denom:.3f} (coef={coef:.3f}) | "
              f"kl={total_kl/denom:.4f} | "
              f"tokens/batch={total_tokens}")

        # ---- save ----
        if args.save_every > 0 and (update + 1) % args.save_every == 0:
            out_dir = os.path.join(args.output_dir, f"checkpoint-{update+1}")
            os.makedirs(out_dir, exist_ok=True)
            policy.save_pretrained(out_dir, safe_serialization=True)
            tok.save_pretrained(out_dir)
            print("Saved:", out_dir)

    # final save
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy.save_pretrained(final_dir, safe_serialization=True)
    tok.save_pretrained(final_dir)
    print("Saved:", final_dir)

if __name__ == "__main__":
    main()
