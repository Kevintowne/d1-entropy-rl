# rl/ppo_train.py
# Minimal PPO with sample-based entropy penalty using TRL.
# - Adds: --algo {ppo,grpo}, optional reward model scoring, reward normalization/clipping,
#   eval hook, config save, smoke-test mode, and main-process-only saving/logging.

import os
import math
import json
import statistics
import numpy as np
import argparse
from typing import List

import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import Dataset, load_dataset, load_from_disk
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


def gather_logprobs(model, tokenizer, query_tensors, response_tensors, device=None):
    """
    Robustly compute per-token logprobs for each generated response.
    Returns: list of 1-D torch tensors containing per-token logprobs for each response (on `device`).
    """
    # resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # ---- normalize queries -> list of 1-D LongTensors on device ----
    queries = []
    if isinstance(query_tensors, dict):
        input_ids = query_tensors.get("input_ids")
        attn = query_tensors.get("attention_mask", None)
        if isinstance(input_ids, torch.Tensor):
            for i in range(input_ids.shape[0]):
                q = input_ids[i]
                if attn is not None:
                    L = int(attn[i].sum().item())
                    if L > 0:
                        q = q[:L].contiguous()
                queries.append(q.to(device).long())
        elif isinstance(input_ids, (list, tuple)):
            for el in input_ids:
                queries.append(torch.tensor(el, dtype=torch.long, device=device))
        else:
            raise TypeError("Unsupported input_ids type in query_tensors: %s" % type(input_ids))
    elif isinstance(query_tensors, (list, tuple)):
        for el in query_tensors:
            if isinstance(el, torch.Tensor):
                t = el.detach()
                if t.dim() > 1 and t.size(0) == 1:
                    t = t.squeeze(0)
                queries.append(t.to(device).view(-1).long())
            elif isinstance(el, (list, tuple)):
                queries.append(torch.tensor(el, dtype=torch.long, device=device))
            else:
                raise TypeError("Unsupported element in query_tensors list: %s" % type(el))
    else:
        raise TypeError("Unsupported query_tensors type: %s" % type(query_tensors))

    # ---- normalize responses -> list of 1-D LongTensors on device ----
    responses = []
    for i, resp in enumerate(response_tensors):
        if isinstance(resp, torch.Tensor):
            t = resp.detach()
            if t.dim() > 1 and t.size(0) == 1:
                t = t.squeeze(0)
            t = t.view(-1).long().to(device)
            responses.append(t)
            continue

        if isinstance(resp, dict):
            for key in ("sequences", "sequences_token_ids", "generated_token_ids", "tokens", "output_ids"):
                if key in resp:
                    val = resp[key]
                    if isinstance(val, torch.Tensor):
                        t = val.detach()
                    else:
                        try:
                            val = list(val)
                            t = torch.tensor(val, dtype=torch.long)
                        except Exception:
                            t = None
                    if t is not None:
                        if t.dim() > 1 and t.size(0) == 1:
                            t = t.squeeze(0)
                        responses.append(t.long().to(device))
                        break
            else:
                text = resp.get("generated_text") or resp.get("text") or str(resp)
                enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
                responses.append(enc["input_ids"][0].to(device).long())
            continue

        if isinstance(resp, (list, tuple)) and len(resp) > 0 and isinstance(resp[0], int):
            responses.append(torch.tensor(resp, dtype=torch.long, device=device))
            continue

        if isinstance(resp, str):
            enc = tokenizer(resp, return_tensors="pt", add_special_tokens=False)
            responses.append(enc["input_ids"][0].to(device).long())
            continue

        try:
            s = str(resp)
            enc = tokenizer(s, return_tensors="pt", add_special_tokens=False)
            responses.append(enc["input_ids"][0].to(device).long())
        except Exception as e:
            raise TypeError(f"Cannot normalize generated response index={i}, type={type(resp)}") from e

    # broadcast queries if necessary
    if len(queries) != len(responses):
        if len(queries) == 1 and len(responses) > 1:
            queries = queries * len(responses)
        else:
            raise RuntimeError(f"Number of queries ({len(queries)}) != responses ({len(responses)})")

    # ---- compute per-token logprobs for each (q, r) pair ----
    out = []
    model.eval()
    with torch.no_grad():
        for q, r in zip(queries, responses):
            input_ids = torch.cat([q, r], dim=0).unsqueeze(0).to(device)   # shape (1, seq)
            outputs = model(input_ids)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    raise RuntimeError("Model outputs do not contain logits")
            # compute log probs
            logp = F.log_softmax(logits, dim=-1)  # (1, seq, vocab)
            start = q.size(0)
            r_len = r.size(0)
            if r_len == 0:
                out.append(torch.zeros(0, dtype=torch.float32, device=device))
                continue
            resp_logits = logp[0, start:start + r_len, :]            # (r_len, vocab)
            resp_ids = input_ids[0, start:start + r_len]             # (r_len,)
            per_token_logprobs = resp_logits.gather(dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)  # (r_len,)
            out.append(per_token_logprobs.detach().to(device))
    return out


def normalize_query_tensors(qt, device):
    """
    Accepts qt which can be:
      - a dict of tensors (batched) -> returned as-is (moved to device)
      - a list of dicts (per-example) -> collated into batched dict
      - a list/tuple of input_ids tensors -> collated
    Returns: (batched_dict, batch_size)
    """
    import torch
    if isinstance(qt, dict):
        batched = {}
        for k, v in qt.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(device)
            else:
                batched[k] = torch.tensor(v, device=device)
        batch_size = batched.get("input_ids").shape[0] if "input_ids" in batched else \
                     next(iter(batched.values())).shape[0]
        return batched, batch_size

    if isinstance(qt, (list, tuple)):
        if len(qt) > 0 and isinstance(qt[0], torch.Tensor):
            stacked = torch.vstack([t.to(device) for t in qt])
            return {"input_ids": stacked}, stacked.shape[0]
        if len(qt) > 0 and isinstance(qt[0], dict):
            keys = qt[0].keys()
            collated = {}
            for k in keys:
                elems = [d[k] for d in qt]
                if isinstance(elems[0], torch.Tensor):
                    collated[k] = torch.vstack([e.to(device) for e in elems])
                else:
                    collated[k] = torch.tensor(elems, device=device)
            batch_size = collated.get("input_ids").shape[0] if "input_ids" in collated else len(qt)
            return collated, batch_size

    try:
        t = torch.tensor(qt, device=device)
        return {"input_ids": t}, t.shape[0]
    except Exception:
        raise ValueError("Unable to normalize query_tensors: type=%s" % type(qt))


def _to_numpy_deep(x):
    """Recursively convert torch tensors to numpy (float32) and return numpy arrays or scalars."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
            t = t.to(torch.float32)
        return t.numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        converted = [_to_numpy_deep(el) for el in x]
        try:
            return np.asarray(converted)
        except Exception:
            return np.array(converted, dtype=object)
    try:
        return np.asarray(x)
    except Exception:
        return np.array(x, dtype=object)


def ensure_reward_scalars(rewards, reduce="sum"):
    """
    Normalize `rewards` into a Python list of floats, one per example.
    """
    arr = _to_numpy_deep(rewards)

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        out = []
        for el in arr.tolist():
            sub = np.asarray(el)
            if sub.ndim == 0:
                out.append(float(sub))
            elif sub.ndim == 1:
                out.append(float(sub.sum()) if reduce == "sum" else float(sub.mean()))
            else:
                reduced = sub.sum(axis=-1) if reduce == "sum" else sub.mean(axis=-1)
                out.append(float(np.asarray(reduced).item()) if reduced.size == 1 else float(np.asarray(reduced)[0]))
        return out

    if np.ndim(arr) == 0:
        return [float(np.asarray(arr).item())]

    if arr.ndim == 1:
        return [float(x) for x in arr.tolist()]

    if arr.ndim >= 2:
        if reduce == "sum":
            reduced = arr.sum(axis=-1)
        else:
            reduced = arr.mean(axis=-1)
        if reduced.ndim > 1:
            reduced = reduced.reshape(reduced.shape[0], -1).sum(axis=-1)
        return [float(x) for x in np.asarray(reduced).tolist()]

    flat = np.asarray(arr).ravel()
    return [float(x) for x in flat.tolist()]


# ---------- dataset helpers ----------
def load_prompts(dataset_path: str | None):
    if dataset_path and os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        ds = load_dataset("simplescaling/s1K")

    cand_cols = ["prompt", "question", "input", "query"]
    def pick_prompt(ex):
        for k in cand_cols:
            if k in ex and isinstance(ex[k], str) and len(ex[k]) > 0:
                return {"prompt": ex[k]}
        return {"prompt": str(ex)}

    ds = ds["train"].map(pick_prompt, remove_columns=[c for c in ds["train"].column_names if c != "prompt"])
    prompts = [ex["prompt"] for ex in ds]
    return prompts
    
def ensure_lm_head(model: nn.Module):
    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    head = getattr(model, "lm_head", None) if isinstance(getattr(model, "lm_head", None), nn.Module) else None
    if head is None:
        cand_name, cand = None, None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and getattr(mod, "out_features", None) == vocab:
                cand_name, cand = name, mod   # take the last matching linear
        head = cand
    if head is None:
        raise ValueError("Cannot locate a language-model head that projects to vocab_size.")
    # expose head in all expected ways
    model.lm_head = head
    if hasattr(model, "set_output_embeddings"):
        try:
            model.set_output_embeddings(head)
        except Exception:
            pass
    if (not hasattr(model, "get_output_embeddings")) or (model.get_output_embeddings() is None):
        def _get_output_embeddings(self=model): return head
        model.get_output_embeddings = _get_output_embeddings
    # keep config consistent
    try:
        if getattr(model.config, "vocab_size", None) != getattr(head, "out_features", None):
            model.config.vocab_size = getattr(head, "out_features", model.config.vocab_size)
    except Exception:
        pass
    # tie if supported (silences the warning)
    try: model.tie_weights()
    except Exception: pass
    return model
    
def get_device(model):
    if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "device"):
        return model.pretrained_model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_main_process():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


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

    # algo / reward options
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"], help="Algorithm to run")
    parser.add_argument("--reward_model", type=str, default=None, help="Optional reward model (HF id or local path) for base rewards")
    parser.add_argument("--reward_normalize", action="store_true", help="Normalize scalar rewards per batch")
    parser.add_argument("--reward_clip", type=float, default=0.0, help="Clip normalized rewards to +/- this value (0=no clipping)")
    parser.add_argument("--eval_every", type=int, default=0, help="Run eval every N updates (0=disable)")
    parser.add_argument("--eval_prompts_file", type=str, default=None, help="Optional file with validation prompts (one per line)")

    # entropy penalty
    parser.add_argument("--entropy_penalty_coef", type=float, default=0.0)
    parser.add_argument("--entropy_penalty_schedule", type=str, default="linear", choices=["linear", "cosine", "none"])
    parser.add_argument("--entropy_warmup_steps", type=int, default=0)

    parser.add_argument("--save_every", type=int, default=200, help="save model every N updates (0=disable)")
    parser.add_argument("--output_dir", type=str, default="outputs/ppo_entropy")
    parser.add_argument("--seed", type=int, default=42)

    # debugging / smoke
    parser.add_argument("--smoke_test", action="store_true", help="Run a fast smoke test (few updates)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Save run config (main process only)
    if is_main_process():
        try:
            with open(os.path.join(args.output_dir, "ppo_run_config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
        except Exception:
            pass

    # tokenizer (single creation)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # better for generation with lm_head

    # dtype selection
    dtype = (
        torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    ref_id = args.ref_model or args.model_name_or_path
    # adaptive loader (paste instead of the simple from_pretrained ...)
    _WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
    _FORCE_8BIT = bool(int(os.environ.get("FORCE_8BIT", "0")))
    _use_8bit = False
    try:
        import bitsandbytes  # noqa: F401
        _use_8bit = True
    except Exception:
        _use_8bit = False
    if _FORCE_8BIT:
        _use_8bit = True
    _dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
             torch.float16 if torch.cuda.is_available() else torch.float32)

    if _use_8bit:
        print("[PPO] Loading model in 8-bit with device_map='auto'")
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        try:
            from peft import prepare_model_for_kbit_training
            base = prepare_model_for_kbit_training(base)
        except Exception:
            pass
    else:
        if _WORLD_SIZE > 1:
            base = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=_dtype,
                device_map=None,
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                try:
                    base.to("cuda")
                except Exception:
                    pass

    try:
        base.config.use_cache = False
    except Exception:
        pass



    base = ensure_lm_head(base)
    print("[PPO] policy LM head ready")

    # wrap policy with value head
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(base)

    # ---------- adaptive reference model loading (REPLACE existing ref load) ----------
    ref_id = args.ref_model or args.model_name_or_path
    
    # detect env
    _WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
    _FORCE_8BIT = bool(int(os.environ.get("FORCE_8BIT", "0")))
    
    # autodetect bitsandbytes
    _use_8bit = False
    try:
        import bitsandbytes  # noqa: F401
        _use_8bit = True
    except Exception:
        _use_8bit = False
    
    if _FORCE_8BIT:
        _use_8bit = True
    
    # dtype fallback (match policy logic)
    _ref_dtype = (
        torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    
    # Load ref base model adaptively so ref_base is always defined
    if _use_8bit:
        print(f"[PPO] Loading ref model {ref_id} in 8-bit with device_map='auto'")
        ref_base = AutoModelForCausalLM.from_pretrained(
            ref_id,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        if _WORLD_SIZE > 1:
            print(f"[PPO] WORLD_SIZE={_WORLD_SIZE} -> loading ref with device_map='auto', dtype={_ref_dtype}")
            ref_base = AutoModelForCausalLM.from_pretrained(
                ref_id,
                torch_dtype=_ref_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print(f"[PPO] single-device ref load -> dtype={_ref_dtype}")
            ref_base = AutoModelForCausalLM.from_pretrained(
                ref_id,
                torch_dtype=_ref_dtype,
                device_map=None,
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                try:
                    ref_base.to("cuda")
                except Exception:
                    pass
    
    # ensure LM head and disable cache
    ref_base = ensure_lm_head(ref_base)
    try:
        ref_base.config.use_cache = False
    except Exception:
        pass
    
    # wrap ref in TRL value-head wrapper and freeze parameters
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(ref_base)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    # -------------------------------------------------------------------------------


    # turn OFF KV cache
    for m in (policy, ref):
        try:
            m.pretrained_model.config.use_cache = False
            if getattr(m.pretrained_model, "generation_config", None) is not None:
                m.pretrained_model.generation_config.use_cache = False
        except Exception:
            pass

    # Optional reward model (sequence-level scorer)
    rm = None
    rm_tok = None
    if args.reward_model:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer as HFAutoTokenizer
            print("[PPO] loading reward model:", args.reward_model)
            rm_tok = HFAutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
            rm = AutoModelForSequenceClassification.from_pretrained(args.reward_model, trust_remote_code=True, device_map="auto")
            rm.eval()
        except Exception as e:
            print("[WARN] Failed to load reward model:", e)
            rm = None
            rm_tok = None

        def score_with_rm(prompt_texts, response_texts):
            # Map (prompt, response) -> scalar score (float)
            texts = [p + "\n" + r for p, r in zip(prompt_texts, response_texts)]
            enc = rm_tok(texts, truncation=True, padding=True, return_tensors="pt").to(get_device(rm))
            with torch.no_grad():
                out = rm(**enc)
                logits = getattr(out, "logits", None)
                if logits is None:
                    return [0.0] * len(texts)
                logits = logits.cpu()
                if logits.size(-1) == 1:
                    # regression / single-logit => sigmoid -> [0,1]
                    return [float(torch.sigmoid(v).item()) for v in logits.squeeze(-1)]
                else:
                    # else take softmax expectation over classes as numeric score
                    probs = torch.softmax(logits, dim=-1)
                    classes = torch.arange(logits.size(-1), dtype=torch.float32).unsqueeze(0)
                    scores = (probs * classes).sum(dim=-1)
                    return [float(s.item()) for s in scores]
    else:
        rm = rm_tok = None
        def score_with_rm(prompts, responses):
            return [0.0] * len(prompts)

    # PPO config
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
    per_device = args.per_device_batch_size
    ga = args.grad_accum_steps
    global_per_step = per_device * world_size

    ppo_config = PPOConfig(
        batch_size=global_per_step * ga,
        mini_batch_size=global_per_step,
        gradient_accumulation_steps=ga,
        learning_rate=1e-5,
        target_kl=0.1,
        cliprange=0.2,
        ppo_epochs=1,
        seed=42,
        remove_unused_columns=False,
    )

    prompts = load_prompts(args.dataset_path)   # your offline-aware loader
    dataset = Dataset.from_dict({"prompt": prompts})

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref,
        tokenizer=tok,
        dataset=dataset,
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=False,
    )

    # prepare eval prompts (if requested)
    val_prompts = None
    if args.eval_prompts_file and os.path.exists(args.eval_prompts_file):
        with open(args.eval_prompts_file, "r") as f:
            val_prompts = [l.strip() for l in f if l.strip()]
    elif args.eval_every > 0:
        # take first few prompts as validation
        val_prompts = prompts[: min(32, len(prompts))]

    global_step = 0
    from tqdm import trange
    num_prompts = len(prompts)
    # allow a smoke-test quick run
    total_updates = 3 if args.smoke_test else args.total_updates

    for update in trange(total_updates, desc="PPO updates"):
        # --- fixed sampling window ---
        start_idx = (update * args.per_device_batch_size) % num_prompts
        batch_prompts = []
        for i in range(args.per_device_batch_size):
            batch_prompts.append(prompts[(start_idx + i) % num_prompts])

        dev = get_device(policy)

        # tokenization
        batch_encoding = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_len,
        )

        query_tensors = {}
        for k, v in batch_encoding.items():
            if isinstance(v, torch.Tensor):
                query_tensors[k] = v.to(dev)

        if "input_ids" not in query_tensors:
            raise RuntimeError("tokenizer did not return input_ids")

        # --- Prepare list of trimmed 1-D input_id tensors on `dev` for generation ---
        queries_for_gen = []
        input_ids = query_tensors.get("input_ids")
        attention_mask = query_tensors.get("attention_mask", None)
        
        if input_ids is None:
            raise RuntimeError("No input_ids found for generation")
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).to(torch.int64)
            for i in range(input_ids.shape[0]):
                L = int(lengths[i].item())
                if L <= 0:
                    q = input_ids[i].contiguous()
                else:
                    q = input_ids[i, :L].contiguous()
                queries_for_gen.append(q.view(-1).long().to(dev))
        else:
            pad_id = getattr(tok, "pad_token_id", None)
            for i in range(input_ids.shape[0]):
                row = input_ids[i]
                if pad_id is not None:
                    nonpad = (row != pad_id).nonzero(as_tuple=False)
                    if nonpad.numel() == 0:
                        q = row
                    else:
                        last = int(nonpad[-1].item())
                        q = row[: last + 1].contiguous()
                else:
                    q = row
                queries_for_gen.append(q.view(-1).long().to(dev))
        
        # Generate using trainer.generate with token tensors (TRL expects tensors for this version)
        with torch.no_grad():
            response_tensors = trainer.generate(queries_for_gen, **gen_cfg.to_dict())


        # Normalize responses -> list of 1-D LongTensors on device
        normalized_responses = []
        for i, resp in enumerate(response_tensors):
            if isinstance(resp, torch.Tensor):
                t = resp.detach()
                if t.dim() > 1 and t.size(0) == 1:
                    t = t.squeeze(0)
                normalized_responses.append(t.view(-1).long().to(dev))
                continue

            if isinstance(resp, dict):
                for key in ("sequences", "sequences_token_ids", "generated_token_ids", "tokens", "output_ids"):
                    if key in resp:
                        val = resp[key]
                        if isinstance(val, torch.Tensor):
                            t = val.detach()
                        else:
                            try:
                                val = list(val)
                                t = torch.tensor(val, dtype=torch.long)
                            except Exception:
                                t = None
                        if t is not None:
                            if t.dim() > 1 and t.size(0) == 1:
                                t = t.squeeze(0)
                            normalized_responses.append(t.long().to(dev))
                            break
                else:
                    text = resp.get("text") or resp.get("generated_text") or str(resp)
                    enc = tok(text, return_tensors="pt", add_special_tokens=False)
                    normalized_responses.append(enc["input_ids"][0].to(dev).long())
                continue

            if isinstance(resp, (list, tuple)) and len(resp) > 0 and isinstance(resp[0], int):
                normalized_responses.append(torch.tensor(resp, dtype=torch.long, device=dev))
                continue

            if isinstance(resp, str):
                enc = tok(resp, return_tensors="pt", add_special_tokens=False)
                normalized_responses.append(enc["input_ids"][0].to(dev).long())
                continue

            if hasattr(resp, "sequences"):
                seq = getattr(resp, "sequences")
                if isinstance(seq, torch.Tensor):
                    t = seq.detach()
                    if t.dim() > 1 and t.size(0) == 1:
                        t = t.squeeze(0)
                    normalized_responses.append(t.long().to(dev))
                    continue
            if hasattr(resp, "generated_text"):
                txt = getattr(resp, "generated_text")
                enc = tok(txt, return_tensors="pt", add_special_tokens=False)
                normalized_responses.append(enc["input_ids"][0].to(dev).long())
                continue

            try:
                s = str(resp)
                enc = tok(s, return_tensors="pt", add_special_tokens=False)
                normalized_responses.append(enc["input_ids"][0].to(dev).long())
                continue
            except Exception as e:
                raise TypeError(f"Failed to normalize generated response index={i} type={type(resp)}") from e

        response_tensors = normalized_responses
        if is_main_process():
            print("DEBUG normalized response_tensors:", [(type(x), tuple(x.shape)) for x in response_tensors[:4]])

        # compute per-token logprobs on responses (policy)
        resp_logprobs_list = gather_logprobs(policy.pretrained_model, tok, query_tensors, response_tensors, device=dev)

        # --- base rewards ---
        # if an RM is provided, compute sequence-level scalar scores and tile across tokens
        if rm is not None:
            # decode texts (uses policy tokenizer to decode token ids)
            prompt_texts = [tok.decode(q.view(-1).tolist(), skip_special_tokens=True) if isinstance(q, torch.Tensor) else str(q) for q in queries_list_from_encoding(query_tensors, tok, dev)]
            resp_texts = [tok.decode(r.view(-1).tolist(), skip_special_tokens=True) if isinstance(r, torch.Tensor) else str(r) for r in response_tensors]
            try:
                rm_scores = score_with_rm(prompt_texts, resp_texts)
            except Exception as e:
                print("[WARN] reward model scoring failed:", e)
                rm_scores = [0.0] * len(response_tensors)
            base_rewards = [torch.full_like(lp.to(torch.float32), float(s), device=dev) for lp, s in zip(resp_logprobs_list, rm_scores)]
        else:
            base_rewards = [torch.zeros_like(lp, dtype=torch.float32, device=dev) for lp in resp_logprobs_list]

        # entropy penalty coef schedule
        coef = schedule_coef(
            global_step,
            args.entropy_penalty_coef,
            total_steps=total_updates,
            kind=args.entropy_penalty_schedule,
            warmup=args.entropy_warmup_steps,
        )

        # shaped rewards (token-level): base - coef * (-logpi) ; logprobs are per-token
        shaped_rewards = [br - coef * (-lp.to(torch.float32)) for br, lp in zip(base_rewards, resp_logprobs_list)]

        # Convert per-token shaped_rewards -> per-sample scalar scores (floats)
        scores = ensure_reward_scalars(shaped_rewards, reduce="sum")   # or "mean"

        # Optional normalization & clipping
        if args.reward_normalize and len(scores) > 1:
            mean_s = statistics.mean(scores)
            std_s = statistics.pstdev(scores) if len(scores) > 1 else 1.0
            std_s = max(std_s, 1e-6)
            scores = [(s - mean_s) / std_s for s in scores]
        if args.reward_clip and args.reward_clip > 0.0:
            scores = [max(-args.reward_clip, min(args.reward_clip, s)) for s in scores]

        # Normalize query_tensors into batched dict & get batch_size
        query_tensors_batched, batch_size = normalize_query_tensors(query_tensors, dev)

        # validation
        if len(scores) != batch_size:
            print(f"[ERR] reward count {len(scores)} != batch_size {batch_size}")
            print("DEBUG query_tensors keys:", list(query_tensors_batched.keys()))
            for k, v in query_tensors_batched.items():
                try:
                    print("  ", k, getattr(v, "shape", None), type(v))
                except Exception:
                    pass
            raise RuntimeError("Mismatch: rewards length and batch size differ")

        # prepare queries_list (trimmed) used by trainer.step
        queries_list = []
        input_ids = query_tensors_batched.get("input_ids")
        attention_mask = query_tensors_batched.get("attention_mask", None)
        if input_ids is not None:
            input_ids = input_ids.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
                lengths = attention_mask.sum(dim=1).to(torch.int64)
                for i in range(input_ids.shape[0]):
                    L = int(lengths[i].item())
                    if L <= 0:
                        q = input_ids[i].contiguous()
                    else:
                        q = input_ids[i, :L].contiguous()
                    queries_list.append(q.view(-1).long().to(dev))
            else:
                pad_id = getattr(tok, "pad_token_id", None)
                for i in range(input_ids.shape[0]):
                    row = input_ids[i]
                    if pad_id is not None:
                        nonpad = (row != pad_id).nonzero(as_tuple=False)
                        if nonpad.numel() == 0:
                            q = row
                        else:
                            last = int(nonpad[-1].item())
                            q = row[: last + 1].contiguous()
                    else:
                        q = row
                    queries_list.append(q.view(-1).long().to(dev))
        else:
            queries_list = [torch.tensor(q, dtype=torch.long, device=dev).view(-1) if not isinstance(q, torch.Tensor) else q.view(-1).to(dev) for q in query_tensors_batched]

        # prepare responses_list
        responses_list = []
        for resp in response_tensors:
            if isinstance(resp, torch.Tensor):
                t = resp.detach()
                if t.dim() > 1 and t.size(0) == 1:
                    t = t.squeeze(0)
                responses_list.append(t.view(-1).long().to(dev))
            elif isinstance(resp, (list, tuple)):
                responses_list.append(torch.tensor(resp, dtype=torch.long, device=dev).view(-1))
            else:
                enc = tok(str(resp), return_tensors="pt", add_special_tokens=False)
                responses_list.append(enc["input_ids"][0].to(dev).long())

        # Scores_list: convert floats to torch tensors on device (TRL supports float list too; this is fine)
        scores_list = []
        for s in scores:
            if isinstance(s, torch.Tensor):
                scores_list.append(s.detach().to(dev).to(torch.float32))
            else:
                scores_list.append(torch.tensor(float(s), dtype=torch.float32, device=dev))

        # sanity checks
        if not isinstance(queries_list, list) or not all(isinstance(q, torch.Tensor) for q in queries_list):
            raise RuntimeError(f"queries_list must be list[tensor], got: {type(queries_list)}")
        if len(queries_list) != len(responses_list) or len(scores_list) != len(queries_list):
            print("DEBUG lengths:", len(queries_list), len(responses_list), len(scores_list))
            raise RuntimeError("Length mismatch between queries/responses/scores")

        # finally call training step depending on algo
        if args.algo == "ppo":
            stats = trainer.step(queries_list, responses_list, scores_list)
        else:
            # GRPO stub: for now use the same interface as PPO; replace with real GRPO implementation later
            # TODO: implement trainer.grpo_step(...) with GRPO-specific objectives if needed
            stats = trainer.step(queries_list, responses_list, scores_list)

        # logging (entropy estimate) - only print from main
        if is_main_process():
            with torch.no_grad():
                ent_vals = [(-lp).mean().item() if lp.numel() > 0 else 0.0 for lp in resp_logprobs_list]
                avg_ent = sum(ent_vals) / len(ent_vals) if len(ent_vals) > 0 else 0.0
                print(f"[upd {update}] coef={coef:.4f} | entropy_est={avg_ent:.3f} | "
                      f"kl={float(stats.get('objective/kl', 0.0)):.4f}")

        global_step += 1

        # periodic eval
        if args.eval_every > 0 and (update + 1) % args.eval_every == 0 and is_main_process() and val_prompts:
            try:
                print("[EVAL] running eval generation...")
                gens = trainer.generate(val_prompts, **gen_cfg.to_dict())
                # compute RM and entropy estimates for eval set (best-effort)
                eval_resp_texts = []
                for g in gens:
                    if isinstance(g, torch.Tensor):
                        eval_resp_texts.append(tok.decode(g.view(-1).tolist(), skip_special_tokens=True))
                    elif isinstance(g, str):
                        eval_resp_texts.append(g)
                    elif isinstance(g, dict) and ("generated_text" in g or "text" in g):
                        eval_resp_texts.append(g.get("generated_text") or g.get("text"))
                    else:
                        eval_resp_texts.append(str(g))
                if rm is not None:
                    eval_rm = score_with_rm(val_prompts[:len(eval_resp_texts)], eval_resp_texts)
                else:
                    eval_rm = [0.0] * len(eval_resp_texts)
                print("[EVAL] sample RM scores:", eval_rm[:4])
            except Exception as e:
                print("[WARN] eval failed:", e)

        # periodic save (only main)
        if is_main_process() and args.save_every > 0 and (update + 1) % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{update+1}")
            try:
                trainer.model.save_pretrained(save_dir, safe_serialization=True)
                tok.save_pretrained(save_dir)
                print("[SAVE] checkpoint saved:", save_dir)
            except Exception as e:
                print("[WARN] failed to save checkpoint:", e)

    # final save (main only)
    if is_main_process():
        final_dir = os.path.join(args.output_dir, "final")
        trainer.model.save_pretrained(final_dir, safe_serialization=True)
        tok.save_pretrained(final_dir)
        print("Saved:", final_dir)


# helper: build queries_list from batched query_tensors for RM decoding
def queries_list_from_encoding(query_tensors, tokenizer, device):
    """Return list of trimmed 1-D input_id tensors (on cpu) used to decode prompt text."""
    if isinstance(query_tensors, dict) and "input_ids" in query_tensors:
        input_ids = query_tensors["input_ids"]
        attn = query_tensors.get("attention_mask", None)
        out = []
        if attn is not None:
            lengths = attn.sum(dim=1).to(torch.int64)
            for i in range(input_ids.shape[0]):
                L = int(lengths[i].item())
                if L <= 0:
                    q = input_ids[i]
                else:
                    q = input_ids[i, :L].contiguous()
                out.append(q.detach().cpu())
        else:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            for i in range(input_ids.shape[0]):
                row = input_ids[i]
                if pad_id is not None:
                    nonpad = (row != pad_id).nonzero(as_tuple=False)
                    if nonpad.numel() == 0:
                        q = row
                    else:
                        last = int(nonpad[-1].item())
                        q = row[: last + 1].contiguous()
                else:
                    q = row
                out.append(q.detach().cpu())
        return out
    # fallback if it's a list already
    if isinstance(query_tensors, (list, tuple)):
        return [torch.tensor(q, dtype=torch.long).detach().cpu() if not isinstance(q, torch.Tensor) else q.detach().cpu() for q in query_tensors]
    return []


if __name__ == "__main__":
    main()
