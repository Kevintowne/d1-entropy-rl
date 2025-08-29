# eval/quick_eval.py
# Minimal generation script for SFT / PPO / GRPO checkpoints.
# Usage examples (from repo root):
#   python eval/quick_eval.py --model_path /abs/path/to/ckpt_or_model --input "Explain PPO briefly."
#   python eval/quick_eval.py --model_path outputs/ppo_entropy/final --file prompts.txt --save_jsonl gen.jsonl

import os, argparse, json, time
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def get_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_model(model_path: str):
    dtype = get_dtype()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return tok, model

def apply_chat_template_if_needed(tok, text: str, use_chat: bool) -> str:
    if not use_chat:
        return text
    try:
        # generic single-turn chat
        messages = [{"role": "user", "content": text}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # tokenizer not providing chat template → fall back
        return text

def batched(iterable: List[str], bs: int):
    for i in range(0, len(iterable), bs):
        yield iterable[i:i+bs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--input", type=str, default=None, help="single prompt")
    ap.add_argument("--file", type=str, default=None, help="path to a txt file (one prompt per line)")
    ap.add_argument("--use_chat_template", action="store_true", help="wrap prompt by tokenizer chat template")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--num_samples", type=int, default=1, help="samples per prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_jsonl", type=str, default=None)
    args = ap.parse_args()

    # collect prompts
    prompts: List[str] = []
    if args.input:
        prompts.append(args.input)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    if not prompts:
        raise SystemExit("No prompts provided. Use --input or --file.")

    torch.manual_seed(args.seed)
    tok, model = load_model(args.model_path)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    all_records = []
    t0 = time.time()
    for batch in batched(prompts, args.batch_size):
        # apply chat template if requested
        batch_inputs = [apply_chat_template_if_needed(tok, p, args.use_chat_template) for p in batch]
        # duplicate per num_samples
        expanded_inputs = []
        owners = []
        for i, inp in enumerate(batch_inputs):
            for _ in range(args.num_samples):
                expanded_inputs.append(inp)
                owners.append(i)

        enc = tok(expanded_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            out = model.generate(**enc, **gen_cfg.to_dict())
        # slice off the prompt to get only generated tail
        for k, (inp_ids, out_ids) in enumerate(zip(enc.input_ids, out)):
            gen_ids = out_ids[len(inp_ids):]
            text_in = expanded_inputs[k]
            text_out = tok.decode(gen_ids, skip_special_tokens=True)
            rec = {
                "prompt": text_in,
                "response": text_out,
                "owner_index_in_batch": owners[k],
            }
            all_records.append(rec)
            print("\n" + "="*80)
            print(f"[sample #{k+1}]")
            print("PROMPT:\n" + text_in)
            print("-"*80)
            print("RESPONSE:\n" + text_out)

    dt = time.time() - t0
    print(f"\nDone {len(all_records)} generations in {dt:.2f}s")

    if args.save_jsonl:
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved to {args.save_jsonl}")

if __name__ == "__main__":
    main()
