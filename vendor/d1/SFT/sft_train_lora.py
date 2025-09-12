#!/usr/bin/env python3
"""
sft_train_lora.py
LoRA SFT entrypoint: loads base causal model, wraps with PEFT LoRA, and trains adapters.
Launch via `accelerate.launch --num_processes N ... sft_train_lora.py ...`
"""
import argparse
import os
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from sft_trainer import dLLMTrainer, dLLMDataCollator
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import random, numpy as np, torch
from tqdm import tqdm

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", type=str, default="true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def init_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def load_prompts(dataset_path):
    if dataset_path and os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        ds = load_dataset("simplescaling/s1K")
    cand_cols = ["prompt", "question", "input", "query"]
    def pick_prompt(ex):
        for k in cand_cols:
            if k in ex and isinstance(ex[k], str) and len(ex[k])>0:
                return {"text": ex[k]}
        return {"text": str(ex)}
    ds = ds["train"].map(pick_prompt, remove_columns=[c for c in ds["train"].column_names if c != "text"])
    return ds

def preprocess_and_tokenize(ds, tokenizer, max_length):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    tok_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tok_ds.set_format(type="torch")
    return tok_ds

def main():
    args = parse_args()
    init_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- adaptive, memory-aware model loading ----------------
    _WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
    _FORCE_8BIT = bool(int(os.environ.get("FORCE_8BIT", "0")))
    _use_8bit = False
    try:
        import bitsandbytes  # noqa: F401
        _use_8bit = True
    except Exception:
        _use_8bit = False

    # allow the FORCE_8BIT env var to override autodetect
    if _FORCE_8BIT:
        _use_8bit = True

    _dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if _use_8bit:
        print("[SFT] Loading model in 8-bit (bitsandbytes) with device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            print("[SFT] prepare_model_for_kbit_training() skipped:", e)
    else:
        if _WORLD_SIZE > 1:
            # multi-GPU run: ask HF to shard with device_map='auto'
            print("[SFT] WORLD_SIZE>1 => using device_map='auto' and torch_dtype=", _dtype)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # single-GPU or CPU dev box: load to device directly
            print("[SFT] single-device load (no device_map); dtype=", _dtype)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=_dtype,
                device_map=None,
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                try:
                    model.to("cuda")
                except Exception:
                    pass

    # common safety flags
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # try to enable gradient checkpointing when supported
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[SFT] gradient_checkpointing_enable() not supported: {e} -- continuing")
    # Decide which modules to target with LoRA based on model type.
    # Some models (LLaMA-style) use different projection names; fall back to common names.
    model_type = getattr(model.config, "model_type", "") or ""
    model_type = model_type.lower()
    if "llama" in model_type or "alpaca" in model_type:
        target_modules = ["q_proj", "v_proj"]
    else:
        # common projection names for many HF causal LMs
        target_modules = ["q_proj", "k_proj", "v_proj"]

    # apply LoRA (PEFT)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    # ---------------------------------------------------------------------

    # ---------------- dataset preparation & trainer ----------------
    ds = load_prompts(args.dataset_path)
    tok_ds = preprocess_and_tokenize(ds, tokenizer, args.max_length)

    total = len(tok_ds)
    test_size = 0.01 if total >= 100 else 0.10
    split = tok_ds.train_test_split(test_size=test_size, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=(args.bf16.lower() == "true"),
        fp16=False,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
    )

    # choose safe mask token id (fallback chain)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(tokenizer, "eos_token_id", None)
    if mask_id is None:
        mask_id = getattr(tokenizer, "pad_token_id", None)
    if mask_id is None:
        mask_id = getattr(tokenizer, "unk_token_id", None)
    if mask_id is None:
        mask_id = 0

    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(
            tokenizer=tokenizer,
            mask_token_id=mask_id,
            max_length=args.max_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # ---------------------------------------------------------------------

    # # Load base model (full weights). Let accelerate place it on devices.
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # model.config.use_cache = False
    # # model.gradient_checkpointing_enable()
    # try:
    #     model.gradient_checkpointing_enable()
    # except Exception as e:
    #     print(f"[SFT] gradient_checkpointing_enable() not supported: {e} -- continuing without it.")


    # # Optional: if you plan to use 8-bit quantization, prepare for k-bit training:
    # # model = prepare_model_for_kbit_training(model)

    # # Choose target modules heuristically — adjust for your model implementation
    # model_type = getattr(model.config, "model_type", "").lower()
    # if "llama" in model_type or "alpaca" in model_type:
    #     target_modules = ["q_proj", "v_proj"]
    # else:
    #     # fallback common module names for many HF causal models
    #     target_modules = ["q_proj", "k_proj", "v_proj"]

    # lora_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     target_modules=target_modules,
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    # )

    # model = get_peft_model(model, lora_config)
    # # do not call model.to(...) — let accelerate/Trainer do device placement

    #     # load & tokenize dataset
    # ds = load_prompts(args.dataset_path)
    # tok_ds = preprocess_and_tokenize(ds, tokenizer, args.max_length)

    # # create a small eval split (1% by default, fallback to 10% for tiny datasets)
    # total = len(tok_ds)
    # test_size = 0.01 if total >= 100 else 0.10
    # split = tok_ds.train_test_split(test_size=test_size, seed=args.seed)
    # train_dataset = split["train"]
    # eval_dataset = split["test"]

    # # ensure output_dir exists
    # os.makedirs(args.output_dir, exist_ok=True)

    # # training args
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     num_train_epochs=args.num_train_epochs,
    #     learning_rate=args.learning_rate,
    #     bf16=(args.bf16.lower() == "true"),
    #     fp16=False,
    #     save_strategy="epoch",
    #     logging_strategy="steps",
    #     logging_steps=50,
    #     save_total_limit=3,
    #     remove_unused_columns=False,
    #     report_to=[],
    # )
    # # Determine mask token id (fall back to eos/pad/unk/0)
    # mask_id = getattr(tokenizer, "mask_token_id", None)
    # if mask_id is None:
    #     mask_id = getattr(tokenizer, "eos_token_id", None)
    # if mask_id is None:
    #     mask_id = getattr(tokenizer, "pad_token_id", None)
    # if mask_id is None:
    #     mask_id = getattr(tokenizer, "unk_token_id", None)
    # if mask_id is None:
    #     mask_id = 0
    
    # trainer = dLLMTrainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=dLLMDataCollator(
    #         tokenizer=tokenizer,
    #         mask_token_id=mask_id,
    #         max_length=args.max_length,
    #     ),
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
    
    trainer.train()
    model.save_pretrained(args.output_dir)  # saves adapter weights for PEFT
    print("Saved LoRA adapters to", args.output_dir)

if __name__ == "__main__":
    main()
