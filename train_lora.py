"""LoRA fine-tuning for Qwen2.5-0.5B-Instruct (Bhavyam AI accuracy tuning).

Trains a small, cheap LoRA adapter that makes the model more accurate across
formats (math, code, factual, reasoning) using the curated dataset in
`finetune_dataset.jsonl`. It does NOT retrain the whole model, so it runs on
a laptop CPU (slow) or a GPU (fast).

Requirements
------------
    pip install torch transformers peft datasets accelerate

CPU run (slow but works for 0.5B):
    python train_lora.py --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --dataset finetune_dataset.jsonl --output_dir ./lora-accurate \
        --epochs 5 --per_device_train_batch_size 1

GPU run (fast, 4-bit quantised + bf16):
    python train_lora.py --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --dataset finetune_dataset.jsonl --output_dir ./lora-accurate \
        --epochs 3 --per_device_train_batch_size 4 --load_in_4bit

After training, see the section "Using the adapter with this app" at the bottom
of this file (or read TRAIN.md) to wire the adapter (or a merged GGUF) into
MiniChat/app.py.
"""
import argparse
import json
import math
import mmap
import os
import struct
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# safetensors dtype -> (torch.dtype, element_size_bytes)
_SF_DT = {
    "BOOL": (torch.bool, 1),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "U8": (torch.uint8, 1),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "FLOAT16": (torch.float16, 2),
    "BFLOAT16": (torch.bfloat16, 2),
    "FLOAT32": (torch.float32, 4),
    "FLOAT64": (torch.float64, 8),
    "INT8": (torch.int8, 1),
    "INT16": (torch.int16, 2),
    "INT32": (torch.int32, 4),
    "INT64": (torch.int64, 8),
    "UINT8": (torch.uint8, 1),
}


# ---------------------------------------------------------------------------
# Model / LoRA config
# ---------------------------------------------------------------------------
# Qwen2 attention + MLP projection names. LoRA on these is enough to adapt a
# 0.5B model cheaply while keeping the base weights frozen.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="finetune_dataset.jsonl")
    p.add_argument("--output_dir", default="./lora-accurate")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_scheduler_type", default="constant_with_warmup")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Use 4-bit NF4 (CUDA GPU only).")
    p.add_argument("--merge", action="store_true",
                   help="Also merge LoRA into base and save a dense checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Limit training to this many optimizer steps (for quick tests). -1 = unlimited.")
    return p.parse_args()


def load_base_model_cpu(args):
    """Load the base model on a CPU box with little RAM.

    The full checkpoint (~1GB of weights) is attached as READ-ONLY, memory-mapped
    tensors (mmap) taken straight out of the safetensors file. Read-only file
    mappings are backed by the file on disk, not by the pagefile/commit limit,
    so the base weights cost ~0 bytes of committed memory. The base model is
    frozen (requires_grad=False): only the small LoRA adapter is trained, which
    is the whole point of LoRA.

    This avoids the Windows ERROR 1455 ("paging file too small") that a normal
    `from_pretrained` hits when it materialises a 1GB+ tensor on a machine with
    no pagefile and little free RAM.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    local_dir = Path(snapshot_download(args.base_model, allow_patterns=["*.safetensors"]))
    sf_files = sorted(local_dir.glob("*.safetensors"), key=lambda p: p.stat().st_size, reverse=True)
    if not sf_files:
        raise FileNotFoundError(f"No *.safetensors found in {local_dir}")
    sf_path = str(sf_files[0])

    print(f"[info] mmap-loading weights from {sf_path}", flush=True)
    with open(sf_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    data_base = 8 + header_len  # file offset where the tensor data section begins

    cfg = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)

    fh = open(sf_path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    count_loaded = 0
    for name, info in header.items():
        if "data_offsets" not in info:
            continue
        dt, esz = _SF_DT[info["dtype"]]
        start, end = info["data_offsets"]
        n = (end - start) // esz
        # safetensors `data_offsets` are relative to the start of the DATA
        # section, which begins after the 8-byte header-length + JSON header.
        t = torch.frombuffer(mm, dtype=dt, count=n, offset=start + data_base).reshape(info["shape"])
        parts = name.split(".")
        try:
            parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
            setattr(parent, parts[-1], torch.nn.Parameter(t, requires_grad=False))
            count_loaded += 1
        except (AttributeError, TypeError, KeyError):
            continue
    # Some checkpoints (Qwen2) don't store lm_head.weight because the weights are
    # tied to model.embed_tokens.weight. Re-tie NOW (after attaching the real
    # embed tensors) so lm_head.weight is not left as a meta tensor.
    model.tie_weights()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[info] attached {count_loaded} read-only mmap'd tensors; params={model.num_parameters()}", flush=True)
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    has_cuda = torch.cuda.is_available()

    print(f"[info] CUDA available: {has_cuda}; 4bit={args.load_in_4bit and has_cuda}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if has_cuda:
        # GPU path: normal from_pretrained with optional 4-bit NF4 + bf16.
        torch_dtype = torch.bfloat16 if args.load_in_4bit else torch.float16
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.base_model, device_map="auto",
                    quantization_config=quant, dtype=torch_dtype,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"[warn] 4-bit load failed ({e}); falling back to fp16/fp32.", flush=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.base_model, device_map="auto",
                    dtype=torch_dtype, trust_remote_code=True,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model, device_map="auto",
                dtype=torch_dtype, trust_remote_code=True,
            )
    else:
        # CPU path with little RAM: mmap the weights as read-only (file-backed,
        # ~0 commit) and freeze the base. Only the LoRA adapter is trained.
        model = load_base_model_cpu(args)
    model.config.use_cache = False

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- Dataset ----
    raw = load_dataset("json", data_files=args.dataset, split="train")
    print(f"[info] dataset examples: {len(raw)}")

    def tokenize(example):
        messages = example["messages"]
        # HF Qwen2 chat template -> single text string
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # Add EOS so the model learns to stop at the assistant turn.
        if not text.endswith(tokenizer.eos_token):
            text = text + tokenizer.eos_token
        tokenized = tokenizer(
            text, truncation=True, max_length=args.max_seq_len,
            padding=False,
        )
        # Dynamic per-batch padding is handled by the data collator below,
        # which pads each batch to its longest sequence instead of a fixed
        # max_length. This matters on low-RAM CPUs: padding to 512 when the
        # dataset max is ~140 doubles activation memory and trips the
        # Windows ERROR 1455 / commit-limit OOM.
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        # Labels mirror the input ids; the prompt (system+user) tokens are
        # masked to -100 so loss is only computed on the assistant response.
        labels = tokenized["input_ids"][:]
        n_prompt = min(len(prompt_ids), len(labels))
        labels[:n_prompt] = [-100] * n_prompt
        tokenized["labels"] = labels
        return tokenized

    tokenized = raw.map(tokenize, remove_columns=raw.column_names)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors="pt"
    )

    # ---- Training ----
    steps_per_epoch = max(
        1, math.ceil(len(tokenized) / args.per_device_train_batch_size)
    )
    total_steps = steps_per_epoch * args.epochs
    print(f"[info] ~{steps_per_epoch} steps/epoch, {total_steps} total steps "
          f"(CPU: expect {total_steps} slow steps; GPU: fast)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_steps=max(10, total_steps // 4),
        save_total_limit=2,
        fp16=has_cuda and not args.load_in_4bit,
        bf16=has_cuda and args.load_in_4bit,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()

    # ---- Save adapter + tokenizer ----
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] LoRA adapter saved to {args.output_dir}")

    # ---- Optional: merge into a dense checkpoint ----
    if args.merge:
        merged = model.merge_and_unload()
        merged_path = os.path.join(args.output_dir, "merged_dense")
        merged.save_pretrained(merged_path, safe_serialization=True)
        print(f"[done] Merged dense weights saved to {merged_path}")
        print("Next: convert to GGUF with llama.cpp, then run:")
        print("  ./convert_hf_to_gguf.py --model_dir "
              f"{merged_path} --outfile ./qwen2.5-0.5b-accurate.gguf")
        print("  ./quantize ./qwen2.5-0.5b-accurate.gguf "
              "./qwen2.5-0.5b-accurate-q4_k_m.gguf Q4_K_M")

    # ---- Using the adapter with this app (MiniChat/app.py) ----
    print("\n===== Using the adapter with MiniChat/app.py =====")
    print("Option A - runtime LoRA (keeps base GGUF, lightweight):")
    print("  llama-cpp-python's Llama() supports lora_adapter=. Convert this")
    print("  HF adapter to GGUF-LoRA with llama.cpp's tools, then set on the")
    print("  Llama() call in app.py: Llama(..., lora_base=BASE, lora_adapter=ADAPTER_GGUF).")
    print("Option B - baked-in (recommended for simplicity):")
    print("  Run with --merge, convert the merged_dense dir to a GGUF with Q4_K_M,")
    print("  place it in .model_cache, and add it to MODEL_OPTIONS in app.py as a")
    print("  new entry the app can download/load like the others.")


if __name__ == "__main__":
    main()
