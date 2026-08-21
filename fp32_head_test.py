import logging, gc
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
import torch
from transformers import AutoTokenizer
from train_lora import load_base_model_cpu
import train_lora
train_lora.snapshot_download = lambda *a, **k: SNAP  # use local cache snapshot

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"

class A: 
    base_model = SNAP
    torch_dtype = "bfloat16"
    load_in_4bit = False

model = load_base_model_cpu(A())
tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True, trust_remote_code=True)
tok.pad_token = tok.eos_token
model.eval()

ids = tok("Hello, what is 2+2?", return_tensors="pt").input_ids
with torch.no_grad():
    # Run transformer body (bf16, mmap weights ~0 commit); captures sane hidden states
    hs = model.model(input_ids=ids).last_hidden_state
    print("hidden norm:", float(hs.float().norm()), "max:", float(hs.float().max()), "min:", float(hs.float().min()), flush=True)
    # bf16 lm_head matmul (the broken path)
    logits_bf16 = model.lm_head(hs)
    print("bf16 logits max:", float(logits_bf16.float().max()), "min:", float(logits_bf16.float().min()), flush=True)
    # fp32 head matmul on SAME hidden states
    logits_fp32 = hs.float() @ model.lm_head.weight.float().t()
    print("fp32 logits max:", float(logits_fp32.max()), "min:", float(logits_fp32.min()), "mean:", float(logits_fp32.mean()), flush=True)
    print("fp32 argmax last:", int(logits_fp32[0,-1].argmax()), flush=True)
print("DONE fp32-head test")
