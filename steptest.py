import logging, gc
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from train_lora import load_base_model_cpu, TARGET_MODULES
import train_lora
SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
train_lora.snapshot_download = lambda *a, **k: SNAP

class A:
    base_model = SNAP
    torch_dtype = "bfloat16"
    load_in_4bit = False

model = load_base_model_cpu(A())
tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True, trust_remote_code=True)
tok.pad_token = tok.eos_token
model.config.use_cache = True

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=TARGET_MODULES, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
model.config.use_cache = False
model.eval()

text = tok("Hello, what is 2+2?", return_tensors="pt").input_ids
labels = text.clone()
print("input shape:", tuple(text.shape), flush=True)
with torch.amp.autocast("cpu", dtype=torch.bfloat16):
    out = model(input_ids=text, labels=labels)
    loss = out.loss
print("loss:", float(loss) if loss is not None else None,
      "isnan:", bool(torch.isnan(loss).item()) if loss is not None else None,
      "isinf:", bool(torch.isinf(loss).item()) if loss is not None else None, flush=True)
print("DONE steptest", flush=True)
