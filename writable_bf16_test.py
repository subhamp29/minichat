import os, gc, logging
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from accelerate import init_empty_weights

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
ST_PATH = os.path.join(SNAP, "model.safetensors")

cfg = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(SNAP)
tok.pad_token = tok.eos_token

# Build model as META tensors (zero committed memory)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

def set_param(obj, name, tensor):
    parts = name.split(".")
    parent = obj
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))

# Stream safetensors one tensor at a time; clone to writable bf16 RAM.
# Peak commit ~ 494 MB (bf16 model) + overhead, well under 1.8 GB.
set_count = 0
skip = 0
with safe_open(ST_PATH, framework="torch") as f:
    for key in f.keys():
        if key == "metadata":
            continue
        t = f.get_tensor(key)  # file-backed read tensor
        # convert to bf16 and CLONE to writable, owned RAM storage
        t = t.to(torch.bfloat16).clone()
        try:
            set_param(model, key, t)
            set_count += 1
        except (AttributeError, TypeError, KeyError):
            skip += 1
        del t
model.tie_weights()  # tie lm_head -> embed_tokens
print(f"[info] attached {set_count} writable bf16 RAM tensors (skipped {skip}); params={model.num_parameters()}", flush=True)

# Verify lm_head.weight is writable & contiguous, not file-backed
lmw = model.lm_head.weight
print("lm_head.weight writable:", not lmw.is_storage_shared() if hasattr(lmw, "is_storage_shared") else "n/a",
      "device:", lmw.device, "dtype:", lmw.dtype, "shape:", tuple(lmw.shape), flush=True)

model.eval()
gc.collect()
ids = tok("Hello, what is 2+2?", return_tensors="pt").input_ids
with torch.no_grad():
    out = model(ids)
logits = out.logits.float()
print("logits shape:", tuple(logits.shape))
print("logits max:", float(logits.max()), "min:", float(logits.min()), "mean:", float(logits.mean()))
print("argmax last:", int(logits[0,-1].argmax()))
print("DONE writable-ram bf16 (streaming)")
