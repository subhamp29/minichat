import os, logging, gc
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
ST_PATH = os.path.join(SNAP, "model.safetensors")

cfg = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(SNAP)
tok.pad_token = tok.eos_token

# Build empty model in bf16 (meta tensors)
model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
model.config.use_cache = False

# Load ALL weights into WRITABLE RAM (safetensors load_file materializes in RAM, no mmap)
print("[info] loading safetensors into writable RAM (no mmap) as bf16", flush=True)
state = load_file(ST_PATH)  # dict[str, Tensor]; tensors live in writable memory

emb_weight = state.get("model.embed_tokens.weight", None)
if emb_weight is not None:
    emb_weight = emb_weight.to(torch.bfloat16).clone()
    print("embed weight writable:", True, "dtype", emb_weight.dtype, "shape", tuple(emb_weight.shape))

# Attach weights, writing into model.parameters() (which are meta tensors from from_config)
def set_param(obj, name, tensor):
    parent = obj
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))

set_count = 0
for name, t in state.items():
    t = t.to(torch.bfloat16)
    # ensure writable: clone so storage is owned (load_file tensors are already writable, but be safe)
    if t.is_floating_point():
        t = t.clone()
    try:
        set_param(model, name, t)
        set_count += 1
    except (AttributeError, TypeError, KeyError):
        pass
# Tie lm_head to embed_tokens (Qwen2 stores the weight under model.embed_tokens.weight)
model.tie_weights()
print(f"[info] attached {set_count} writable bf16 tensors; params={model.num_parameters()}", flush=True)

model.eval()
gc.collect()
ids = tok("Hello, what is 2+2?", return_tensors="pt").input_ids
print("lm_head.weight device:", model.lm_head.weight.device, "dtype", model.lm_head.weight.dtype)
with torch.no_grad():
    out = model(ids)
logits = out.logits
print("logits shape:", tuple(logits.shape))
print("logits max:", float(logits.float().max()), "min:", float(logits.float().min()), "mean:", float(logits.float().mean()))
print("argmax last:", int(logits[0,-1].argmax()))
print("DONE writable-ram bf16")
