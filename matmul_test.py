import torch, types, json, struct, os
import numpy as np
import train_lora as T

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAP, "model.safetensors")

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
m.eval()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True, trust_remote_code=True)
ids = tok("hello world", return_tensors="pt").input_ids
with torch.no_grad():
    out = m(ids, output_hidden_states=True)
h = out.hidden_states[-1]  # (1, len, 896) bf16 activation
print("hidden shape", tuple(h.shape), "norm", float(h.float().norm()),
      "max", float(h.float().abs().max()), flush=True)

# Read embed weights from FILE into a writable RAM tensor (NOT via mmap param).
with open(SF, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
info = hdr["model.embed_tokens.weight"]
start, end = info["data_offsets"]
with open(SF, "rb") as f:
    f.seek(start)
    raw = bytearray(f.read(end - start))
emb_bf16 = torch.frombuffer(raw, dtype=torch.bfloat16, count=(end - start) // 2).reshape(info["shape"])
print("embed ram-bf16 max", float(emb_bf16.float().abs().max()), "mean", float(emb_bf16.float().abs().mean()), flush=True)

# bf16 matmul (mirrors what the model does)
lg_bf16 = h @ emb_bf16.T
print("BF16 logits max", float(lg_bf16.max()), "min", float(lg_bf16.min()), "nan", bool(torch.isnan(lg_bf16.any())), flush=True)

# fp32 matmul (same weights, upcast)
lg_fp32 = h.float() @ emb_bf16.float().T
print("FP32 logits max", float(lg_fp32.max()), "min", float(lg_fp32.min()), flush=True)
print("DONE", flush=True)
