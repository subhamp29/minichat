import torch, types, json, struct, os, numpy as np, psutil
torch.set_num_threads(4)
import train_lora as T

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAP, "model.safetensors")

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))
m.config.output_hidden_states = True
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True, trust_remote_code=True)
ids = tok("hello world", return_tensors="pt").input_ids
with torch.no_grad():
    out = m(ids)
h = out.hidden_states[-1]  # real RAM activation
print("hidden norm", float(h.float().norm()), "max", float(h.float().abs().max()), flush=True)

w_mmap = m.lm_head.weight
lg_mmap = h @ w_mmap.T
print("lg_mmap (non-writable) max", float(lg_mmap.max()), "min", float(lg_mmap.min()), flush=True)

# writable RAM copy read from file
with open(SF, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
info = hdr["model.embed_tokens.weight"]
start, end = info["data_offsets"]
with open(SF, "rb") as f:
    f.seek(start)
    raw = f.read(end - start)
arr = np.array(np.frombuffer(raw, dtype=np.float16))  # writable copy (fp16 bytes == bf16 bytes)
w_ram = torch.from_numpy(arr).reshape(info["shape"]).to(torch.bfloat16)  # writable bf16 RAM

lg_ram = h @ w_ram.T
print("lg_ram (writable bf16) max", float(lg_ram.max()), "min", float(lg_ram.min()), flush=True)
print("lg_mmap (non-writable) vs lg_ram (writable) allclose:",
      bool(torch.allclose(lg_mmap, lg_ram, atol=1.0)), flush=True)
print("DONE", flush=True)
