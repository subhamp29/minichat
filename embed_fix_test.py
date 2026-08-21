import torch, types, json, struct, os
torch.set_num_threads(4)
import train_lora as T

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAP, "model.safetensors")

m = T.load_base_model_cpu(types.SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit=False))

with open(SF, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
info = hdr["model.embed_tokens.weight"]
start, end = info["data_offsets"]
with open(SF, "rb") as f:
    f.seek(start)
    raw = f.read(end - start)
emb = torch.frombuffer(raw, dtype=torch.bfloat16, count=(end - start) // 2).reshape(info["shape"])
emb_w = torch.empty_like(emb)
emb_w.copy_(emb)  # writable RAM copy (272 MB)
m.model.embed_tokens.weight = torch.nn.Parameter(emb_w, requires_grad=False)
m.lm_head.weight = m.model.embed_tokens.weight  # re-tie to the writable copy

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True, trust_remote_code=True)
ids = tok("hello world", return_tensors="pt").input_ids
m.eval()
with torch.no_grad():
    out = m(ids)
print("FIXED logits: max", float(out.logits.max()), "min", float(out.logits.min()),
      "argmax", int(out.logits.argmax(-1)[0, 0]), flush=True)
print("DONE", flush=True)
