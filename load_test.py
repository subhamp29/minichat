import json, struct, os, psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAP, "model.safetensors")

DT = {"BOOL": torch.bool, "F16": torch.float16, "BF16": torch.bfloat16,
      "F32": torch.float32, "U8": torch.uint8, "I64": torch.int64,
      "I32": torch.int32, "F64": torch.float64}

print("reading safetensors header (no mmap)...", flush=True)
with open(SF, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))

print("header tensors:", len(hdr), flush=True)

cfg = AutoConfig.from_pretrained(SNAP)
torch.set_default_dtype(torch.bfloat16)
m = AutoModelForCausalLM.from_config(cfg)
m.tie_weights()
sd = m.state_dict()
torch.set_default_dtype(torch.float32)
print("model built; params:", m.num_parameters(), "rss MB:", round(psutil.Process().memory_info().rss / 1e6, 2), flush=True)


def resolve(obj, path):
    for part in path.split("."):
        obj = obj[int(part)] if part.lstrip("-").isdigit() else getattr(obj, part)
    return obj


print("loading weights one tensor at a time via f.read()...", flush=True)
with open(SF, "rb") as f:
    for name, info in hdr.items():
        if name not in sd:
            continue
        start, end = info["data_offsets"]
        f.seek(start)
        t = torch.frombuffer(f.read(end - start), dtype=DT[info["dtype"]]).reshape(info["shape"])
        resolve(m, name).copy_(t)
        del t

print("LOADED ok params:", m.num_parameters(), "rss MB:", round(psutil.Process().memory_info().rss / 1e6, 2), flush=True)

ids = torch.randint(0, getattr(cfg, "vocab_size", 151936), (1, 8))
with torch.no_grad():
    out = m(ids)
print("forward ok logits:", tuple(out.logits.shape), "rss MB:", round(psutil.Process().memory_info().rss / 1e6, 2), flush=True)
