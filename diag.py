import json, struct, os, mmap, torch
from pathlib import Path
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download

_SF_DT = {"F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2),
          "F32": (torch.float32, 4), "BOOL": (torch.bool, 1),
          "U8": (torch.uint8, 1), "I64": (torch.int64, 8),
          "I32": (torch.int32, 4), "I16": (torch.int16, 2),
          "I8": (torch.int8, 1), "F64": (torch.float64, 8)}

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAPSHOT_DIR := SNAP, "model.safetensors")
local_dir = Path(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.safetensors"]))
sf_files = sorted(local_dir.glob("*.safetensors"), key=lambda p: p.stat().st_size, reverse=True)
SF = str(sf_files[0])

with open(SF, "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hl).decode("utf-8"))
print("header tensor count:", len([k for k, v in hdr.items() if "data_offsets" in v]))

cfg = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
m.tie_weights()

fh = open(SF, "rb"); mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
for name, info in hdr.items():
    if "data_offsets" not in info:
        continue
    dt, esz = _SF_DT[info["dtype"]]; start, end = info["data_offsets"]
    t = torch.frombuffer(mm, dtype=dt, count=(end - start) // esz, offset=start).reshape(info["shape"])
    parts = name.split(".")
    parent = m.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else m
    setattr(parent, parts[-1], torch.nn.Parameter(t, requires_grad=False))

model_param_names = set(n for n, _ in m.named_parameters())
header_names = set(k for k, v in hdr.items() if "data_offsets" in v)
missing = model_param_names - header_names
extra = header_names - model_param_names
print("params NOT in header (would stay meta):")
for n in sorted(missing):
    p = m.get_submodule(n.rsplit(".", 1)[0]) if "." in n else m
    # find param obj
    leaf = n.rsplit(".", 1)[-1]
    try:
        obj = m.get_submodule(n.rsplit(".", 1)[0])
        p = getattr(obj, leaf)
        print(f"  {n}  device={p.device}  is_meta={p.is_meta}")
    except Exception as e:
        print(f"  {n}  <resolve error {e}>")
print("header NOT in model:", sorted(extra)[:10], "..." if len(extra) > 10 else "")
print("meta params remaining:")
for n, p in m.named_parameters():
    if p.is_meta:
        print("  META:", n)
