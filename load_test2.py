import json, struct, os, mmap, torch, psutil
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
SF = os.path.join(SNAP, "model.safetensors")

DT = {"BOOL": torch.bool, "F16": torch.float16, "BF16": torch.bfloat16,
      "F32": torch.float32, "FLOAT16": torch.float16, "BFLOAT16": torch.bfloat16,
      "U8": torch.uint8, "I64": torch.int64, "I32": torch.int32, "F64": torch.float64}
ESZ = {"BOOL": 1, "F16": 2, "BF16": 2, "F32": 4, "FLOAT16": 2, "BFLOAT16": 2,
       "U8": 1, "I64": 8, "I32": 4, "F64": 8}


def rss_mb():
    return round(psutil.Process().memory_info().rss / 1e6, 1)


print("reading header...", flush=True)
with open(SF, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
print("header tensors:", len(hdr), flush=True)

cfg = AutoConfig.from_pretrained(SNAP)
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg)
m.tie_weights()
print("meta skeleton built; rss MB:", rss_mb(), flush=True)

fh = open(SF, "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
print("file mmap'd read-only (commit ~0 for weights); rss MB:", rss_mb(), flush=True)

for name, info in hdr.items():
    if "data_offsets" not in info:
        continue  # __metadata__ or non-tensor entry
    parts = name.split(".")
    parent = m.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else m
    leaf = parts[-1]
    start, end = info["data_offsets"]
    esz = ESZ[info["dtype"]]
    count = (end - start) // esz
    t = torch.frombuffer(mm, dtype=DT[info["dtype"]], count=count, offset=start).reshape(info["shape"])
    try:
        setattr(parent, leaf, torch.nn.Parameter(t, requires_grad=False))
    except (AttributeError, TypeError):
        print("  skip (buffer/other):", name, flush=True)

print("weights attached; rss MB:", rss_mb(), "params:", m.num_parameters(), flush=True)

ids = torch.randint(0, cfg.vocab_size, (2, 8))
m.eval()
with torch.no_grad():
    out = m(ids)
print("forward ok logits:", tuple(out.logits.shape), "rss MB:", rss_mb(), flush=True)
