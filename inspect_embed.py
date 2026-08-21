import os, struct, json, mmap, logging, sys
logging.basicConfig(level=logging.WARNING)
import torch
torch.set_num_threads(4)

SNAP = r"C:\Users\siddhartha mahapatra\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
ST_PATH = os.path.join(SNAP, "model.safetensors")
from train_lora import _SF_DT

def p(*a):
    print(*a, flush=True)

with open(ST_PATH, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len).decode("utf-8"))

fh = open(ST_PATH, "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

# small known-good tensor
k = "model.layers.0.self_attn.q_proj.weight"
info = header[k]
start, end = info["data_offsets"]
dt, esz = _SF_DT[info["dtype"]]
n = (end - start) // esz
t = torch.frombuffer(mm, dtype=dt, count=n, offset=start).reshape(info["shape"])
p(f"q_proj dtype={t.dtype} shape={tuple(t.shape)} min={float(t.min())} max={float(t.max())} mean={float(t.mean())} std={float(t.std())}")

# embed
k = "model.embed_tokens.weight"
info = header[k]
p(f"{k} dtype={info['dtype']} shape={info['shape']}")
start, end = info["data_offsets"]
dt, esz = _SF_DT[info["dtype"]]
n = (end - start) // esz
p(f"data_offsets={start},{end} n={n} esz={esz} expected_bytes={n*esz}")
t = torch.frombuffer(mm, dtype=dt, count=n, offset=start).reshape(info["shape"])
p(f"embed torch_dtype={t.dtype} shape={tuple(t.shape)}")
p(f"embed min={float(t.min())} max={float(t.max())} mean={float(t.mean())}")
p(f"embed row0[:8]={[round(x,4) for x in t[0,:8].tolist()]}")
p(f"embed abs>100 count={int((t.float().abs()>100).sum()) if True else 'skip'}")
p("DONE")
